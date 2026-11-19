"""Clip library and target-pose selection ("motion matching" for MotionBricks targets).

MotionBricks needs *target poses*. For locomotion we pick a 4-frame window from mocap whose
local root velocity matches the desired velocity, then place it in the world at the point
the character's simulated trajectory reaches at the prediction horizon - the same idea as
the upstream Smart Primitives, but driven continuously by a spring-damper trajectory.

Windows are stored canonicalised: root ground position at the origin, facing +Z.
"""
from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import codec
from . import contract as C
from . import math3d as m3
from .skeleton import Clip, Skeleton, load_bvh


@dataclass
class Window:
    clip: str
    frame: int                 # first frame of the 4-frame window
    local: np.ndarray          # [4, J, 7] canonical (root ground at origin, facing +Z)
    velocity: np.ndarray       # [2] local (x lateral, z forward) root velocity m/s
    yaw_rate: float            # rad/s
    speed: float
    height: float              # hips height (crouch detection)
    phase: float               # left/right foot phase in [0, 1)
    tags: dict = field(default_factory=dict)


def heading_of(world: np.ndarray, sk: Skeleton) -> np.ndarray:
    r, l = sk.index(C.HEADING_JOINTS[0]), sk.index(C.HEADING_JOINTS[1])
    return codec.heading_from_hips(world[..., r, :3], world[..., l, :3])


def place(window_local: np.ndarray, position: np.ndarray, yaw: float) -> np.ndarray:
    """Move a canonical window so its root ground point is at `position` and it faces `yaw`."""
    T = m3.tf(np.array([position[0], 0.0, position[2]]), m3.quat_from_yaw(np.array(yaw)))
    out = window_local.copy()
    out[:, 0] = m3.tf_mul(T, window_local[:, 0])
    return out


def canonicalize(local: np.ndarray, sk: Skeleton, ref: int = 0) -> tuple[np.ndarray, float, np.ndarray]:
    """Return (canonical poses, original yaw, original ground position) using frame `ref` as anchor."""
    w = sk.fk(local[ref:ref + 1])[0]
    yaw = float(heading_of(w, sk))
    p = local[ref, 0, :3] * np.array([1.0, 0.0, 1.0])
    inv = m3.tf_inv(m3.tf(p, m3.quat_from_yaw(np.array(yaw))))
    out = local.copy()
    out[:, 0] = m3.tf_mul(inv, local[:, 0])
    return out, yaw, p


class ClipLibrary:
    def __init__(self, root: str | Path, skeleton: Skeleton | None = None, fps: float = C.SAMPLE_RATE) -> None:
        self.root = Path(root)
        self.fps = fps
        self.clips: dict[str, Clip] = {}
        self.skeleton = skeleton
        self.windows: list[Window] = []
        self._feat: np.ndarray | None = None

    # ------------------------------------------------------------------ loading
    def load(self, pattern: str = "**/*.bvh") -> "ClipLibrary":
        for f in sorted(self.root.glob(pattern)):
            rel = str(f.relative_to(self.root))
            if rel in self.clips:
                continue
            clip = load_bvh(f).resample(self.fps)
            if self.skeleton is None:
                self.skeleton = clip.skeleton
            elif clip.skeleton.names != self.skeleton.names:
                continue
            parts = rel.split("/")
            clip.tags = {"category": parts[1] if parts[0] == "locomotion" else parts[0], "path": rel}
            self.clips[rel] = clip
        return self

    def add_clip(self, name: str, clip: Clip, tags: dict | None = None) -> None:
        """Register any user clip (e.g. sprint, roll, climb) on the same skeleton."""
        if self.skeleton is not None and clip.skeleton.names != self.skeleton.names:
            raise ValueError("clip skeleton differs from library skeleton; retarget it first")
        clip = clip.resample(self.fps)
        clip.tags = dict(tags or {}, path=name)
        self.clips[name] = clip
        self._feat = None

    def find(self, source: str) -> list[str]:
        return [k for k in self.clips if k == source or k.startswith(source.rstrip("/") + "/")
                or fnmatch.fnmatch(k, source)]

    # ------------------------------------------------------------ window index
    def index_windows(self, sources: list[str] | None = None, stride: int = 2) -> None:
        sk = self.skeleton
        lf, rf = sk.index("LeftFoot"), sk.index("RightFoot")
        names = [n for s in sources for n in self.find(s)] if sources else list(self.clips)
        self.windows = []
        for name in dict.fromkeys(names):
            clip = self.clips[name]
            w_all = clip.world()
            head = heading_of(w_all, sk)
            hips = w_all[:, 0, :3]
            # foot phase from the relative forward offset of the feet (sign changes once per step)
            for f in range(1, clip.num_frames - C.POSES_PER_TOKEN - 1, stride):
                canon, yaw, _ = canonicalize(clip.local[f:f + C.POSES_PER_TOKEN], sk, ref=0)
                vel_w = (hips[f + 1] - hips[f - 1]) * self.fps / 2
                fwd = np.array([np.sin(yaw), np.cos(yaw)])     # heading_from_hips: atan2(x, z) convention
                right = np.array([fwd[1], -fwd[0]])
                v = np.array([vel_w[[0, 2]] @ right, vel_w[[0, 2]] @ fwd])
                yr = float(m3.wrap_angle(head[f + 1] - head[f - 1]) * self.fps / 2)
                d = (w_all[f, lf, [0, 2]] - w_all[f, rf, [0, 2]]) @ fwd
                ph = 0.0 if d > 0 else 0.5
                self.windows.append(Window(name, f, canon, v, yr, float(np.linalg.norm(v)), float(hips[f, 1]), ph,
                                           dict(clip.tags)))
        self._feat = np.array([[w.velocity[0], w.velocity[1], w.yaw_rate, w.height] for w in self.windows])

    # ---------------------------------------------------------------- matching
    def match(self, velocity_local: np.ndarray, yaw_rate: float = 0.0, *, crouch: bool | None = None,
              category: str | None = None, variation: float = 0.0, rng: np.random.Generator | None = None,
              weights=(1.0, 1.0, 0.15, 2.0)) -> Window:
        if self._feat is None or not len(self.windows):
            raise RuntimeError("call index_windows() first")
        f = self._feat
        cost = (weights[0] * (f[:, 0] - velocity_local[0]) ** 2 + weights[1] * (f[:, 1] - velocity_local[1]) ** 2
                + weights[2] * (f[:, 2] - yaw_rate) ** 2)
        if crouch is not None:
            is_crouch = f[:, 3] < 0.75
            cost = cost + np.where(is_crouch == crouch, 0.0, 1e3)
        if category is not None:
            cost = cost + np.array([0.0 if w.tags.get("category") == category else 1e3 for w in self.windows])
        if variation > 0:
            rng = rng or np.random.default_rng()
            # pick uniformly among windows within a tolerance band of the best cost
            band = np.min(cost) + variation * 0.25 + 1e-9
            cand = np.flatnonzero(cost <= band)
            return self.windows[int(rng.choice(cand))]
        return self.windows[int(np.argmin(cost))]

    def window_at(self, name: str, frame: int) -> Window:
        clip = self.clips[name]
        frame = int(np.clip(frame, 0, clip.num_frames - C.POSES_PER_TOKEN))
        canon, _, _ = canonicalize(clip.local[frame:frame + C.POSES_PER_TOKEN], self.skeleton)
        return Window(name, frame, canon, np.zeros(2), 0.0, 0.0, float(clip.local[frame, 0, 1]), 0.0, dict(clip.tags))
