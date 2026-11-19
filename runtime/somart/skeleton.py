"""Skeleton hierarchy and BVH clip loading (NumPy port of the upstream Warp loader)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import math3d as m3


@dataclass
class Skeleton:
    """Joint hierarchy + reference (bind) pose as local transforms [J, 7]."""

    names: list[str]
    parents: np.ndarray
    reference: np.ndarray
    name: str = "skeleton"
    up_axis: str = "y"
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.parents = np.asarray(self.parents, dtype=np.int32)
        self.reference = np.asarray(self.reference, dtype=np.float64)
        if len(self.names) != len(self.parents) or self.reference.shape != (len(self.names), 7):
            raise ValueError("Skeleton names, parents and reference must have matching length.")
        if self.parents[0] != -1 or any(self.parents[j] >= j for j in range(1, len(self.parents))):
            raise ValueError("Skeleton must be topologically sorted with the root at index 0.")
        self._index = {n: i for i, n in enumerate(self.names)}
        self._levels = m3.depth_levels(self.parents)

    @property
    def num_joints(self) -> int:
        return len(self.names)

    def index(self, name: str) -> int:
        return self._index.get(name, -1)

    def fk(self, local: np.ndarray) -> np.ndarray:
        return m3.forward_kinematics(local, self.parents, levels=self._levels)

    def subset(self, keep: list[str]) -> "Skeleton":
        """Return a skeleton containing only `keep` joints (re-parented to nearest kept ancestor).

        Local transforms of removed ancestors are folded into the reference pose of the child.
        """
        keep_idx = [self.index(n) for n in keep]
        if -1 in keep_idx:
            raise ValueError("unknown joints: " + ", ".join(n for n, i in zip(keep, keep_idx) if i < 0))
        world = self.fk(self.reference)
        new_parents, new_ref = [], []
        for n, j in zip(keep, keep_idx):
            p = self.parents[j]
            while p >= 0 and self.names[p] not in keep:
                p = self.parents[p]
            new_parents.append(keep.index(self.names[p]) if p >= 0 else -1)
            new_ref.append(world[j] if p < 0 else m3.tf_mul(m3.tf_inv(world[p]), world[j]))
        return Skeleton(list(keep), np.array(new_parents), np.array(new_ref), name=self.name + "_subset")

    def to_dict(self) -> dict:
        return {"name": self.name, "names": self.names, "parents": self.parents.tolist(),
                "reference": self.reference.tolist(), "up_axis": self.up_axis, "meta": self.meta}

    @classmethod
    def from_dict(cls, d: dict) -> "Skeleton":
        return cls(d["names"], np.array(d["parents"]), np.array(d["reference"]), name=d.get("name", "skeleton"),
                   up_axis=d.get("up_axis", "y"), meta=d.get("meta", {}))


@dataclass
class Clip:
    """Local-transform animation [F, J, 7] at `fps`, optional foot contacts [F, 4]."""

    skeleton: Skeleton
    local: np.ndarray
    fps: float
    name: str = "clip"
    contacts: np.ndarray | None = None
    tags: dict = field(default_factory=dict)

    @property
    def num_frames(self) -> int:
        return self.local.shape[0]

    @property
    def duration(self) -> float:
        return (self.num_frames - 1) / self.fps

    def world(self) -> np.ndarray:
        return self.skeleton.fk(self.local)

    def sample(self, t: np.ndarray) -> np.ndarray:
        """Sample local poses at times `t` (s) with lerp/slerp; times are clamped."""
        t = np.clip(np.asarray(t, dtype=np.float64), 0.0, self.duration)
        f = t * self.fps
        i0 = np.floor(f).astype(int)
        i1 = np.minimum(i0 + 1, self.num_frames - 1)
        w = (f - i0)[..., None]  # [T, 1] -> broadcasts over joints
        return m3.tf_blend(self.local[i0], self.local[i1], w)

    def resample(self, fps: float) -> "Clip":
        if abs(fps - self.fps) < 1e-6:
            return self
        n = int(np.floor(self.duration * fps + 1e-6)) + 1
        t = np.arange(n) / fps
        ratio = self.fps / fps
        if abs(ratio - round(ratio)) < 1e-6:  # exact integer decimation, no interpolation error
            local = self.local[:: int(round(ratio))][:n]
        else:
            local = self.sample(t)
        return Clip(self.skeleton, local, fps, self.name, None, dict(self.tags))

    def slice(self, start: int, stop: int) -> "Clip":
        c = None if self.contacts is None else self.contacts[start:stop]
        return Clip(self.skeleton, self.local[start:stop], self.fps, self.name, c, dict(self.tags))


# ------------------------------------------------------------------------------ BVH
_AXIS = {"x": np.array([1.0, 0, 0]), "y": np.array([0, 1.0, 0]), "z": np.array([0, 0, 1.0])}


def _euler_to_quat(angles_deg: np.ndarray, order: list[str]) -> np.ndarray:
    """Intrinsic composition in file order (q = q_a * q_b * q_c), same as upstream."""
    q = m3.quat_identity(angles_deg.shape[:-1])
    for k, axis in enumerate(order):
        q = m3.quat_mul(q, m3.quat_from_axis_angle(_AXIS[axis], np.radians(angles_deg[..., k])))
    return m3.normalize(q)


def load_bvh(path: str | Path, *, drop_static_root: str | None = "Root", units: float = 0.01) -> Clip:
    """Load a BVH clip (cm -> m by default). Optionally drop a static wrapper root joint.

    SOMA clips have an identity, non-animated ``Root`` above ``Hips``; upstream maps clips onto
    a skeleton without it. If the wrapper root *is* animated, its transform is folded into the
    child so no motion is lost.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    head, _, motion = text.partition("MOTION")
    names, parents, offsets, channels = [], [], [], []
    stack: list[int] = []
    in_end = False
    for line in head.splitlines():
        tok = line.split()
        if not tok:
            continue
        if tok[0] in ("ROOT", "JOINT"):
            names.append(tok[1].rsplit(":", 1)[-1])
            parents.append(stack[-1] if stack else -1)
            stack.append(len(names) - 1)
            offsets.append([0.0, 0.0, 0.0])
            channels.append([])
        elif tok[:2] == ["End", "Site"]:
            in_end = True
        elif tok[0] == "OFFSET" and not in_end:
            offsets[stack[-1]] = [float(v) for v in tok[1:4]]
        elif tok[0] == "CHANNELS":
            channels[stack[-1]] = tok[2:]
        elif tok[0] == "}":
            if in_end:
                in_end = False
            else:
                stack.pop()
    m = re.search(r"Frame Time:\s*([0-9.eE+-]+)", motion)
    if not m:
        raise ValueError(f"BVH frame time missing: {path}")
    fps = round(1.0 / float(m.group(1)), 2)
    body = motion[m.end():]
    nch = sum(len(c) for c in channels)
    data = np.array(body.split(), dtype=np.float64).reshape(-1, nch)
    J, F = len(names), data.shape[0]
    offsets = np.asarray(offsets) * units
    pos = np.broadcast_to(offsets, (F, J, 3)).copy()
    quat = m3.quat_identity((F, J))
    col = 0
    for j, ch in enumerate(channels):
        rot_cols, rot_order = [], []
        for c in ch:
            c = c.lower()
            if c.endswith("position"):
                pos[:, j, "xyz".index(c[0])] = data[:, col] * units
            elif c.endswith("rotation"):
                rot_cols.append(col)
                rot_order.append(c[0])
            col += 1
        if rot_cols:
            quat[:, j] = _euler_to_quat(data[:, rot_cols], rot_order)
    local = np.concatenate([pos, quat], axis=-1)
    reference = np.concatenate([offsets, m3.quat_identity((J,))], axis=-1)
    skel = Skeleton(names, np.array(parents), reference, name=path.stem)

    if drop_static_root and names[0] == drop_static_root and J > 1:
        child = 1
        if parents[child] != 0:
            raise ValueError("wrapper root must have the first joint as child")
        local[:, child] = m3.tf_mul(local[:, 0], local[:, child])
        keep = names[1:]
        new_parents = np.array([-1] + [p - 1 for p in parents[2:]])
        ref = reference[1:].copy()
        ref[0] = m3.tf_mul(reference[0], reference[1])
        skel = Skeleton(keep, new_parents, ref, name="soma_bvh")
        local = local[:, 1:]
    return Clip(skel, local, fps, name=path.stem)
