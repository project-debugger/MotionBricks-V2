"""Anatomical clean-up learned from mocap + procedural fingers.

Why: in the SOMA training data (Bones RP1, 30 Hz mocap) the knees and elbows are pure hinges, the
toes have 1-2 axes and the jaw never moves. The network predicts *global* rotations independently,
so decoded local rotations carry small twists and off-axis bends there (measured: elbows up to 17 deg,
knees ~6 deg, toes ~7 deg, jaw ~3 deg at the 95th percentile). Skinned onto a body mesh those show as
twisted forearms and bowed shins. :class:`JointLimits` projects each constrained joint back onto the
axes the mocap actually uses, clamps it to the observed range, and gives the child joint the
difference so hands and feet keep exactly the orientation the model predicted.

Fingers: the training data has *no* finger motion (one fixed relaxed hand pose), so the model cannot
animate them. :class:`FingerAnimator` adds the standard game technique instead: the relaxed pose plus
procedural curl that tightens with speed, follows through when the wrist swings, and varies slightly
per finger. It is clearly a procedural layer, not model output.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import contract as C
from . import math3d as m3
from .skeleton import Skeleton

FINGERS = ("Thumb", "Index", "Middle", "Ring", "Pinky")


# ----------------------------------------------------------------------------- joint limits
@dataclass
class JointLimit:
    joint: str
    rank: int                         # number of rotation axes used by the mocap (0..2)
    axes: np.ndarray                  # [rank, 3] orthonormal rotation-vector basis (local frame)
    lo: np.ndarray                    # [rank] min angle (rad) along each axis
    hi: np.ndarray                    # [rank] max angle (rad)
    rest: np.ndarray                  # [4] mean rotation (used for rank 0)


@dataclass
class JointLimits:
    limits: list[JointLimit]
    relaxed: dict[str, list[float]] = field(default_factory=dict)  # unmapped joint -> constant quat

    # ------------------------------------------------------------------ fitting / io
    @classmethod
    def fit(cls, local: np.ndarray, sk: Skeleton, joints=C.JOINT_NAMES[1:], margin_deg: float = 3.0,
            rank_tol: float = 1e-3) -> "JointLimits":
        """Learn constraints from mocap local poses [F, J, 7] on skeleton `sk`."""
        lims = []
        for name in joints:
            j = sk.index(name)
            r = m3.quat_to_scaled_axis(m3.quat_abs(local[:, j, 3:]))
            sv = np.linalg.svd(r - r.mean(0), compute_uv=False)
            if sv[0] < 1e-6 and np.linalg.norm(r.mean(0)) < 1e-6:
                rank = 0
            else:
                rank = int((sv > rank_tol * max(sv[0], 1e-9)).sum())
            if rank >= 3:
                continue
            basis = np.linalg.svd(r, full_matrices=False)[2][:max(rank, 0)] if rank else np.zeros((0, 3))
            coords = r @ basis.T if rank else np.zeros((len(r), 0))
            m = math.radians(margin_deg)
            lo = np.percentile(coords, 0.2, axis=0) - m if rank else np.zeros(0)
            hi = np.percentile(coords, 99.8, axis=0) + m if rank else np.zeros(0)
            rest = m3.quat_from_scaled_axis(r.mean(0))
            lims.append(JointLimit(name, rank, basis, lo, hi, rest))
        mapped = set(C.JOINT_NAMES)
        relaxed = {}
        for j, name in enumerate(sk.names):
            if name in mapped:
                continue
            r = m3.quat_to_scaled_axis(m3.quat_abs(local[:, j, 3:]))
            if np.degrees(np.abs(r - r.mean(0)).max()) < 2.0:      # constant in the data
                relaxed[name] = m3.quat_from_scaled_axis(r.mean(0)).tolist()
        return cls(lims, relaxed)

    def to_dict(self) -> dict:
        return {"limits": [{"joint": l.joint, "rank": l.rank, "axes": l.axes.tolist(), "lo": l.lo.tolist(),
                            "hi": l.hi.tolist(), "rest": l.rest.tolist()} for l in self.limits],
                "relaxed": self.relaxed,
                "note": "Learned from SOMA mocap (axes, angle ranges, constant relaxed finger pose)."}

    @classmethod
    def from_dict(cls, d: dict) -> "JointLimits":
        return cls([JointLimit(x["joint"], x["rank"], np.array(x["axes"]).reshape(-1, 3), np.array(x["lo"]),
                               np.array(x["hi"]), np.array(x["rest"])) for x in d["limits"]], d.get("relaxed", {}))

    @classmethod
    def load(cls, path: str | Path) -> "JointLimits":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))

    # ------------------------------------------------------------------ apply
    def bind(self, sk: Skeleton) -> "BoundLimits":
        return BoundLimits(self, sk)


class BoundLimits:
    def __init__(self, lim: JointLimits, sk: Skeleton) -> None:
        self.lim, self.sk = lim, sk
        self.items = []
        for l in lim.limits:
            j = sk.index(l.joint)
            if j < 0:
                continue
            children = [c for c in range(sk.num_joints) if sk.parents[c] == j]
            self.items.append((j, children, l))
        self.relaxed = [(sk.index(n), np.array(q)) for n, q in lim.relaxed.items() if sk.index(n) >= 0]

    def apply(self, local: np.ndarray, relax_unmapped: bool = False) -> np.ndarray:
        """local [..., J, 7] -> constrained copy; children keep their world rotations."""
        out = local.copy()
        for j, children, l in self.items:
            q_old = m3.quat_abs(out[..., j, 3:])
            if l.rank == 0:
                q_new = np.broadcast_to(l.rest, q_old.shape).copy()
            else:
                r = m3.quat_to_scaled_axis(q_old)
                c = np.clip(r @ l.axes.T, l.lo, l.hi)
                q_new = m3.quat_from_scaled_axis(c @ l.axes)
            # child_new = inv(q_new) * q_old * child  => child world rotation unchanged
            delta = m3.quat_mul(m3.quat_conj(q_new), q_old)
            out[..., j, 3:] = q_new
            for c_ in children:
                out[..., c_, 3:] = m3.normalize(m3.quat_mul(delta, out[..., c_, 3:]))
        if relax_unmapped:
            for j, q in self.relaxed:
                out[..., j, 3:] = q
        return out


# ----------------------------------------------------------------------------- fingers
@dataclass
class FingerJoint:
    index: int
    axis: np.ndarray      # local curl axis (unit)
    weight: float         # share of the finger's curl at this joint
    finger: int           # 0..4
    hand: int             # 0 left, 1 right


class FingerAnimator:
    """Procedural finger layer: relaxed pose + speed-dependent curl + follow-through + variation."""

    def __init__(self, sk: Skeleton, relaxed_pose: np.ndarray, *, curl_idle=6.0, curl_walk=12.0, curl_run=32.0,
                 follow=10.0, variation=3.0, seed=0) -> None:
        self.sk = sk
        self.curl_idle, self.curl_walk, self.curl_run = curl_idle, curl_walk, curl_run
        self.follow, self.variation = follow, variation
        self.joints: list[FingerJoint] = []
        world = sk.fk(relaxed_pose)
        for hand, side in enumerate(("Left", "Right")):
            h = sk.index(f"{side}Hand")
            m1, mt = sk.index(f"{side}HandMiddle1"), sk.index(f"{side}HandMiddleEnd")
            if min(h, m1, mt) < 0:
                continue
            # palm-side normal: where the relaxed (slightly curled) middle finger bends towards
            base_dir = m3.normalize(world[m1, :3] - world[h, :3])
            tip = world[mt, :3] - world[m1, :3]
            palm = m3.normalize(tip - (tip @ base_dir) * base_dir)
            for f, name in enumerate(FINGERS):
                chain = [sk.index(f"{side}Hand{name}{k}") for k in (1, 2, 3, 4)]
                chain = [c for c in chain if c >= 0]
                weights = ([0.35, 0.55, 0.5] if name == "Thumb" else [0.15, 1.0, 1.0, 0.75])[:len(chain)]
                for c, wgt in zip(chain, weights):
                    kids = [k for k in range(sk.num_joints) if sk.parents[k] == c]
                    if not kids:
                        continue
                    b = m3.normalize(world[kids[0], :3] - world[c, :3])
                    n = palm if name != "Thumb" else m3.normalize(palm + 0.6 * base_dir)
                    a_world = m3.normalize(np.cross(b, n))          # +angle moves the bone towards the palm
                    a_local = m3.quat_rotate_inv(world[c, 3:], a_world)
                    self.joints.append(FingerJoint(c, a_local, wgt, f, hand))
        self._idx = np.array([fj.index for fj in self.joints], dtype=np.int64)
        self._axis = np.array([fj.axis for fj in self.joints]).reshape(-1, 3)
        self._w = np.array([fj.weight for fj in self.joints])
        self._f = np.array([fj.finger for fj in self.joints], dtype=np.int64)
        self._h = np.array([fj.hand for fj in self.joints], dtype=np.int64)
        self._thumb = np.where(self._f == 0, 0.7, 1.0)
        # root -> hand chains for the follow-through drive (partial FK instead of all 77 joints)
        self._chains = []
        for side in ("Left", "Right"):
            j, chain = sk.index(f"{side}Hand"), []
            while j >= 0:
                chain.append(j)
                j = int(sk.parents[j])
            self._chains.append(chain[::-1])
        rng = np.random.default_rng(seed)
        self.phase = rng.uniform(0, 2 * np.pi, size=(2, 5, 2))
        self.freq = rng.uniform(0.15, 0.45, size=(2, 5, 2))
        self.x = np.zeros((2, 5))      # follow-through spring state per finger (deg)
        self.v = np.zeros((2, 5))
        self.prev_hand = None
        self.t = 0.0

    def apply(self, local: np.ndarray, speed: float, dt: float) -> np.ndarray:
        """Return `local` with finger rotations replaced by the procedural layer (one pose)."""
        self.t += dt
        s = max(0.0, speed)
        if s < 1.6:
            target = self.curl_idle + (self.curl_walk - self.curl_idle) * min(1.0, s / 1.6)
        else:
            u = min(1.0, (s - 1.6) / 2.8)
            target = self.curl_walk + (self.curl_run - self.curl_walk) * (u * u * (3 - 2 * u))
        # follow-through: fingers lag the wrist's angular velocity (critically damped spring)
        if self.follow > 0 and dt > 0:
            world_q = np.stack([self._chain_rot(local, ch) for ch in self._chains])
            drive = np.zeros(2)
            if self.prev_hand is not None:
                dq = m3.quat_mul(world_q, m3.quat_conj(self.prev_hand))
                drive = np.linalg.norm(m3.quat_to_scaled_axis(dq), axis=-1) / dt  # rad/s
            self.prev_hand = world_q
            k, c = 120.0, 2 * math.sqrt(120.0)
            for h in range(2):
                acc = -k * self.x[h] - c * self.v[h] + self.follow * drive[h] * np.array([1.0, 0.8, 1.0, 1.1, 1.2])
                self.v[h] += acc * dt
                self.x[h] += self.v[h] * dt
        ph = self.phase[self._h, self._f]                       # [n, 2]
        fr = self.freq[self._h, self._f]
        noise = self.variation * (0.6 * np.sin(2 * np.pi * fr[:, 0] * self.t + ph[:, 0])
                                  + 0.4 * np.sin(2 * np.pi * fr[:, 1] * self.t + ph[:, 1]))
        curl = (target + self.x[self._h, self._f] + noise) * self._thumb
        rot = m3.quat_from_axis_angle(self._axis, np.radians(curl) * self._w)
        out = local.copy()
        out[self._idx, 3:] = m3.normalize(m3.quat_mul(local[self._idx, 3:], rot))
        return out

    @staticmethod
    def _chain_rot(local: np.ndarray, chain: list[int]) -> np.ndarray:
        q = local[chain[0], 3:]
        for j in chain[1:]:
            q = m3.quat_mul(q, local[j, 3:])
        return q
