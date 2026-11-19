"""Per-frame animation clean-up: inertialization, foot locking (two-bone IK), smoothing, intensity.

All operate on a single pose of local transforms [J, 7] (root in world space) per tick and
cost well under a millisecond in NumPy.
"""
from __future__ import annotations

import math

import numpy as np

from . import math3d as m3
from .skeleton import Skeleton

LN2 = math.log(2.0)


def _damper_decay(x: np.ndarray, v: np.ndarray, halflife: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact critically-damped spring towards zero (Holden, 'Spring-It-On')."""
    y = (4.0 * LN2 / max(halflife, 1e-5)) / 2.0
    j1 = v + x * y
    e = math.exp(-y * dt)
    return e * (x + j1 * dt), e * (v - j1 * y * dt)


class Inertializer:
    """Offset-based inertialization: hides the pop when the playing motion is replaced.

    On ``transition`` it stores the difference between what was on screen and the new motion
    (positions for the root, rotation vectors for every joint) and then decays it to zero with
    a critically damped spring of the configured half-life, preserving velocity continuity.
    """

    def __init__(self, num_joints: int, halflife: float = 0.08) -> None:
        self.halflife = halflife
        self.rot_off = np.zeros((num_joints, 3))
        self.rot_vel = np.zeros((num_joints, 3))
        self.pos_off = np.zeros(3)
        self.pos_vel = np.zeros(3)

    def transition(self, src: np.ndarray, src_prev: np.ndarray, dst: np.ndarray, dst_prev: np.ndarray,
                   dt: float) -> None:
        """src/src_prev: the poses *displayed* this and last tick (offsets already applied);
        dst/dst_prev: the new raw motion at the same two instants."""
        cur, cur_prev = src, src_prev
        r_off = m3.quat_to_scaled_axis(m3.quat_mul(cur[:, 3:], m3.quat_conj(dst[:, 3:])))
        r_off_prev = m3.quat_to_scaled_axis(m3.quat_mul(cur_prev[:, 3:], m3.quat_conj(dst_prev[:, 3:])))
        self.rot_off = r_off
        self.rot_vel = (r_off - r_off_prev) / max(dt, 1e-6)
        self.pos_off = cur[0, :3] - dst[0, :3]
        self.pos_vel = ((cur[0, :3] - cur_prev[0, :3]) - (dst[0, :3] - dst_prev[0, :3])) / max(dt, 1e-6)
        # keep the horizontal root offset small: the root path is owned by the new motion
        self.pos_vel = np.clip(self.pos_vel, -5.0, 5.0)

    def apply(self, pose: np.ndarray, dt: float, decay: bool = True) -> np.ndarray:
        if decay and dt > 0:
            self.rot_off, self.rot_vel = _damper_decay(self.rot_off, self.rot_vel, self.halflife, dt)
            self.pos_off, self.pos_vel = _damper_decay(self.pos_off, self.pos_vel, self.halflife, dt)
        out = pose.copy()
        out[:, 3:] = m3.quat_mul(m3.quat_from_scaled_axis(self.rot_off), pose[:, 3:])
        out[0, :3] = pose[0, :3] + self.pos_off
        return out

    @property
    def active(self) -> bool:
        return float(np.abs(self.rot_off).max()) > 1e-4 or float(np.abs(self.pos_off).max()) > 1e-4


# ---- scalar helpers: per-frame IK works on single joints, where NumPy call overhead (~10 us per
# op on 3/4-element arrays) dominates; plain-float maths is ~20x faster for these sizes.
def _qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz)


def _qconj(q):
    return (-q[0], -q[1], -q[2], q[3])


def _qrot(q, v):
    qx, qy, qz, qw = q
    tx = 2.0 * (qy * v[2] - qz * v[1]); ty = 2.0 * (qz * v[0] - qx * v[2]); tz = 2.0 * (qx * v[1] - qy * v[0])
    return (v[0] + qw * tx + (qy * tz - qz * ty), v[1] + qw * ty + (qz * tx - qx * tz),
            v[2] + qw * tz + (qx * ty - qy * tx))


def _qnorm(q):
    n = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]) or 1.0
    return (q[0] / n, q[1] / n, q[2] / n, q[3] / n)


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _len(a):
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def _unit(a):
    n = _len(a) or 1.0
    return (a[0] / n, a[1] / n, a[2] / n)


def _ang(u, v):
    u, v = _unit(u), _unit(v)
    return math.acos(max(-1.0, min(1.0, u[0] * v[0] + u[1] * v[1] + u[2] * v[2])))


def _axis_angle(axis, angle):
    h = 0.5 * angle
    s_ = math.sin(h)
    return (axis[0] * s_, axis[1] * s_, axis[2] * s_, math.cos(h))


def _tf_mul(a, b):  # a, b: (p3, q4)
    pr = _qrot(a[1], b[0])
    return ((a[0][0] + pr[0], a[0][1] + pr[1], a[0][2] + pr[2]), _qmul(a[1], b[1]))


def two_bone_ik(local: np.ndarray, world: np.ndarray, sk: Skeleton, hip: int, knee: int, ankle: int,
                target: np.ndarray, weight: float = 1.0) -> np.ndarray:
    """Analytic two-bone IK moving `ankle` to `target` (exact when reachable).

    Stage 1 bends hip/knee in the chain plane so |hip->ankle| equals |hip->target|; stage 2
    re-composes the chain and aims hip->ankle at the target. The ankle keeps its world rotation
    so a planted foot stays flat. Only the three chain joints are recomposed (no full FK).
    """
    a, b, c = tuple(world[hip, :3]), tuple(world[knee, :3]), tuple(world[ankle, :3])
    tt = tuple(np.asarray(target, dtype=np.float64))
    t = tuple(c[i] + (tt[i] - c[i]) * weight for i in range(3))
    eps = 1e-5
    lab, lcb = _len(_sub(b, a)), _len(_sub(b, c))
    if lab < eps or lcb < eps:
        return local
    lat = min(max(_len(_sub(t, a)), eps), lab + lcb - eps)
    ac_ab_0, ba_bc_0 = _ang(_sub(c, a), _sub(b, a)), _ang(_sub(a, b), _sub(c, b))
    ac_ab_1 = math.acos(max(-1.0, min(1.0, (lcb * lcb - lab * lab - lat * lat) / (-2 * lab * lat))))
    ba_bc_1 = math.acos(max(-1.0, min(1.0, (lat * lat - lab * lab - lcb * lcb) / (-2 * lab * lcb))))
    a_gr, b_gr = tuple(world[hip, 3:]), tuple(world[knee, 3:])
    axis0 = _cross(_sub(c, a), _sub(b, a))
    if _len(axis0) < 1e-9:  # straight leg: bend about the hip's lateral axis
        axis0 = _qrot(a_gr, (1.0, 0.0, 0.0))
    axis0 = _unit(axis0)
    r0 = _axis_angle(_qrot(_qconj(a_gr), axis0), ac_ab_1 - ac_ab_0)
    r1 = _axis_angle(_qrot(_qconj(b_gr), axis0), ba_bc_1 - ba_bc_0)
    hip_q = _qnorm(_qmul(tuple(local[hip, 3:]), r0))
    knee_q = _qnorm(_qmul(tuple(local[knee, 3:]), r1))
    hp = int(sk.parents[hip])
    parent_w = ((tuple(world[hp, :3]), tuple(world[hp, 3:])) if hp >= 0 else ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)))
    hip_p, knee_p, ankle_p = tuple(local[hip, :3]), tuple(local[knee, :3]), tuple(local[ankle, :3])
    hw = _tf_mul(parent_w, (hip_p, hip_q))
    kw = _tf_mul(hw, (knee_p, knee_q))
    c1 = _tf_mul(kw, (ankle_p, (0.0, 0.0, 0.0, 1.0)))[0]
    axis1 = _cross(_sub(c1, a), _sub(t, a))
    if _len(axis1) > 1e-9:
        r2 = _axis_angle(_qrot(_qconj(hw[1]), _unit(axis1)), _ang(_sub(c1, a), _sub(t, a)))
        hip_q = _qnorm(_qmul(hip_q, r2))
        hw = _tf_mul(parent_w, (hip_p, hip_q))
        kw = _tf_mul(hw, (knee_p, knee_q))
    out = local.copy()
    out[hip, 3:] = hip_q
    out[knee, 3:] = knee_q
    if int(sk.parents[ankle]) == knee:
        out[ankle, 3:] = _qnorm(_qmul(_qconj(kw[1]), tuple(world[ankle, 3:])))
    return out


class FootLocker:
    """Pins feet while the model predicts contact; releases smoothly. Removes residual sliding."""

    def __init__(self, sk: Skeleton, chains: list[list[str]], contact_index: list[int], *, contact_threshold=0.55,
                 release_threshold=0.35, max_correction=0.12, blend_in=0.06, blend_out=0.10, ground=0.0) -> None:
        self.sk = sk
        self.chains = [[sk.index(n) for n in ch] for ch in chains]
        self.contact_index = contact_index
        self.on, self.off = contact_threshold, release_threshold
        self.max_corr, self.bin, self.bout = max_correction, blend_in, blend_out
        self.ground = ground
        self.locked = [False] * len(chains)
        self.lock_pos = [np.zeros(3) for _ in chains]
        self.weight = [0.0] * len(chains)
        # joints whose world transform IK needs: root -> ... -> hip -> knee -> ankle, per chain.
        # Partial FK along these paths replaces a full-skeleton FK (77 joints for the SOMA BVH rig).
        need: list[int] = []
        for hip, knee, ankle in self.chains:
            path, j = [], hip
            while j >= 0:
                path.append(j)
                j = int(sk.parents[j])
            for jj in path[::-1] + [knee, ankle]:
                if jj not in need:
                    need.append(jj)
        self._fk_order = need  # parents always precede children in this order

    def reset(self) -> None:
        self.locked = [False] * len(self.chains)
        self.weight = [0.0] * len(self.chains)

    def _partial_fk(self, local: np.ndarray) -> np.ndarray:
        world = np.zeros_like(local)
        cache = {}
        for j in self._fk_order:
            p = int(self.sk.parents[j])
            lj = (tuple(local[j, :3]), tuple(local[j, 3:]))
            cache[j] = lj if p < 0 else _tf_mul(cache[p], lj)
            world[j, :3], world[j, 3:] = cache[j]
        return world

    def apply(self, local: np.ndarray, contacts: np.ndarray, dt: float) -> np.ndarray:
        world = self._partial_fk(local)
        out = local
        for i, (hip, knee, ankle) in enumerate(self.chains):
            c = float(contacts[self.contact_index[i]])
            pos = world[ankle, :3]
            if not self.locked[i] and c >= self.on:
                self.locked[i] = True
                self.lock_pos[i] = pos.copy()
            elif self.locked[i] and (c <= self.off or np.linalg.norm(self.lock_pos[i] - pos) > self.max_corr):
                self.locked[i] = False
            rate = dt / max(self.bin if self.locked[i] else self.bout, 1e-4)
            self.weight[i] = float(np.clip(self.weight[i] + (rate if self.locked[i] else -rate), 0.0, 1.0))
            if self.weight[i] > 1e-3:
                target = self.lock_pos[i].copy()
                target[1] = max(target[1], pos[1] if not self.locked[i] else target[1])
                # legs are independent chains below the pelvis: one FK per frame is enough
                out = two_bone_ik(out, world, self.sk, hip, knee, ankle, target, self.weight[i])
        return out


class Smoother:
    """Optional exponential smoothing of local rotations (0 = off). Adds ~alpha/(1-alpha) frames lag."""

    def __init__(self, amount: float = 0.0) -> None:
        self.amount = amount
        self.state: np.ndarray | None = None

    def apply(self, local: np.ndarray, dt: float) -> np.ndarray:
        if self.amount <= 0:
            self.state = None
            return local
        if self.state is None or self.state.shape != local.shape:
            self.state = local.copy()
            return local
        alpha = 1.0 - self.amount ** (dt * 30.0)
        out = local.copy()
        out[1:, 3:] = m3.quat_slerp(self.state[1:, 3:], local[1:, 3:], alpha)
        self.state = out
        return out


def scale_intensity(seg: np.ndarray, intensity: float) -> np.ndarray:
    """Scale each joint's rotation around its mean over the segment (1 = unchanged)."""
    if abs(intensity - 1.0) < 1e-6 or len(seg) < 2:
        return seg
    q = seg[:, 1:, 3:]
    ref = q[0:1]
    aligned = np.where(np.sum(q * ref, -1, keepdims=True) < 0, -q, q)
    mean = m3.normalize(aligned.mean(axis=0, keepdims=True))
    dev = m3.quat_to_scaled_axis(m3.quat_mul(m3.quat_conj(mean), aligned)) * intensity
    out = seg.copy()
    out[:, 1:, 3:] = m3.quat_mul(np.broadcast_to(mean, aligned.shape), m3.quat_from_scaled_axis(dev))
    return out
