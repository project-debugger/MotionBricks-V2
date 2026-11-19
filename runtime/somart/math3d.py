"""Vectorised rigid-transform math in NumPy.

Conventions match the upstream Warp runtime so results are directly comparable:
- quaternions are XYZW, unit length;
- a transform is an array [..., 7] = (px, py, pz, qx, qy, qz, qw);
- composition ``mul(a, b)`` applies ``b`` first, then ``a`` (parent * local).

All functions broadcast over leading dimensions and never loop in Python over
frames or joints, which keeps encode/decode in the sub-millisecond range.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-8


def normalize(v: np.ndarray, axis: int = -1) -> np.ndarray:
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, EPS)


# --------------------------------------------------------------------- quaternions
def quat_identity(shape=()) -> np.ndarray:
    q = np.zeros(tuple(shape) + (4,), dtype=np.float64)
    q[..., 3] = 1.0
    return q


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], axis=-1)


def quat_conj(q: np.ndarray) -> np.ndarray:
    return q * np.array([-1.0, -1.0, -1.0, 1.0], dtype=q.dtype)


quat_inv = quat_conj  # unit quaternions


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qv = q[..., :3]
    w = q[..., 3:4]
    t = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


def quat_rotate_inv(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return quat_rotate(quat_conj(q), v)


def quat_from_axis_angle(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    axis = normalize(np.asarray(axis, dtype=np.float64))
    half = 0.5 * np.asarray(angle, dtype=np.float64)[..., None]
    return np.concatenate([axis * np.sin(half), np.cos(half)], axis=-1)


def quat_from_yaw(angle: np.ndarray) -> np.ndarray:
    """Rotation about +Y (SOMA up axis)."""
    angle = np.asarray(angle, dtype=np.float64)
    half = 0.5 * angle
    z = np.zeros_like(half)
    return np.stack([z, np.sin(half), z, np.cos(half)], axis=-1)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = np.moveaxis(q, -1, 0)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    m = np.stack([
        1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
        2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
        2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy),
    ], axis=-1)
    return m.reshape(q.shape[:-1] + (3, 3))


def quat_from_matrix(m: np.ndarray) -> np.ndarray:
    """Robust (Shepperd) conversion, vectorised."""
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    tr = m00 + m11 + m22
    q = np.empty(m.shape[:-2] + (4,), dtype=np.float64)

    c0 = tr > 0
    s = np.sqrt(np.maximum(tr + 1.0, EPS)) * 2
    q0 = np.stack([(m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, 0.25 * s], -1)
    c1 = (~c0) & (m00 > m11) & (m00 > m22)
    s = np.sqrt(np.maximum(1.0 + m00 - m11 - m22, EPS)) * 2
    q1 = np.stack([0.25 * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s], -1)
    c2 = (~c0) & (~c1) & (m11 > m22)
    s = np.sqrt(np.maximum(1.0 + m11 - m00 - m22, EPS)) * 2
    q2 = np.stack([(m01 + m10) / s, 0.25 * s, (m12 + m21) / s, (m02 - m20) / s], -1)
    s = np.sqrt(np.maximum(1.0 + m22 - m00 - m11, EPS)) * 2
    q3 = np.stack([(m02 + m20) / s, (m12 + m21) / s, 0.25 * s, (m10 - m01) / s], -1)
    q[...] = q3
    q[c2] = q2[c2]
    q[c1] = q1[c1]
    q[c0] = q0[c0]
    return normalize(q)


def quat_slerp(a: np.ndarray, b: np.ndarray, t) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)[..., None] if np.ndim(t) else np.float64(t)
    d = np.sum(a * b, axis=-1, keepdims=True)
    b = np.where(d < 0, -b, b)
    d = np.abs(d)
    theta = np.arccos(np.clip(d, -1.0, 1.0))
    sin_t = np.sin(theta)
    small = sin_t < 1e-6
    w_a = np.where(small, 1.0 - t, np.sin((1.0 - t) * theta) / np.where(small, 1.0, sin_t))
    w_b = np.where(small, t, np.sin(t * theta) / np.where(small, 1.0, sin_t))
    return normalize(w_a * a + w_b * b)


def quat_to_scaled_axis(q: np.ndarray) -> np.ndarray:
    """Log map (rotation vector, radians * axis); picks the short arc."""
    q = np.where(q[..., 3:4] < 0, -q, q)
    v = q[..., :3]
    s = np.linalg.norm(v, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(s, q[..., 3:4])
    return np.where(s > 1e-9, v / np.maximum(s, 1e-12) * angle, 2.0 * v)


def quat_from_scaled_axis(r: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(r, axis=-1, keepdims=True)
    half = 0.5 * angle
    k = np.where(angle > 1e-9, np.sin(half) / np.maximum(angle, 1e-12), 0.5)
    return normalize(np.concatenate([r * k, np.cos(half)], axis=-1))


def quat_abs(q: np.ndarray) -> np.ndarray:
    return np.where(q[..., 3:4] < 0, -q, q)


def quat_fraction(q: np.ndarray, w: np.ndarray) -> np.ndarray:
    """slerp(identity, q, w) via the log map: one sin/cos per element, no arccos branches.

    q [..., 4] broadcast against w [...]; equals ``quat_slerp(identity, q, w)`` exactly
    (short arc), but ~3x cheaper, which matters for per-frame, per-joint corrections.
    """
    q = quat_abs(q)
    v = q[..., :3]
    s = np.linalg.norm(v, axis=-1)
    theta = 2.0 * np.arctan2(s, q[..., 3])
    axis = v / np.maximum(s, 1e-12)[..., None]
    half = 0.5 * theta * w
    return np.concatenate([axis * np.sin(half)[..., None], np.cos(half)[..., None]], axis=-1)


def quat_angle_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Geodesic angle (rad) between rotations; invariant to the q / -q double cover.

    Uses 2*atan2(|v|, |w|) of the relative rotation, which stays accurate near zero
    (2*acos(|dot|) loses ~4 digits there).
    """
    r = quat_mul(quat_conj(normalize(a)), normalize(b))
    return 2.0 * np.arctan2(np.linalg.norm(r[..., :3], axis=-1), np.abs(r[..., 3]))


# -------------------------------------------------------------------- 6D rotations
def rot6d_from_quat(q: np.ndarray) -> np.ndarray:
    """First two matrix columns (x axis, y axis), the layout the SOMA model uses."""
    x = quat_rotate(q, np.broadcast_to([1.0, 0.0, 0.0], q.shape[:-1] + (3,)))
    y = quat_rotate(q, np.broadcast_to([0.0, 1.0, 0.0], q.shape[:-1] + (3,)))
    return np.concatenate([x, y], axis=-1)


def quat_from_rot6d(r6: np.ndarray) -> np.ndarray:
    """Gram-Schmidt exactly as the upstream decode kernel (x first, z = x cross y)."""
    x = normalize(r6[..., 0:3])
    z = normalize(np.cross(x, r6[..., 3:6]))
    y = np.cross(z, x)
    m = np.stack([x, y, z], axis=-1)  # columns
    return quat_from_matrix(m)


# --------------------------------------------------------------------- transforms
def tf(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return np.concatenate([p, q], axis=-1)


def tf_identity(shape=()) -> np.ndarray:
    t = np.zeros(tuple(shape) + (7,), dtype=np.float64)
    t[..., 6] = 1.0
    return t


def tf_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    pa, qa = a[..., :3], a[..., 3:]
    pb, qb = b[..., :3], b[..., 3:]
    return tf(pa + quat_rotate(qa, pb), quat_mul(qa, qb))


def tf_inv(a: np.ndarray) -> np.ndarray:
    qi = quat_conj(a[..., 3:])
    return tf(-quat_rotate(qi, a[..., :3]), qi)


def tf_blend(a: np.ndarray, b: np.ndarray, t) -> np.ndarray:
    """Lerp positions, slerp rotations (upstream ``blend_transform``)."""
    tt = np.asarray(t, dtype=np.float64)
    tp = tt[..., None] if tt.ndim else tt
    return tf(a[..., :3] + (b[..., :3] - a[..., :3]) * tp, quat_slerp(a[..., 3:], b[..., 3:], t))


# --------------------------------------------------------------------- hierarchy
def depth_levels(parents: np.ndarray) -> list[np.ndarray]:
    """Group joints by tree depth so FK runs one vectorised op per level (not per joint)."""
    depth = np.zeros(len(parents), dtype=np.int64)
    for j in range(1, len(parents)):
        depth[j] = depth[parents[j]] + 1
    return [np.where(depth == d)[0] for d in range(1, int(depth.max()) + 1)] if len(parents) > 1 else []


def forward_kinematics(local: np.ndarray, parents: np.ndarray, root_tx: np.ndarray | None = None,
                       levels: list[np.ndarray] | None = None) -> np.ndarray:
    """local [..., J, 7] -> world [..., J, 7]. Parents must precede children."""
    world = np.empty(local.shape, dtype=np.float64)
    world[..., 0, :] = local[..., 0, :] if root_tx is None else tf_mul(root_tx, local[..., 0, :])
    if levels is None:
        levels = depth_levels(parents)
    for idx in levels:
        world[..., idx, :] = tf_mul(world[..., parents[idx], :], local[..., idx, :])
    return world


def inverse_kinematics_local(world: np.ndarray, parents: np.ndarray) -> np.ndarray:
    """world [..., J, 7] -> local [..., J, 7]."""
    local = np.empty_like(world)
    local[..., 0, :] = world[..., 0, :]
    par = parents[1:]
    local[..., 1:, :] = tf_mul(tf_inv(world[..., par, :]), world[..., 1:, :])
    return local


def wrap_angle(a: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(a), np.cos(a))
