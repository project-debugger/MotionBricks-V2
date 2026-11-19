"""Encode guide poses into SOMA model tensors and decode predictions back to poses.

A faithful, vectorised NumPy port of the upstream Warp kernels
(``motionbricks/model/soma/kernels.py`` + ``model/kernels.py`` at the pinned commit):

- ``compute_guide_alignment_kernel``      -> :func:`guide_alignment`
- ``encode_motionbricks_inputs_kernel``   -> :func:`encode`
- ``write_motionbricks_conditions_kernel``-> :func:`conditions`
- ``decode_motionbricks_output_kernel``   -> :func:`decode`
- ``compute/apply_endpoint_deltas_kernel``-> :func:`endpoint_correction`

Every function accepts a leading batch axis so many characters can be encoded at once.
All poses are in SOMA model space (Y up, metres), which is also the BVH clip space.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import contract as C
from . import math3d as m3
from .skeleton import Skeleton


@dataclass
class SkeletonBinding:
    """Mapping between a runtime skeleton and the 30 model joints (upstream ``_bind_pose_skeleton``)."""

    skeleton: Skeleton
    model_to_pose: np.ndarray     # [30] pose index of each model joint
    pose_to_model: np.ndarray     # [J] model index or -1 (keeps reference transform)
    pose_parent_model: np.ndarray  # [J] model index of the pose joint's parent (or -1)
    right_leg: int
    left_leg: int

    @classmethod
    def bind(cls, skeleton: Skeleton) -> "SkeletonBinding":
        m2p = np.array([skeleton.index(n) for n in C.JOINT_NAMES], dtype=np.int64)
        if (m2p < 0).any():
            missing = [n for n, i in zip(C.JOINT_NAMES, m2p) if i < 0]
            raise ValueError("skeleton is missing model joints: " + ", ".join(missing))
        if m2p[0] != 0:
            raise ValueError("skeleton must store the root (Hips) at index 0")
        p2m = np.full(skeleton.num_joints, -1, dtype=np.int64)
        for mj, pj in enumerate(m2p):
            par = skeleton.parents[pj]
            if par == -1 or p2m[par] >= 0:
                p2m[pj] = mj
        ppm = np.array([p2m[skeleton.parents[j]] if skeleton.parents[j] >= 0 else -1
                        for j in range(skeleton.num_joints)])
        return cls(skeleton, m2p, p2m, ppm, skeleton.index(C.HEADING_JOINTS[0]),
                   skeleton.index(C.HEADING_JOINTS[1]))


def heading_from_hips(right: np.ndarray, left: np.ndarray) -> np.ndarray:
    """Heading (rad) of the body from the leg roots, as the training representation."""
    ax = right[..., 0] - left[..., 0]
    az = right[..., 2] - left[..., 2]
    n = np.sqrt(ax * ax + az * az)
    ok = n > 1e-6
    ax = np.where(ok, ax / np.where(ok, n, 1), 0.0)
    az = np.where(ok, az / np.where(ok, n, 1), 0.0)
    # forward = up x across = (az, 0, -ax)
    return np.where(ok, np.arctan2(az, -ax), 0.0)


def guide_alignment(world: np.ndarray, b: SkeletonBinding, heading=(1.0, 0.0)) -> np.ndarray:
    """world [N, 8, J, 7] -> alignment transform [N, 7] for the last context pose."""
    last = C.POSES_PER_TOKEN - 1
    hips = world[:, last, 0, :3]
    cur = heading_from_hips(world[:, last, b.right_leg, :3], world[:, last, b.left_leg, :3])
    desired = np.arctan2(heading[0], heading[1])
    p = np.stack([hips[:, 0], np.zeros_like(hips[:, 0]), hips[:, 2]], axis=-1)
    return m3.tf(p, m3.quat_from_yaw(cur - desired))


def encode(world: np.ndarray, b: SkeletonBinding, alignment: np.ndarray) -> dict[str, np.ndarray]:
    """world [N, 8, J, 7] (4 context + 4 target poses) -> float tensors."""
    N = world.shape[0]
    aq = alignment[:, None, None, 3:]
    ap = alignment[:, None, None, :3]
    pos = m3.quat_rotate_inv(np.broadcast_to(aq, world.shape[:-1] + (4,)), world[..., :3] - ap)
    rot = m3.quat_mul(m3.quat_conj(np.broadcast_to(aq, world.shape[:-1] + (4,))), world[..., 3:])

    frames = np.arange(C.CONTEXT_AND_TARGET_POSES)
    nxt = np.where(frames == C.POSES_PER_TOKEN - 1, frames, np.minimum(frames + 1, 7))
    hips = pos[:, :, 0]
    heading = heading_from_hips(pos[:, :, b.right_leg], pos[:, :, b.left_leg])
    dh = m3.wrap_angle(heading[:, nxt] - heading)
    g = np.empty((N, 8, 5))
    g[..., 0:3] = hips
    g[..., 3] = np.cos(heading)
    g[..., 4] = np.sin(heading)
    lr = np.empty((N, 8, 4))
    lr[..., 0] = dh * C.SAMPLE_RATE
    lr[..., 1] = (hips[:, nxt, 0] - hips[..., 0]) * C.SAMPLE_RATE
    lr[..., 2] = (hips[:, nxt, 2] - hips[..., 2]) * C.SAMPLE_RATE
    lr[..., 3] = hips[..., 1]

    root_ground = hips * np.array([1.0, 0.0, 1.0])
    mpos = pos[:, :, b.model_to_pose[1:]] - root_ground[:, :, None]
    mrot = rot[:, :, b.model_to_pose]
    lp = np.concatenate([mpos.reshape(N, 8, -1), m3.rot6d_from_quat(mrot).reshape(N, 8, -1)], axis=-1)
    return {"global_root_values": g.astype(np.float32), "local_root_values": lr.astype(np.float32),
            "local_poses": lp.astype(np.float32)}


def conditions(num_target_poses: np.ndarray, allowed_counts) -> dict[str, np.ndarray]:
    """Masks for N requests (upstream ``write_motionbricks_conditions_kernel``)."""
    nt = np.atleast_1d(np.asarray(num_target_poses, dtype=np.int64))
    idx = np.arange(C.CONTEXT_AND_TARGET_POSES)[None]
    given = (idx < C.POSES_PER_TOKEN + nt[:, None]).astype(np.int32)
    lr = given.copy()
    lr[(idx == C.POSES_PER_TOKEN - 1).repeat(len(nt), 0)] = 0
    lr[idx == (C.POSES_PER_TOKEN + nt[:, None] - 1)] = 0
    if isinstance(allowed_counts[0], (list, tuple, np.ndarray)):
        allowed = np.array([C.allowed_token_mask(a) for a in allowed_counts], dtype=np.int32)
    else:
        allowed = np.tile(np.array(C.allowed_token_mask(allowed_counts), dtype=np.int32), (len(nt), 1))
    return {"has_global_root_values": given, "has_local_root_values": lr, "has_local_poses": given.copy(),
            "allowed_pred_num_tokens": allowed}


def pad_token(poses: np.ndarray) -> tuple[np.ndarray, int]:
    """1..4 poses [k, J, 7] -> 4 poses (last one repeated), plus k."""
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim == 2:
        poses = poses[None]
    k = poses.shape[0]
    if not 1 <= k <= C.POSES_PER_TOKEN:
        raise ValueError("a token holds 1 to 4 poses")
    idx = np.minimum(np.arange(C.POSES_PER_TOKEN), k - 1)
    return poses[idx], k


def decode(pred: np.ndarray, tokens: np.ndarray, b: SkeletonBinding, alignment: np.ndarray
           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """pred [N, 64, 366] -> local poses [N, 58, J, 7], contacts [N, 58, 4], valid frame counts [N]."""
    N = pred.shape[0]
    raw = np.clip(tokens.astype(np.int64) * C.POSES_PER_TOKEN, 0, pred.shape[1])
    counts = np.maximum(0, raw - 2 * C.BOUNDARY_FRAMES)
    fr = pred[:, C.BOUNDARY_FRAMES:C.BOUNDARY_FRAMES + C.MAX_VALID_FRAMES].astype(np.float64)
    contacts = np.clip(fr[..., C.OUT_CONTACTS], 0.0, 1.0)
    r6 = fr[..., C.OUT_ROT_OFFSET:C.OUT_ROT_OFFSET + 6 * C.NUM_JOINTS].reshape(N, -1, C.NUM_JOINTS, 6)
    gq = m3.quat_from_rot6d(r6)  # [N, F, 30, 4] global (aligned) rotations
    J = b.skeleton.num_joints
    out = np.broadcast_to(b.skeleton.reference, (N, fr.shape[1], J, 7)).copy()
    hips = m3.tf(fr[..., 0:3], gq[:, :, 0])
    out[:, :, 0] = m3.tf_mul(alignment[:, None], hips)
    mapped = np.where(b.pose_to_model >= 0)[0]
    mapped = mapped[mapped > 0]
    child = gq[:, :, b.pose_to_model[mapped]]
    parent = gq[:, :, b.pose_parent_model[mapped]]
    out[:, :, mapped, 3:] = m3.normalize(m3.quat_mul(m3.quat_conj(parent), child))
    return out, contacts, counts


def velocity_matching(local: np.ndarray, counts: np.ndarray, ctx_prev: np.ndarray, ctx_last: np.ndarray,
                      tgt_first: np.ndarray, tgt_second: np.ndarray | None, tau: float = 4.0) -> np.ndarray:
    """Make velocities continuous at both joins (an addition to upstream endpoint correction).

    Endpoint correction matches *poses* at the boundaries but not their *velocities*, which shows
    up as a kink/pop when the motion is played after the context. We add an offset
    ``x(t) = dv * t * exp(-t / tau)`` (zero at the boundary, derivative ``dv`` there, decaying
    with time constant ``tau`` frames) that cancels the velocity mismatch ``dv`` - for the root
    position (world) and every joint rotation (local, rotation-vector space). The end is treated
    symmetrically backwards in time when the target holds >= 2 poses.
    Shapes: local [N, F, J, 7]; ctx_* / tgt_* [N, J, 7]; tgt_second may be None.
    """
    out = local.copy()
    N, F, J = local.shape[:3]
    frame = np.arange(F, dtype=np.float64)

    def rotvel(a, b):  # angular step taking a -> b, as a rotation vector [.., J, 3]
        return m3.quat_to_scaled_axis(m3.quat_mul(b[..., 3:], m3.quat_conj(a[..., 3:])))

    for n in range(N):
        c = int(counts[n])
        if c < 4:
            continue
        t = frame[:c]
        seg = out[n, :c]
        # --- start: context velocity vs generated velocity
        dv_p = (ctx_last[n, 0, :3] - ctx_prev[n, 0, :3]) - (seg[1, 0, :3] - seg[0, 0, :3])
        dv_r = rotvel(ctx_prev[n], ctx_last[n]) - rotvel(seg[0], seg[1])           # [J, 3]
        w = (t * np.exp(-t / tau))[:, None]                                         # [c, 1]
        off_p = w * dv_p
        off_r = w[:, :, None] * dv_r[None]
        if tgt_second is not None:
            s = t[::-1]  # frames remaining to the end
            ws = (s * np.exp(-s / tau))[:, None]
            de_p = (tgt_second[n, 0, :3] - tgt_first[n, 0, :3]) - (seg[c - 1, 0, :3] - seg[c - 2, 0, :3])
            de_r = rotvel(tgt_first[n], tgt_second[n]) - rotvel(seg[c - 2], seg[c - 1])
            off_p = off_p - ws * de_p
            off_r = off_r - ws[:, :, None] * de_r[None]
        seg[:, 0, :3] = seg[:, 0, :3] + off_p
        seg[:, :, 3:] = m3.normalize(m3.quat_mul(m3.quat_from_scaled_axis(off_r), seg[:, :, 3:]))
        out[n, :c] = seg
    return out


def _root_delta(gen: np.ndarray, guide: np.ndarray) -> np.ndarray:
    q = m3.normalize(m3.quat_mul(m3.quat_conj(m3.normalize(gen[..., 3:])), m3.normalize(guide[..., 3:])))
    return m3.tf(guide[..., :3] - gen[..., :3], q)


def contact_aware_weights(contacts: np.ndarray, counts: np.ndarray, floor: float = 0.05) -> np.ndarray:
    """End-correction weight ramp [N, F] that advances mostly while the feet are unloaded.

    Moving the root while a foot is planted drags that foot; ramping the root correction by the
    cumulative "mobility" 1 - max(contact) spends it in flight / swing / weight-transfer frames.
    """
    N, F = contacts.shape[:2]
    mob = np.clip(1.0 - contacts.max(axis=-1), floor, 1.0)                 # [N, F]
    frame = np.arange(F)[None, :]
    mob = np.where(frame < counts[:, None], mob, 0.0)
    cum = np.cumsum(mob, axis=1) - mob[:, :1]
    total = np.take_along_axis(cum, np.clip(counts[:, None] - 1, 0, F - 1), axis=1)
    return np.clip(cum / np.maximum(total, 1e-9), 0.0, 1.0)


def endpoint_correction(local: np.ndarray, counts: np.ndarray, context_last: np.ndarray,
                        target_first: np.ndarray, contacts: np.ndarray | None = None) -> np.ndarray:
    """Blend start/end deltas so motion starts at the last context pose and ends at the first target.

    local [N, F, J, 7], context_last/target_first [N, J, 7]. Returns a corrected copy.
    With ``contacts`` [N, F, 4] the *root* correction follows :func:`contact_aware_weights`
    (somart extension, reduces foot sliding); otherwise it is the upstream linear ramp.
    """
    N, F, J = local.shape[:3]
    counts = np.asarray(counts, dtype=np.int64)
    valid = counts >= 2
    if not valid.any():
        return local.copy()
    last = np.clip(counts - 1, 0, F - 1)
    n_idx = np.arange(N)
    first_pose, last_pose = local[:, 0], local[n_idx, last]               # [N, J, 7]
    frame = np.arange(F)[None, :]
    w_end = np.clip(frame / np.maximum(counts[:, None] - 1, 1), 0.0, 1.0)  # [N, F]
    w_start = 1.0 - w_end
    rw_end = contact_aware_weights(contacts, counts) if contacts is not None else w_end
    rw_start = 1.0 - rw_end

    # Root: world-space position offset; rotation delta applied after the generated rotation.
    sd = _root_delta(first_pose[:, 0], context_last[:, 0])                 # [N, 7]
    ed = _root_delta(last_pose[:, 0], target_first[:, 0])
    sq = m3.quat_fraction(sd[:, None, 3:], rw_start)
    eq = m3.quat_fraction(ed[:, None, 3:], rw_end)
    rq = m3.normalize(m3.quat_mul(m3.quat_mul(local[:, :, 0, 3:], sq), eq))
    rp = local[:, :, 0, :3] + sd[:, None, :3] * rw_start[..., None] + ed[:, None, :3] * rw_end[..., None]

    # Other joints: local-space deltas blended from identity (upstream apply_delta_transform):
    # blend(I, d, w) = (w * d.p, slerp(I, d.q, w)).
    sdel = m3.tf_mul(m3.tf_inv(first_pose[:, 1:]), context_last[:, 1:])   # [N, J-1, 7]
    edel = m3.tf_mul(m3.tf_inv(last_pose[:, 1:]), target_first[:, 1:])
    ws = w_start[:, :, None]
    we = w_end[:, :, None]
    sblend = m3.tf(sdel[:, None, :, :3] * ws[..., None], m3.quat_fraction(sdel[:, None, :, 3:], ws))
    eblend = m3.tf(edel[:, None, :, :3] * we[..., None], m3.quat_fraction(edel[:, None, :, 3:], we))
    body = m3.tf_mul(m3.tf_mul(local[:, :, 1:], sblend), eblend)

    out = local.copy()
    mask = valid[:, None] & (frame < counts[:, None])                       # [N, F]
    out[:, :, 0] = np.where(mask[..., None], m3.tf(rp, rq), local[:, :, 0])
    out[:, :, 1:] = np.where(mask[..., None, None], body, local[:, :, 1:])
    return out
