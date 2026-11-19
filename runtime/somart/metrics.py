"""Animation-quality metrics (all positions in metres, 30 Hz unless stated).

These are standard kinematic metrics, chosen so they can be computed for any model
variant *and* for the ground-truth mocap, giving a reference level for each:

- mpjpe            mean per-joint position error vs a reference motion
- root_error       mean root (Hips) position error
- rot_error_deg    mean geodesic joint rotation error (local rotations)
- foot_slide       mean horizontal foot speed (m/s) while the foot is in contact
- jerk             mean magnitude of the 3rd time-derivative of joint positions (m/s^3)
- boundary_accel   acceleration spike at the context->motion and motion->target joins
                   relative to the median in-clip acceleration (1.0 == seamless)
"""
from __future__ import annotations

import numpy as np

from . import contract as C
from . import math3d as m3

FOOT_JOINTS = C.FOOT_CONTACT_JOINTS


def mpjpe(a_world: np.ndarray, b_world: np.ndarray) -> float:
    n = min(len(a_world), len(b_world))
    return float(np.linalg.norm(a_world[:n, :, :3] - b_world[:n, :, :3], axis=-1).mean())


def root_error(a_world: np.ndarray, b_world: np.ndarray) -> float:
    n = min(len(a_world), len(b_world))
    return float(np.linalg.norm(a_world[:n, 0, :3] - b_world[:n, 0, :3], axis=-1).mean())


def rot_error_deg(a_local: np.ndarray, b_local: np.ndarray, joints: np.ndarray | None = None) -> float:
    n = min(len(a_local), len(b_local))
    a, b = a_local[:n, :, 3:], b_local[:n, :, 3:]
    if joints is not None:
        a, b = a[:, joints], b[:, joints]
    return float(np.degrees(m3.quat_angle_between(a, b)).mean())


def foot_contacts_from_height(world: np.ndarray, foot_idx: list[int], fps: float = C.SAMPLE_RATE,
                              height: float | None = None, speed: float = 0.6) -> np.ndarray:
    """Heuristic ground-truth contacts: foot low AND slow (works for mocap without labels)."""
    p = world[:, foot_idx, :3]
    y = p[..., 1]
    ground = np.percentile(y, 5, axis=0) if height is None else np.full(len(foot_idx), height)
    v = np.zeros_like(y)
    v[1:] = np.linalg.norm(np.diff(p[..., [0, 2]], axis=0), axis=-1) * fps
    return ((y - ground) < 0.05) & (v < speed)


def foot_slide(world: np.ndarray, contacts: np.ndarray, foot_idx: list[int], fps: float = C.SAMPLE_RATE,
               threshold: float = 0.5) -> float:
    """Mean horizontal speed (m/s) of feet over frames where contact > threshold."""
    p = world[:, foot_idx, :3]
    v = np.linalg.norm(np.diff(p[..., [0, 2]], axis=0), axis=-1) * fps  # [F-1, 4]
    c = (contacts[1:] > threshold) & (contacts[:-1] > threshold) if contacts.dtype != bool else \
        (contacts[1:] & contacts[:-1])
    return float(v[c].mean()) if c.any() else 0.0


def jerk(world: np.ndarray, fps: float = C.SAMPLE_RATE) -> float:
    p = world[..., :3]
    if len(p) < 4:
        return 0.0
    j = np.diff(p, n=3, axis=0) * fps ** 3
    return float(np.linalg.norm(j, axis=-1).mean())


def accel_profile(world: np.ndarray, fps: float = C.SAMPLE_RATE) -> np.ndarray:
    p = world[..., :3]
    a = np.diff(p, n=2, axis=0) * fps ** 2
    return np.linalg.norm(a, axis=-1).mean(axis=-1)  # [F-2] mean over joints


def boundary_accel(context_world: np.ndarray, gen_world: np.ndarray, target_world: np.ndarray,
                   fps: float = C.SAMPLE_RATE) -> dict:
    """Acceleration at the joins of [context | generated[1:-1] | target] vs in-clip median.

    gen[0] equals the last context pose and gen[-1] equals the first target pose, so they are
    dropped to avoid duplicate frames.
    """
    seq = np.concatenate([context_world, gen_world[1:-1], target_world], axis=0)
    a = accel_profile(seq, fps)
    nc = len(context_world)
    i_start = nc - 2  # accel centred on the last context frame and first generated frame
    i_end = nc + len(gen_world) - 2 - 2
    med = float(np.median(a)) + 1e-6
    s = float(a[max(i_start, 0):i_start + 2].max())
    e = float(a[max(i_end, 0):i_end + 2].max())
    return {"start": s / med, "end": e / med, "median_accel": med}


def summarize(values: list[float]) -> dict:
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return {}
    return {"mean": float(v.mean()), "p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95)),
            "p99": float(np.percentile(v, 99)), "min": float(v.min()), "max": float(v.max()),
            "std": float(v.std()), "n": int(v.size)}
