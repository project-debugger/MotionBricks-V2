"""Write local-transform animation to BVH (ZYX Euler, centimetres) for viewing in any DCC tool."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import math3d as m3
from .skeleton import Skeleton


def quat_to_euler_zyx_deg(q: np.ndarray) -> np.ndarray:
    """q = qz * qy * qx  ->  (z, y, x) degrees; inverse of the loader's composition."""
    R = m3.quat_to_matrix(q)
    y = np.arcsin(np.clip(-R[..., 2, 0], -1.0, 1.0))
    x = np.arctan2(R[..., 2, 1], R[..., 2, 2])
    z = np.arctan2(R[..., 1, 0], R[..., 0, 0])
    return np.degrees(np.stack([z, y, x], axis=-1))


def write_bvh(path: str | Path, sk: Skeleton, local: np.ndarray, fps: float = 30.0, units: float = 100.0) -> Path:
    path = Path(path)
    children = {j: [] for j in range(sk.num_joints)}
    for j in range(1, sk.num_joints):
        children[int(sk.parents[j])].append(j)
    lines = ["HIERARCHY"]

    def emit(j: int, depth: int) -> None:
        ind = "  " * depth
        off = sk.reference[j, :3] * units if j else np.zeros(3)
        lines.append(f"{ind}{'ROOT' if j == 0 else 'JOINT'} {sk.names[j]}")
        lines.append(f"{ind}{{")
        lines.append(f"{ind}  OFFSET {off[0]:.6f} {off[1]:.6f} {off[2]:.6f}")
        if j == 0:
            lines.append(f"{ind}  CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation")
        else:
            lines.append(f"{ind}  CHANNELS 3 Zrotation Yrotation Xrotation")
        for c in children[j]:
            emit(c, depth + 1)
        if not children[j]:
            lines.extend([f"{ind}  End Site", f"{ind}  {{", f"{ind}    OFFSET 0.0 0.0 0.0", f"{ind}  }}"])
        lines.append(f"{ind}}}")

    emit(0, 0)
    eul = quat_to_euler_zyx_deg(local[..., 3:])           # [F, J, 3]
    root = local[:, 0, :3] * units
    data = np.concatenate([root, eul.reshape(len(local), -1)], axis=1)
    lines += ["MOTION", f"Frames: {len(local)}", f"Frame Time: {1.0 / fps:.6f}"]
    lines += [" ".join(f"{v:.5f}" for v in row) for row in data]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path
