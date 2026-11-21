"""Dog motion-capture loading for the animal motion network (MANN dog data, CC BY-NC 4.0).

Data: https://starke-consult.de/AI4Animation/SIGGRAPH_2018/MotionCapture.zip (Zhang, Starke, Komura,
Saito: "Mode-Adaptive Neural Networks for Quadruped Motion Control", SIGGRAPH 2018). 51 BVH clips of a
dog at 60 fps. Licence: Attribution-NonCommercial 4.0 International.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class BVH:
    names: list
    parents: list
    offsets: np.ndarray          # [J, 3] (cm)
    channels: list               # per joint: list of channel names
    frames: np.ndarray           # [F, C]
    frame_time: float
    end_of: dict = field(default_factory=dict)  # end-site point index -> parent joint index


def load_bvh(path: str | Path) -> BVH:
    text = Path(path).read_text().replace("\r", "")
    tok = text.split()
    i, names, parents, offsets, channels, stack, end_of = 0, [], [], [], [], [], {}
    while tok[i] != "MOTION":
        t = tok[i]
        if t in ("ROOT", "JOINT"):
            names.append(tok[i + 1]); parents.append(stack[-1] if stack else -1); channels.append([]); offsets.append(None); i += 2
        elif t == "End":  # End Site: treat as a point without channels
            names.append(names[stack[-1]] + "End"); parents.append(stack[-1]); channels.append([]); offsets.append(None)
            end_of[len(names) - 1] = stack[-1]; i += 2
        elif t == "{":
            stack.append(len(names) - 1); i += 1
        elif t == "}":
            stack.pop(); i += 1
        elif t == "OFFSET":
            offsets[stack[-1]] = [float(tok[i + 1]), float(tok[i + 2]), float(tok[i + 3])]; i += 4
        elif t == "CHANNELS":
            n = int(tok[i + 1]); channels[stack[-1]] = tok[i + 2:i + 2 + n]; i += 2 + n
        else:
            i += 1
    i += 1
    assert tok[i] == "Frames:"; F = int(tok[i + 1]); i += 2
    assert tok[i] == "Frame" and tok[i + 1] == "Time:"; ft = float(tok[i + 2]); i += 3
    C = sum(len(c) for c in channels)
    frames = np.array(tok[i:i + F * C], dtype=np.float64).reshape(F, C)
    return BVH(names, parents, np.array(offsets, dtype=np.float64), channels, frames, ft, end_of)


def _rot(axis: str, deg: np.ndarray) -> np.ndarray:
    a = np.radians(deg); c, s = np.cos(a), np.sin(a); o, z = np.ones_like(a), np.zeros_like(a)
    if axis == "X":
        return np.stack([np.stack([o, z, z], -1), np.stack([z, c, -s], -1), np.stack([z, s, c], -1)], -2)
    if axis == "Y":
        return np.stack([np.stack([c, z, s], -1), np.stack([z, o, z], -1), np.stack([-s, z, c], -1)], -2)
    return np.stack([np.stack([c, -s, z], -1), np.stack([s, c, z], -1), np.stack([z, z, o], -1)], -2)


def forward_kinematics(b: BVH, scale: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """Global positions [F, P, 3] (metres) and rotations [F, P, 3, 3] of every joint and end site."""
    F, P = len(b.frames), len(b.names)
    pos = np.zeros((F, P, 3)); rot = np.zeros((F, P, 3, 3))
    col = 0
    local_r = [None] * P; local_t = [None] * P
    for j in range(P):
        R = np.broadcast_to(np.eye(3), (F, 3, 3)).copy(); t = np.broadcast_to(b.offsets[j], (F, 3)).copy()
        for ch in b.channels[j]:
            v = b.frames[:, col]; col += 1
            if ch.endswith("position"):
                t[:, "XYZ".index(ch[0])] = v
            else:
                R = R @ _rot(ch[0], v)
        local_r[j], local_t[j] = R, t
    for j in range(P):
        p = b.parents[j]
        if p < 0:
            rot[:, j] = local_r[j]; pos[:, j] = local_t[j]
        else:
            rot[:, j] = rot[:, p] @ local_r[j]
            pos[:, j] = pos[:, p] + np.einsum("fij,fj->fi", rot[:, p], local_t[j])
    return pos * scale, rot


def load_all(folder: str | Path):
    clips = []
    for f in sorted(Path(folder).glob("*.bvh")):
        b = load_bvh(f)
        pos, _ = forward_kinematics(b)
        clips.append((f.name, b, pos))
    return clips
