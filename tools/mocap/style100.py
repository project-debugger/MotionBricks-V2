#!/usr/bin/env python3
"""Real human walking reference clips from 100STYLE, retargeted onto the SOMA rig.

100STYLE (Mason, Starke, Komura 2022, https://doi.org/10.5281/zenodo.8127870) is optical motion capture of
one performer walking, running and idling in 100 styles, released under CC BY 4.0. We use a few natural
styles as extra *targets* for the SOMA controller (the network still generates every frame).

  python3 -I tools/mocap/style100.py fetch      # range-downloads only the BVH files we use (~60 MB of a 1.5 GB zip)
  python3 -I tools/mocap/style100.py retarget   # -> data/external/100style/soma/*.npz (SOMA local poses, 30 fps)
  python3 -I tools/mocap/style100.py check      # retarget quality report (bone directions, foot heights, sliding)

Retargeting: the SOMA mesh's bind pose is an A-pose, 100STYLE's rest pose is a T-pose with identity joint
frames. We build a SOMA T-pose by swinging each mapped bone of the bind pose onto the 100STYLE rest bone
direction (top-down, so twist comes from the bind pose; feet and toes only in heading, so a flat foot stays
flat), then copy each 100STYLE joint's world rotation onto it. Joints with no counterpart (second neck joint, jaw, eyes, fingers, ends) keep their calibrated local
rotation. The pelvis is placed so SOMA's hip sockets follow the performer's, scaled by leg length, and the
clip is shifted vertically so its stance feet sit at the same height as in the NVIDIA sample clips.

The downloaded files are untrusted input: they are only parsed as numbers (BVH) and CSV text.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "runtime"))
from somart import math3d as m3  # noqa: E402
from somart.skeleton import Clip, Skeleton, load_bvh  # noqa: E402

URL = "https://zenodo.org/records/8127870/files/100STYLE.zip?download=1"
CACHE = ROOT / "data/external/100style"
LICENSE = ("100STYLE dataset by Ian Mason, Sebastian Starke and Taku Komura (CC BY 4.0, "
           "https://doi.org/10.5281/zenodo.8127870), retargeted onto the SOMA rig")
# style -> locomotion kinds used. FW forward walk, FR forward run (a slow jog here), ID idle,
# TR1 transitions (starts, stops, speed changes)
STYLES = {"Neutral": ["FW", "FR", "ID", "TR1"]}
# optional walk styles (Settings > Walk style), each loaded by the page only when chosen: forward walk,
# idle and transitions (starts and stops) in that style. Label shown in the game -> 100STYLE style name.
# Left out after looking at them on SOMA: Elated (an exaggerated crouched bounce), hands-in-pockets and other
# styles whose hands would sink into the unclothed body, and acted styles (zombie, robot, ...).
WALK_STYLES = {"Proud": "Proud", "Strutting": "Strutting", "Rushed": "Rushed", "Sad": "Depressed",
               "Elderly": "Old", "Hands behind back": "ArmsBehindBack"}
STYLE_KINDS = ["FW", "ID", "TR1"]
CATEGORY = {"FW": "100style_walk", "FR": "100style_jog", "ID": "100style_idle", "TR1": "100style_transitions"}

# SOMA joint -> (100STYLE joint, SOMA joint the bone points to, 100STYLE joint or rest vector it points to)
MAP = {
    "Hips": ("Hips", "Spine1", "Chest"),
    "Spine1": ("Chest", "Spine2", "Chest2"),
    "Spine2": ("Chest2", "Chest", "Chest3"),
    "Chest": ("Chest4", "Neck1", "Neck"),
    "Neck1": ("Neck", "Head", "Head"),
    "Head": ("Head", "HeadEnd", (0.0, 1.0, 0.0)),
}
for side, sx in (("Left", 1.0), ("Right", -1.0)):
    MAP.update({
        f"{side}Shoulder": (f"{side}Collar", f"{side}Arm", f"{side}Shoulder"),
        f"{side}Arm": (f"{side}Shoulder", f"{side}ForeArm", f"{side}Elbow"),
        f"{side}ForeArm": (f"{side}Elbow", f"{side}Hand", f"{side}Wrist"),
        f"{side}Hand": (f"{side}Wrist", f"{side}HandMiddle1", (sx, 0.0, 0.0)),
        f"{side}Leg": (f"{side}Hip", f"{side}Shin", f"{side}Knee"),
        f"{side}Shin": (f"{side}Knee", f"{side}Foot", f"{side}Ankle"),
        f"{side}Foot": (f"{side}Ankle", f"{side}ToeBase", f"{side}Toe"),
        f"{side}ToeBase": (f"{side}Toe", f"{side}ToeEnd", (0.0, -1.292898, 4.893873)),
    })


# Feet and toes: 100STYLE's foot is taller (ankle 11 cm above the floor vs SOMA's 7 cm), so aligning the foot
# bone exactly would tip SOMA onto its toes. Both rest poses stand flat-footed, so only the heading is aligned.
FLAT = {f"{s}{j}" for s in ("Left", "Right") for j in ("Foot", "ToeBase")}


# --------------------------------------------------------------------------------------------- fetch
class _HttpFile(io.RawIOBase):
    """Seekable read-only view of a remote file through HTTP range requests."""

    def __init__(self, url: str) -> None:
        self.url, self.pos = url, 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
            self.size = int(r.headers["Content-Length"])

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, off: int, whence: int = 0) -> int:
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def readinto(self, b) -> int:
        if self.pos >= self.size:
            return 0
        end = min(self.size, self.pos + len(b)) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req, timeout=180) as r:
            data = r.read()
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def fetch(styles: dict[str, list[str]] = STYLES, out: Path = CACHE / "raw") -> None:
    out.mkdir(parents=True, exist_ok=True)
    want = ["100STYLE/Frame_Cuts.csv", "100STYLE/Dataset_List.csv"] + [
        f"100STYLE/{s}/{s}_{k}.bvh" for s, kinds in styles.items() for k in kinds]
    todo = [n for n in want if not (out / Path(n).name).exists()]
    if not todo:
        print("all files cached in", out)
        return
    z = zipfile.ZipFile(io.BufferedReader(_HttpFile(URL), buffer_size=1 << 20))
    for n in todo:
        with z.open(n) as f:
            data = f.read()
        (out / Path(n).name).write_bytes(data)
        print(f"{n}: {len(data) / 1e6:.1f} MB")


def frame_cuts(raw: Path = CACHE / "raw") -> dict[str, dict[str, tuple[int, int]]]:
    rows = [r.strip().split(",") for r in (raw / "Frame_Cuts.csv").read_text().splitlines() if r.strip()]
    head, out = rows[0], {}
    for r in rows[1:]:
        cuts = {}
        for i, h in enumerate(head):
            if h.endswith("_START") and r[i] != "N/A" and r[i + 1] != "N/A":
                cuts[h[:-6]] = (int(r[i]), int(r[i + 1]))
        out[r[0]] = cuts
    return out


# ----------------------------------------------------------------------------------------- retarget
def quat_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Shortest rotation taking direction a onto direction b (xyzw)."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    c = np.cross(a, b)
    w = 1.0 + float(np.dot(a, b))
    if w < 1e-8:  # opposite: any perpendicular axis
        ax = np.cross(a, [1.0, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1.0, 0])
        return np.concatenate([ax / np.linalg.norm(ax), [0.0]])
    q = np.concatenate([c, [w]])
    return q / np.linalg.norm(q)


def soma_bind() -> tuple[np.ndarray, np.ndarray]:
    """SOMA mesh bind pose: world rotations [J, 4] and positions [J, 3]."""
    from somart.mirror import Mirror
    meta = json.loads((ROOT / "showcase/build/cache/soma_mesh.json").read_text())
    raw = (ROOT / "showcase/build/cache/soma_mesh.bin").read_bytes()
    ib = meta["inverseBind"]
    mats = np.frombuffer(raw, np.float32, count=ib["length"], offset=ib["offset"]).reshape(-1, 16)
    pos = np.array([np.linalg.inv(m.reshape(4, 4).T.astype(np.float64))[:3, 3] for m in mats])
    return Mirror.bind_from_inverse_bind(mats), pos


class Style100Retargeter:
    def __init__(self, sk: Skeleton, src: Skeleton) -> None:
        self.sk, self.src = sk, src
        bq, bp = soma_bind()
        sw = src.fk(src.reference)
        self.map = {sk.index(d): src.index(s) for d, (s, _, _) in MAP.items()}
        # calibrated SOMA T-pose (world rotations), built top-down from the bind pose
        W = np.zeros((sk.num_joints, 4))
        for j in range(sk.num_joints):
            p = sk.parents[j]
            name = sk.names[j]
            # feet and toes keep their bind (flat on the floor) world orientation instead of inheriting the leg's swing
            w0 = bq[j] if p < 0 or name in FLAT else m3.quat_mul(W[p], m3.quat_mul(m3.quat_conj(bq[p]), bq[j]))
            if name in MAP:
                _, dc, sc = MAP[name]
                d_local = m3.quat_rotate(m3.quat_conj(bq[j]), bp[sk.index(dc)] - bp[j])
                d_soma = m3.quat_rotate(w0, d_local)
                d_src = np.asarray(sc, float) if isinstance(sc, tuple) else sw[src.index(sc), :3] - sw[src.index(MAP[name][0]), :3]
                if name in FLAT:  # both rest feet stand flat: match only the heading, keep SOMA's own foot pitch
                    d_soma, d_src = d_soma * [1, 0, 1], d_src * [1, 0, 1]
                w0 = m3.quat_mul(quat_between(d_soma, d_src), w0)
            W[j] = m3.normalize(w0)
        self.tpose = W
        self.cal_local = np.array([W[j] if sk.parents[j] < 0 else m3.quat_mul(m3.quat_conj(W[sk.parents[j]]), W[j])
                                   for j in range(sk.num_joints)])
        # pelvis: SOMA hip-socket midpoint in the hips frame; leg-length scale
        ll, rl = sk.index("LeftLeg"), sk.index("RightLeg")
        self.socket = 0.5 * (sk.reference[ll, :3] + sk.reference[rl, :3])
        leg = lambda s, a, b, c: np.linalg.norm(s.reference[s.index(b), :3]) + np.linalg.norm(s.reference[s.index(c), :3])  # noqa: E731
        self.scale = leg(sk, "LeftLeg", "LeftShin", "LeftFoot") / leg(src, "LeftHip", "LeftKnee", "LeftAnkle")
        self.src_hips = (src.index("LeftHip"), src.index("RightHip"))

    def __call__(self, src_local: np.ndarray) -> np.ndarray:
        sk, F = self.sk, src_local.shape[0]
        sw = self.src.fk(src_local)
        world = np.zeros((F, sk.num_joints, 4))
        for j in range(sk.num_joints):
            p = sk.parents[j]
            if j in self.map:
                world[:, j] = m3.quat_mul(sw[:, self.map[j], 3:], self.tpose[j])
            else:
                world[:, j] = self.cal_local[j] if p < 0 else m3.quat_mul(world[:, p], self.cal_local[j])
        out = np.broadcast_to(sk.reference, (F,) + sk.reference.shape).copy()
        for j in range(sk.num_joints):
            p = sk.parents[j]
            q = world[:, j] if p < 0 else m3.quat_mul(m3.quat_conj(world[:, p]), world[:, j])
            q = m3.normalize(q)
            out[:, j, 3:] = np.where(q[:, 3:4] < 0, -q, q)
        mid = 0.5 * (sw[:, self.src_hips[0], :3] + sw[:, self.src_hips[1], :3]) * self.scale
        out[:, 0, :3] = mid - m3.quat_rotate(world[:, 0], np.broadcast_to(self.socket, (F, 3)))
        return out


FEET = ("LeftFoot", "RightFoot", "LeftToeBase", "RightToeBase")


def stance_height(sk: Skeleton, local: np.ndarray) -> float:
    """Typical height of the lowest foot joint while standing on it (10th percentile of the per-frame minimum)."""
    w = sk.fk(local)
    lo = w[:, [sk.index(n) for n in FEET], 1].min(axis=1)
    return float(np.percentile(lo, 10))


def soma_skeleton() -> tuple[Skeleton, float]:
    """SOMA skeleton (from the NVIDIA sample clips) and the stance foot height of their walk clips."""
    from somart.clips import ClipLibrary
    from somart.config import Config
    cfg = Config.load("balanced")
    lib = ClipLibrary(ROOT / cfg.get("clips.root")).load("locomotion/bs_1d_linear/*.bvh")
    hs = [stance_height(lib.skeleton, c.local) for c in lib.clips.values()]
    return lib.skeleton, float(np.median(hs))


def retarget_all(styles: dict[str, list[str]] = STYLES, raw: Path = CACHE / "raw", out: Path = CACHE / "soma") -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    sk, target_h = soma_skeleton()
    cuts, rt, written = frame_cuts(raw), None, []
    for style, kinds in styles.items():
        for k in kinds:
            src = load_bvh(raw / f"{style}_{k}.bvh", drop_static_root=None)
            a, b = cuts[style].get(k, (0, src.num_frames))
            src = Clip(src.skeleton, src.local[a:b], src.fps, src.name).resample(30.0)
            rt = rt or Style100Retargeter(sk, src.skeleton)
            local = rt(src.local)
            local[:, 0, 1] += target_h - stance_height(sk, local)
            f = out / f"{style}_{k}.npz"
            np.savez_compressed(f, local=local.astype(np.float32), fps=30.0, style=style, kind=k,
                                category=CATEGORY.get(k, "100style"), license=LICENSE)
            written.append(f)
            print(f"{f.name}: {len(local)} frames ({len(local) / 30:.0f} s)")
    return written


def load_retargeted(out: Path = CACHE / "soma") -> list[dict]:
    """Retargeted clips as dicts {name, category, local [F, J, 7] float64} (empty if not built)."""
    clips = []
    for f in sorted(out.glob("*.npz")):
        d = np.load(f, allow_pickle=False)
        clips.append({"name": f"100STYLE/{f.stem}.bvh", "category": str(d["category"]), "style": str(d["style"]),
                      "kind": str(d["kind"]), "local": d["local"].astype(np.float64)})
    return clips


# ------------------------------------------------------------------------------------------- check
def check(raw: Path = CACHE / "raw") -> None:
    """Retarget quality: bone-direction agreement, foot height and stance sliding vs the NVIDIA clips."""
    sk, target_h = soma_skeleton()
    src = load_bvh(raw / "Neutral_FW.bvh", drop_static_root=None).resample(30.0)
    rt = Style100Retargeter(sk, src.skeleton)
    local = rt(src.local[300:1200])
    sw, w = src.skeleton.fk(src.local[300:1200]), sk.fk(local)
    print(f"leg scale {rt.scale:.3f}; stance foot height target {target_h:.3f} m")
    errs = []
    for d, (s, dc, sc) in MAP.items():
        if isinstance(sc, tuple) or d in FLAT:
            continue
        a = w[:, sk.index(dc), :3] - w[:, sk.index(d), :3]
        b = sw[:, src.skeleton.index(sc), :3] - sw[:, src.skeleton.index(s), :3]
        cos = np.sum(a * b, -1) / np.linalg.norm(a, axis=-1) / np.linalg.norm(b, axis=-1)
        errs.append((d, float(np.degrees(np.arccos(np.clip(cos, -1, 1))).mean())))
    print("mean bone direction error (deg):", ", ".join(f"{n} {e:.1f}" for n, e in errs))
    for name, (sw_, w_, skel, feet) in {"100STYLE src": (None, sw, src.skeleton, ("LeftToe", "RightToe")),
                                        "SOMA retargeted": (None, w, sk, ("LeftToeBase", "RightToeBase"))}.items():
        slide = []
        for n in feet:
            p = w_[:, skel.index(n)]
            v = np.linalg.norm(np.diff(p[:, [0, 2]], axis=0), axis=1) * 30
            low = p[1:, 1] < np.percentile(p[:, 1], 25)
            slide.append(float(np.median(v[low])))
        print(f"{name}: median toe speed while low {np.mean(slide):.3f} m/s")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["fetch", "retarget", "check", "all"])
    a = ap.parse_args()
    styles = {**STYLES, **{v: STYLE_KINDS for v in WALK_STYLES.values()}}
    if a.cmd in ("fetch", "all"):
        fetch(styles)
    if a.cmd in ("retarget", "all"):
        retarget_all(styles)
    if a.cmd == "check":
        check()
    return 0


if __name__ == "__main__":
    sys.exit(main())
