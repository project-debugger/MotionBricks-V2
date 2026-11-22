#!/usr/bin/env python3
"""Running, crouching, jumping, sitting, ducking and dodging from the CMU Motion Capture Database, retargeted to SOMA.

The CMU Graphics Lab Motion Capture Database (http://mocap.cs.cmu.edu) is "free for all uses": the data may be
copied, modified or redistributed, and included in products, but not resold directly. It was created with funding
from NSF EIA-0196217. We use a few dozen trials as *targets* for the SOMA controller (the network still generates
every frame), alongside 100STYLE (tools/mocap/style100.py) for walking, strafing, jogging and idling.

  python3 -I tools/mocap/cmu.py fetch      # downloads the ASF/AMC files listed in TRIALS (~60 MB) into data/external/cmu/raw
  python3 -I tools/mocap/cmu.py retarget   # -> data/external/cmu/soma/*.npz (SOMA local poses, 30 fps)
  python3 -I tools/mocap/cmu.py check      # speeds, foot pitch and stance sliding per trial

ASF/AMC: each bone has a rest direction, length and a local axis frame C; an AMC frame gives its rotation R in
that frame (Euler angles, x then y then z about fixed axes). The bone's world rotation is parent * C R C^-1, so in
rest every rotation is the identity: the same convention as BVH, which lets Style100Retargeter do the retargeting
with a CMU joint map. Units: ASF lengths are inches / 0.45.

The downloaded files are untrusted input: they are only parsed as numbers and names.
"""
from __future__ import annotations

import argparse
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "runtime"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from somart import math3d as m3  # noqa: E402
from somart.skeleton import Clip, Skeleton  # noqa: E402
import style100  # noqa: E402

BASE = "http://mocap.cs.cmu.edu/subjects"
CACHE = ROOT / "data/external/cmu"
LICENSE = ("CMU Graphics Lab Motion Capture Database, http://mocap.cs.cmu.edu (free for all uses; created with "
           "funding from NSF EIA-0196217), retargeted onto the SOMA rig")
UNIT = 0.0254 / 0.45  # ASF length unit -> metres

# trial -> (category, frame range in the 30 fps clip or None for all). Picked from the CMU index by description.
TRIALS = {
    # running: subject 9 (fast runs), subject 16 (run/jog, turns, sudden stops), subject 35 (jogs)
    **{f"09_{i:02d}": ("cmu_run", None) for i in range(1, 12)},
    **{f"16_{i:02d}": ("cmu_run", None) for i in (35, 36, 45, 46, 55, 56)},
    **{f"16_{i:02d}": ("cmu_run_turn", None) for i in (37, 38, 39, 40, 41, 42, 43, 44, 48, 49, 50, 51, 52, 53, 54)},
    "16_08": ("cmu_run_stop", None), "16_57": ("cmu_run_stop", None),
    **{f"35_{i:02d}": ("cmu_jog", None) for i in range(17, 27)},
    # actions
    "13_39": ("cmu_jump", None), "13_40": ("cmu_jump", None), "13_41": ("cmu_jump", None), "13_42": ("cmu_jump", None),
    "16_01": ("cmu_jump", None), "16_02": ("cmu_jump", None),
    "143_18": ("cmu_sit", None), "113_15": ("cmu_sit", None),
    "77_09": ("cmu_duck", None), "127_29": ("cmu_duck", None),
    "127_13": ("cmu_sidestep", None), "127_14": ("cmu_sidestep", None),
    # crouch walking (forward, backward) and sneaking sideways: knees bent ~90 deg, pelvis ~0.85 m
    "136_09": ("cmu_crouch", None), "136_10": ("cmu_crouch", None), "136_11": ("cmu_crouch", None), "136_12": ("cmu_crouch", None),
    "139_14": ("cmu_crouch", None), "139_15": ("cmu_crouch", None),
}

# SOMA joint -> (CMU joint, SOMA joint the bone points to, CMU joint or rest vector it points to).
# CMU joints are named after the bone that starts there (lfemur = hip, ltibia = knee, lfoot = ankle, ltoes = toe base).
MAP = {
    "Hips": ("root", "Spine1", "upperback"),
    "Spine1": ("lowerback", "Spine2", "upperback"),
    "Spine2": ("upperback", "Chest", "thorax"),
    "Chest": ("thorax", "Neck1", "lowerneck"),
    "Neck1": ("lowerneck", "Neck2", "upperneck"),
    "Neck2": ("upperneck", "Head", "head"),
    "Head": ("head", "HeadEnd", (0.0, 1.0, 0.0)),
}
for side, c in (("Left", "l"), ("Right", "r")):
    MAP.update({
        f"{side}Shoulder": (f"{c}clavicle", f"{side}Arm", f"{c}humerus"),
        f"{side}Arm": (f"{c}humerus", f"{side}ForeArm", f"{c}radius"),
        f"{side}ForeArm": (f"{c}radius", f"{side}Hand", f"{c}wrist"),
        f"{side}Hand": (f"{c}wrist", f"{side}HandMiddle1", f"{c}hand"),
        f"{side}Leg": (f"{c}femur", f"{side}Shin", f"{c}tibia"),
        f"{side}Shin": (f"{c}tibia", f"{side}Foot", f"{c}foot"),
        f"{side}Foot": (f"{c}foot", f"{side}ToeBase", f"{c}toes"),
        f"{side}ToeBase": (f"{c}toes", f"{side}ToeEnd", f"{c}toes_end"),
    })


class CMURetargeter(style100.Style100Retargeter):
    # CMU's rest foot points along the ground (ankle and toe at about the same height), unlike a standing foot:
    # match only its heading and keep SOMA's own flat-foot pitch, as for 100STYLE.
    MAP, FLAT = MAP, style100.FLAT
    HIPS = ("lfemur", "rfemur")
    LEG = ("lfemur", "ltibia", "lfoot")


# ------------------------------------------------------------------------------------------ parsing
def _euler(deg: np.ndarray) -> np.ndarray:
    """Rotations about fixed x, then y, then z (degrees [..., 3]) -> quaternions xyzw."""
    a = np.radians(deg)
    qx = m3.quat_from_axis_angle(np.array([1.0, 0, 0]), a[..., 0])
    qy = m3.quat_from_axis_angle(np.array([0, 1.0, 0]), a[..., 1])
    qz = m3.quat_from_axis_angle(np.array([0, 0, 1.0]), a[..., 2])
    return m3.quat_mul(qz, m3.quat_mul(qy, qx))


def load_asf(path: Path) -> tuple[Skeleton, dict]:
    """ASF -> Skeleton with identity rest rotations; joints at bone starts, plus '<bone>_end' joints at the leaves."""
    text = path.read_text(errors="replace")
    bones, cur = {}, None
    sec = None
    hierarchy = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(":"):
            sec = line.split()[0][1:]
            continue
        tok = line.split()
        if sec == "bonedata":
            if tok[0] == "begin":
                cur = {"dof": [], "axis": np.zeros(3), "dir": np.zeros(3), "len": 0.0}
            elif tok[0] == "end":
                bones[cur["name"]] = cur
            elif tok[0] == "name":
                cur["name"] = tok[1]
            elif tok[0] == "direction":
                cur["dir"] = np.array([float(x) for x in tok[1:4]])
            elif tok[0] == "length":
                cur["len"] = float(tok[1])
            elif tok[0] == "axis":
                if tok[4].upper() != "XYZ":
                    raise ValueError(f"{path.name}: axis order {tok[4]} not supported")
                cur["axis"] = np.array([float(x) for x in tok[1:4]])
            elif tok[0] == "dof":
                cur["dof"] = [t.lower() for t in tok[1:]]
        elif sec == "hierarchy" and tok[0] not in ("begin", "end"):
            hierarchy.append((tok[0], tok[1:]))
    children = {p: cs for p, cs in hierarchy}
    names, parents, offsets = [], [], []

    def add(name, parent, offset):
        names.append(name); parents.append(parent); offsets.append(offset)
        return len(names) - 1

    def visit(bone, parent_idx, offset):
        j = add(bone, parent_idx, offset)
        vec = bones[bone]["dir"] * bones[bone]["len"] * UNIT if bone != "root" else np.zeros(3)
        kids = children.get(bone, [])
        for k in kids:
            visit(k, j, vec)
        if not kids and bone != "root":
            add(bone + "_end", j, vec)

    visit("root", -1, np.zeros(3))
    ref = np.concatenate([np.array(offsets), m3.quat_identity((len(names),))], axis=-1)
    axes = {n: _euler(b["axis"]) for n, b in bones.items()}
    axes["root"] = m3.quat_identity()
    dofs = {n: b["dof"] for n, b in bones.items()}
    return Skeleton(names, np.array(parents), ref, name=path.stem), {"axes": axes, "dofs": dofs}


def load_amc(path: Path, sk: Skeleton, info: dict, fps: float) -> Clip:
    """AMC frames -> Clip of local transforms [F, J, 7] (root position in metres)."""
    frames, cur = [], None
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", ":")):
            continue
        if re.fullmatch(r"\d+", line):
            cur = {}
            frames.append(cur)
            continue
        if cur is not None:
            tok = line.split()
            cur[tok[0]] = np.array([float(x) for x in tok[1:]])
    F, J = len(frames), sk.num_joints
    local = np.broadcast_to(sk.reference, (F, J, 7)).copy()
    for j, name in enumerate(sk.names):
        if name == "root":
            v = np.stack([f["root"] for f in frames])
            local[:, 0, :3] = v[:, :3] * UNIT
            local[:, 0, 3:] = _euler(v[:, 3:6])
            continue
        dof = info["dofs"].get(name, [])
        if not dof or name not in frames[0]:
            continue
        eul = np.zeros((F, 3))
        vals = np.stack([f[name] for f in frames])
        for k, d in enumerate(dof):
            if d in ("rx", "ry", "rz"):
                eul[:, "xyz".index(d[1])] = vals[:, k]
        C = info["axes"][name]
        q = m3.quat_mul(C, m3.quat_mul(_euler(eul), m3.quat_conj(C)))
        local[:, j, 3:] = m3.normalize(q)
    return Clip(sk, local, fps, path.stem)


# --------------------------------------------------------------------------------------------- fetch
def _get(url: str, dst: Path) -> None:
    with urllib.request.urlopen(url, timeout=120) as r:
        data = r.read()
    if len(data) < 100 or b"<html" in data[:300].lower():
        raise RuntimeError(f"{url}: unexpected response")
    dst.write_bytes(data)


def fetch(trials=TRIALS, out: Path = CACHE / "raw") -> None:
    out.mkdir(parents=True, exist_ok=True)
    for t in trials:
        subj = t.split("_")[0]
        for name in (f"{subj}.asf", f"{t}.amc"):
            dst = out / name
            if not dst.exists():
                _get(f"{BASE}/{int(subj):02d}/{name}", dst)
                print(f"{name}: {dst.stat().st_size / 1e6:.1f} MB")
    (out / "README.txt").write_text(LICENSE + "\n")


FPS = 120  # every trial in TRIALS was captured at 120 fps (CMU index)


def load_trial(t: str, raw: Path = CACHE / "raw") -> Clip:
    subj = t.split("_")[0]
    sk, info = load_asf(raw / f"{subj}.asf")
    return load_amc(raw / f"{t}.amc", sk, info, FPS)


def retarget_all(trials=TRIALS, raw: Path = CACHE / "raw", out: Path = CACHE / "soma") -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    sk, target_h = style100.soma_skeleton()
    written = []
    for t, (cat, cut) in trials.items():
        src = load_trial(t, raw).resample(30.0)
        local = style100.hinge_arms(sk, CMURetargeter(sk, src.skeleton)(src.local))  # one retargeter per subject skeleton
        if cut:
            local = local[cut[0]:cut[1]]
        local[:, 0, 1] += target_h - style100.stance_height(sk, local)
        f = out / f"{t}.npz"
        np.savez_compressed(f, local=local.astype(np.float32), fps=30.0, trial=t, category=cat, license=LICENSE)
        written.append(f)
        print(f"{t} ({cat}): {len(local)} frames ({len(local) / 30:.1f} s)")
    return written


def load_retargeted(out: Path = CACHE / "soma") -> list[dict]:
    """Retargeted trials as dicts {name, category, local [F, J, 7] float64} (empty if not built)."""
    clips = []
    for f in sorted(out.glob("*.npz")):
        d = np.load(f, allow_pickle=False)
        clips.append({"name": f"CMU/{f.stem}", "category": str(d["category"]), "trial": str(d["trial"]),
                      "local": d["local"].astype(np.float64)})
    return clips


def check(out: Path = CACHE / "soma") -> None:
    """Per trial: ground speed, stance foot pitch (SOMA bind: about 21 deg) and toe sliding while low."""
    sk, _ = style100.soma_skeleton()
    F_, T_ = sk.index("LeftFoot"), sk.index("LeftToeBase")
    for c in load_retargeted(out):
        w = sk.fk(c["local"])
        v = np.linalg.norm(np.diff(w[:, 0, [0, 2]], axis=0), axis=1) * 30
        d = w[:, T_, :3] - w[:, F_, :3]
        pitch = np.degrees(np.arctan2(-d[:, 1], np.linalg.norm(d[:, [0, 2]], axis=1)))
        low = (w[:, T_, 1] < np.percentile(w[:, T_, 1], 30)) & (w[:, F_, 1] < np.percentile(w[:, F_, 1], 30))
        toe = np.linalg.norm(np.diff(w[:, T_, :][:, [0, 2]], axis=0), axis=1) * 30
        print(f"{c['name']:12s} {c['category']:13s} speed p50 {np.median(v):4.2f} p90 {np.percentile(v, 90):4.2f} m/s, "
              f"stance pitch {np.median(pitch[low]):5.1f} deg, toe slide {np.median(toe[low[1:]]):.2f} m/s, "
              f"pelvis {w[:, 0, 1].mean():.2f} m")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["fetch", "retarget", "check", "all"])
    a = ap.parse_args()
    if a.cmd in ("fetch", "all"):
        fetch()
    if a.cmd in ("retarget", "all"):
        retarget_all()
    if a.cmd in ("check", "all"):
        check()
    return 0


if __name__ == "__main__":
    sys.exit(main())
