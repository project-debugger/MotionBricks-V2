#!/usr/bin/env python3
"""Export the data the browser runtime needs into site/data/.

  skeleton.json        77-joint SOMA rig (names, parents, reference local transforms)
  windows.bin/.json    canonical 4-frame target windows for locomotion matching + action key poses
                       (root position float32, quaternions int16-normalised: 2.5 KB per window)
  clips.bin/.json      continuous 30 fps locomotion/idle clips + mirrored copies (phase-aware targets)
  joint_limits.json    learned anatomical limits (copy of configs/skeletons/soma_joint_limits.json)
  soma_mesh.bin/.json  NVIDIA SOMA body mesh for skinning (from showcase/build/extract_mesh.py)
  config.json          defaults + presets for the web controller (from configs/)

NOTE: the windows are derived from the upstream sample clips (NVIDIA Sample Data Evaluation
License, internal evaluation/testing). Keep the site private unless you have the right to publish it.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime"))
from somart import contract as C  # noqa: E402
from somart.anatomy import JointLimits  # noqa: E402
from somart.clips import ClipLibrary  # noqa: E402
from somart.config import Config  # noqa: E402

OUT = ROOT / "site/data"


def key_frame(clip, key):
    h = clip.local[:, 0, 1]
    last = clip.num_frames - C.POSES_PER_TOKEN
    if isinstance(key, (int, float)) and not isinstance(key, bool):
        return int(np.clip(float(key) * last, 0, last))
    return {"min_height": int(np.clip(np.argmin(h), 0, last)), "max_height": int(np.clip(np.argmax(h), 0, last)),
            "end": last}.get(key, last // 2)


CLIP_SOURCES = ["locomotion/bs_1d_linear", "locomotion/bs_2d_directional", "locomotion/bs_2d_directional_crouch",
                "locomotion/idle/idle_loop_001__A268.bvh", "locomotion/idle/legs_relax_002__A407.bvh",
                "locomotion/idle/neutral_looking_around_R_001__A542.bvh", "locomotion/idle/crouch_idle_001__A529.bvh"]


def export_clips(lib, sk, lim) -> None:
    """Continuous 30 fps clips (plus left/right mirrored copies) for phase-aware target selection.

    clips.bin: per clip, float32 root position [F, 3] then int16 quaternions [F, Jm, 4] (/32767) for the
    Jm joints the network predicts; every other joint takes the relaxed pose from joint_limits.json.

    Real human walking from 100STYLE (CC BY 4.0) is appended when it has been retargeted
    (tools/mocap/style100.py fetch + retarget). Its turns are balanced left/right, so no mirrored copies.
    """
    from somart.mirror import Mirror
    meta_m = json.loads((ROOT / "showcase/build/cache/soma_mesh.json").read_text())
    raw = (ROOT / "showcase/build/cache/soma_mesh.bin").read_bytes()
    ib = meta_m["inverseBind"]
    mirror = Mirror(sk, Mirror.bind_from_inverse_bind(
        np.frombuffer(raw, np.float32, count=ib["length"], offset=ib["offset"]).reshape(-1, 16)))
    joints = [sk.index(n) for n in C.JOINT_NAMES]
    names = [n for s in CLIP_SOURCES for n in lib.find(s)]
    have = set(names)
    entries, blobs, off = [], [], 0
    for name in dict.fromkeys(names):
        clip = lib.clips[name]
        local = clip.local.astype(np.float64).copy()
        local[:, 1:, :3] = sk.reference[None, 1:, :3]
        variants = [(name, local, False)]
        if not (name.endswith("_M.bvh") or name.replace(".bvh", "_M.bvh") in have):
            variants.append((name.replace(".bvh", "_mirror.bvh"), mirror(local), True))
        for vname, loc, mir in variants:
            loc = lim.apply(loc, relax_unmapped=True)
            q = loc[:, joints, 3:]
            q = np.where(q[..., 3:4] < 0, -q, q)
            root = loc[:, 0, :3].astype(np.float32)
            qi = np.round(q * 32767).astype(np.int16)
            blob = root.tobytes() + qi.tobytes()
            entries.append({"name": vname, "category": clip.tags.get("category", ""), "frames": int(len(loc)),
                            "offset": off, "mirrored": mir})
            blobs.append(blob)
            off += len(blob)
    sys.path.insert(0, str(ROOT / "tools/mocap"))
    import style100

    def real_blob(c, offset):
        loc = lim.apply(c["local"], relax_unmapped=True)
        q = loc[:, joints, 3:]
        q = np.where(q[..., 3:4] < 0, -q, q)
        blob = loc[:, 0, :3].astype(np.float32).tobytes() + np.round(q * 32767).astype(np.int16).tobytes()
        return blob, {"name": c["name"], "category": c["category"], "frames": int(len(loc)), "offset": offset,
                      "mirrored": False, "source": "100STYLE"}

    real = style100.load_retargeted()
    base = [c for c in real if c["style"] == "Neutral"]
    for c in base:
        blob, e = real_blob(c, off)
        entries.append(e)
        blobs.append(blob)
        off += len(blob)
    print(f"{len(base)} real walking clips from 100STYLE" if real else
          "100STYLE clips not built (tools/mocap/style100.py fetch && retarget): exporting NVIDIA clips only")
    lic = "Derived from NVIDIA MotionBricks sample clips (NVIDIA Sample Data Evaluation License)"
    if real:
        lic += "; clips named 100STYLE/*: " + style100.LICENSE
    # optional walk styles: one file each, fetched by the page only when the style is chosen.
    # pace = the style's typical forward-walk speed (75th percentile), used as its walking speed.
    styles = []
    (OUT / "styles").mkdir(exist_ok=True)
    for label, st in style100.WALK_STYLES.items():
        cs = [dict(c, local=c["local"][:3600]) for c in real if c["style"] == st]  # at most 2 minutes per kind
        if not cs:
            continue
        s_entries, s_blobs, s_off = [], [], 0
        for c in cs:
            blob, e = real_blob(c, s_off)
            s_entries.append(e)
            s_blobs.append(blob)
            s_off += len(blob)
        (OUT / "styles" / f"{st}.bin").write_bytes(b"".join(s_blobs))
        (OUT / "styles" / f"{st}.json").write_text(json.dumps({
            "fps": lib.fps, "joints": [sk.names[j] for j in joints], "clips": s_entries, "style": st,
            "layout": "per clip: float32 root[F,3] then int16 quat[F,Jm,4] (/32767); other joints: relaxed pose",
            "license": style100.LICENSE}))
        fw = next(c["local"] for c in cs if c["kind"] == "FW")
        sp = np.linalg.norm(np.gradient(fw[:, 0, [0, 2]], axis=0), axis=1) * lib.fps
        pace = round(float(np.percentile(sp[sp > 0.3], 75)) * 20) / 20
        styles.append({"key": st, "label": label, "file": f"data/styles/{st}", "pace": pace,
                       "frames": sum(e["frames"] for e in s_entries), "bytes": s_off})
        print(f"style {label} ({st}): {styles[-1]['frames']} frames, {s_off / 1e6:.2f} MB, pace {pace} m/s")
    with open(OUT / "clips.bin", "wb") as f:
        for b in blobs:
            f.write(b)
    (OUT / "clips.json").write_text(json.dumps({
        "fps": lib.fps, "joints": [sk.names[j] for j in joints], "clips": entries,
        "layout": "per clip: float32 root[F,3] then int16 quat[F,Jm,4] (/32767); other joints: relaxed pose",
        "license": lic, "styles": styles}))
    total = sum(e["frames"] for e in entries)
    print(f"{len(entries)} clips, {total} frames ({off / 1e6:.2f} MB) -> clips.bin")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = Config.load("balanced")
    lib = ClipLibrary(ROOT / cfg.get("clips.root")).load()
    sk = lib.skeleton
    lim = JointLimits.load(ROOT / cfg.get("skeleton.joint_limits")).bind(sk)
    loco = [cfg.get(f"actions.{a}.source") for a in ("walk", "strafe", "crouch", "idle", "crouch_idle")]
    lib.index_windows(loco, stride=3)
    wins = list(lib.windows)
    meta = [{"v": [round(float(w.velocity[0]), 4), round(float(w.velocity[1]), 4)], "yr": round(float(w.yaw_rate), 4),
             "h": round(float(w.height), 4), "cat": w.tags.get("category", ""), "kind": "loco"} for w in wins]
    actions = {}
    for name, a in cfg.get("actions").items():
        if a.get("kind") != "oneshot" and name not in ("idle", "crouch_idle"):
            continue
        names = lib.find(a["source"])
        if not names:
            continue
        clip = lib.clips[names[0]]
        k = key_frame(clip, a.get("key", "middle")) if a.get("kind") == "oneshot" else 30
        w = lib.window_at(names[0], k)
        # clip root motion from frame 0 to the key frame (for anchor: clip)
        ctrl_like = __import__("somart.controller", fromlist=["CharacterController"]).CharacterController
        g0 = ctrl_like._ground_and_yaw(type("S", (), {"sk": sk})(), clip.local[0])
        gk = ctrl_like._ground_and_yaw(type("S", (), {"sk": sk})(), clip.local[k])
        actions[name] = {"window": len(wins), "anchor": a.get("anchor", ""), "kind": a.get("kind", ""),
                         "root0": [float(g0[0][0]), float(g0[0][2]), g0[1]], "rootk": [float(gk[0][0]), float(gk[0][2]), gk[1]]}
        wins.append(w)
        meta.append({"v": [0, 0], "yr": 0, "h": round(float(w.height), 4), "cat": name, "kind": "action"})
    export_clips(lib, sk, lim)
    local = np.stack([lim.apply(w.local, relax_unmapped=True) for w in wins])     # [W, 4, J, 7]
    root = local[:, :, 0, :3].astype(np.float32)                                 # [W, 4, 3]
    q = local[..., 3:]
    q = np.where(q[..., 3:4] < 0, -q, q)
    qi = np.round(q * 32767).astype(np.int16)                                    # [W, 4, J, 4]
    with open(OUT / "windows.bin", "wb") as f:
        f.write(root.tobytes())
        f.write(qi.tobytes())
    (OUT / "windows.json").write_text(json.dumps({
        "count": len(wins), "frames": 4, "joints": sk.num_joints, "layout": "float32 root[W,4,3] then int16 quat[W,4,J,4] (/32767)",
        "windows": meta, "actions": actions,
        "license": "Derived from NVIDIA MotionBricks sample clips (NVIDIA Sample Data Evaluation License)"}))
    (OUT / "skeleton.json").write_text(json.dumps({"names": sk.names, "parents": sk.parents.tolist(),
                                                    "reference": np.round(sk.reference, 6).tolist()}))
    shutil.copy(ROOT / cfg.get("skeleton.joint_limits"), OUT / "joint_limits.json")
    mesh = ROOT / "showcase/build/cache"
    for f in ("soma_mesh.bin", "soma_mesh.json"):
        shutil.copy(mesh / f, OUT / f)
    print(f"{len(wins)} windows ({(OUT / 'windows.bin').stat().st_size / 1e6:.2f} MB), {len(actions)} actions -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
