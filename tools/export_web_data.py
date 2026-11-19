#!/usr/bin/env python3
"""Export the data the browser runtime needs into site/data/.

  skeleton.json        77-joint SOMA rig (names, parents, reference local transforms)
  windows.bin/.json    canonical 4-frame target windows for locomotion matching + action key poses
                       (root position float32, quaternions int16-normalised: 2.5 KB per window)
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
