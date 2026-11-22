#!/usr/bin/env python3
"""Measure the SOMA network's own hinge axes (elbows, knees, toes) on its raw output.

The network bends each elbow, knee and toe about one axis of the rig (median 1-5 deg off it). Retargeted mocap
does not: each source rig's elbow frame differs (the 100STYLE and CMU elbow axes are 16-30 deg off the network's),
so the joint limits and the retargeters (style100.hinge_arms) use the network's axes, measured here.

    cd site && node build/serve.mjs 8123 --isolate &     # then, in site/:
    node build/tools/gen_bank.cjs 8123 0 --raw           # SOMA output without joint limits -> data/raw_output.*
    python3 -I tools/fit_hinges.py                        # -> configs/skeletons/soma_hinges.json

Output: per joint the unit axis (scaled-axis space of the local rotation) and the 1st/99th percentile flexion (rad).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime"))
from somart import contract as C  # noqa: E402
from somart import math3d as m3  # noqa: E402

JOINTS = [f"{s}{j}" for s in ("Left", "Right") for j in ("ForeArm", "Shin", "ToeBase")]
OUT = ROOT / "configs/skeletons/soma_hinges.json"


def main() -> int:
    meta = json.loads((ROOT / "site/data/raw_output.json").read_text())
    raw = (ROOT / "site/data/raw_output.bin").read_bytes()
    names, J = meta["joints"], len(meta["joints"])
    q = np.concatenate([np.frombuffer(raw, np.int16, count=e["frames"] * J * 4, offset=e["offset"] + e["frames"] * 12)
                        .reshape(e["frames"], J, 4) / 32767.0 for e in meta["entries"]])
    out = {"note": "SOMA network hinge axes, measured on its raw output (tools/fit_hinges.py)", "frames": int(len(q)), "joints": {}}
    for name in JOINTS:
        r = m3.quat_to_scaled_axis(m3.quat_abs(m3.normalize(q[:, names.index(name)])))
        ax = np.linalg.svd(r, full_matrices=False)[2][0]
        c = r @ ax
        if np.median(c) > 0:  # flexion negative, as in the rig's other limits
            ax, c = -ax, -c
        off = np.degrees(np.median(np.linalg.norm(r - np.outer(c, ax), axis=1)))
        out["joints"][name] = {"axis": ax.tolist(), "lo": float(np.percentile(c, 1)), "hi": float(np.percentile(c, 99)),
                               "off_axis_deg": round(float(off), 2)}
        print(f"{name:13s} off-axis {off:4.1f} deg, flexion {np.degrees(np.percentile(c, 1)):.0f}..{np.degrees(np.percentile(c, 99)):.0f} deg")
    assert set(JOINTS) <= set(C.JOINT_NAMES)
    OUT.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
