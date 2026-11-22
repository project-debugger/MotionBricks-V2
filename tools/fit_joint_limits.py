#!/usr/bin/env python3
"""Fit anatomical joint limits (axes, ranges, relaxed hand pose) on the freely licensed clips.

Writes configs/skeletons/soma_joint_limits.json and site/data/joint_limits.json (aggregate statistics only, no
motion). Needs the retargeted 100STYLE and CMU clips (see tools/export_web_data.py, which also runs this).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import export_web_data as E  # noqa: E402

if __name__ == "__main__":
    sk, _ = E.style100.soma_skeleton()
    lib, _ = E.load_free(sk)
    E.fit_limits(lib, sk)
