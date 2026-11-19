#!/usr/bin/env python3
"""Learn anatomical joint limits (axes, ranges, relaxed finger pose) from the eval mocap.

Writes configs/skeletons/soma_joint_limits.json (aggregate statistics only, no motion).
Requires datasets/build_eval_set.py to have been run.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime"))
from somart.anatomy import JointLimits  # noqa: E402
from somart.skeleton import Skeleton  # noqa: E402

d = np.load(ROOT / "datasets/soma_eval/eval_requests.npz")
sk = Skeleton.from_dict(json.loads((ROOT / "datasets/soma_eval/skeleton.json").read_text()))
gt = np.concatenate([d["gt"][i, :d["lengths"][i]] for i in range(len(d["lengths"]))]).astype(float)
lim = JointLimits.fit(gt, sk)
lim.save(ROOT / "configs/skeletons/soma_joint_limits.json")
print(f"{len(lim.limits)} constrained joints, {len(lim.relaxed)} relaxed joints from {len(gt)} mocap frames")
