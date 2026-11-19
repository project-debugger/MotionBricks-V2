#!/usr/bin/env python3
"""Convert a folder of BVH clips on the SOMA skeleton into model-ready 30 Hz arrays.

This is the data-preparation step for future fine-tuning / evaluation on *your own licensed*
motion (see docs/TRAINING.md). It records provenance: a licence string is mandatory and is
written into the manifest next to every clip.

Per clip it stores: local poses [F, J, 7] (30 Hz), world positions of the 30 model joints
[F, 30, 3], heuristic foot contacts [F, 4], root ground velocity [F, 2], heading [F] and, with
--windows, encoded MotionBricks input windows (4 context + 4 target poses, like the eval set).

Usage:
  .venv/bin/python datasets/prepare_motion_dataset.py --src my_mocap/ --out datasets/my_set \
      --license "CC-BY-4.0 (source: ...)" [--windows --gap-tokens 8,10,12]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime"))
from somart import codec  # noqa: E402
from somart import contract as C  # noqa: E402
from somart import metrics as M  # noqa: E402
from somart.skeleton import load_bvh  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder with .bvh files (searched recursively)")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--license", required=True, help="licence / provenance of the source data")
    ap.add_argument("--windows", action="store_true", help="also export encoded model input windows")
    ap.add_argument("--gap-tokens", default="8,10,12", help="token counts for window gaps (gap = 4T - 7)")
    ap.add_argument("--stride", type=int, default=16)
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(src.rglob("*.bvh"))
    manifest = {"license": args.license, "source": str(src), "fps": C.SAMPLE_RATE, "clips": [], "skipped": []}
    windows = {"context": [], "target": [], "clip": [], "t0": [], "tokens": []}
    for f in files:
        rel = str(f.relative_to(src))
        try:
            clip = load_bvh(f).resample(C.SAMPLE_RATE)
            binding = codec.SkeletonBinding.bind(clip.skeleton)
        except ValueError as exc:
            manifest["skipped"].append({"clip": rel, "reason": str(exc)})
            continue
        sk = clip.skeleton
        world = sk.fk(clip.local)
        foot_idx = [sk.index(n) for n in C.FOOT_CONTACT_JOINTS]
        contacts = M.foot_contacts_from_height(world, foot_idx).astype(np.float32)
        hips = world[:, 0, :3]
        vel = np.zeros((len(hips), 2))
        vel[1:] = np.diff(hips[:, [0, 2]], axis=0) * C.SAMPLE_RATE
        heading = codec.heading_from_hips(world[:, binding.right_leg, :3], world[:, binding.left_leg, :3])
        name = rel.replace("/", "__").removesuffix(".bvh")
        np.savez_compressed(out / f"{name}.npz", local=clip.local.astype(np.float32),
                            model_joint_positions=world[:, binding.model_to_pose, :3].astype(np.float32),
                            contacts=contacts, root_velocity=vel.astype(np.float32),
                            heading=heading.astype(np.float32))
        manifest["clips"].append({"clip": rel, "file": f"{name}.npz", "frames": clip.num_frames,
                                  "seconds": round(clip.duration, 3), "joints": sk.num_joints})
        if args.windows:
            for T in [int(t) for t in args.gap_tokens.split(",")]:
                gap = 4 * T - 7
                for t0 in range(3, clip.num_frames - gap - 4, args.stride):
                    windows["context"].append(clip.local[t0 - 3:t0 + 1].astype(np.float32))
                    windows["target"].append(clip.local[t0 + gap:t0 + gap + 4].astype(np.float32))
                    windows["clip"].append(rel); windows["t0"].append(t0); windows["tokens"].append(T)
        (out / "skeleton.json").write_text(json.dumps(sk.to_dict()))
    if args.windows and windows["context"]:
        np.savez_compressed(out / "windows.npz", context=np.array(windows["context"]),
                            target=np.array(windows["target"]), t0=np.array(windows["t0"]),
                            tokens=np.array(windows["tokens"]), clip=np.array(windows["clip"]))
        manifest["num_windows"] = len(windows["context"])
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    total = sum(c["seconds"] for c in manifest["clips"])
    print(f"{len(manifest['clips'])} clips ({total / 60:.1f} min) -> {out}; skipped {len(manifest['skipped'])}"
          + (f"; {manifest.get('num_windows', 0)} windows" if args.windows else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
