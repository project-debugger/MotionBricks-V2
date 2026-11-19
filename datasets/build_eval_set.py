#!/usr/bin/env python3
"""Build the SOMA inbetweening evaluation set from the upstream sample BVH clips.

For each clip (resampled to 30 Hz) we cut windows: 4 context poses, a gap, 4 target poses.
The gap is chosen so the generated length equals the true gap exactly:
    valid_frames(T) = 4T - 6 = gap + 1   ->   gap = 4T - 7
and only that token count T is allowed, so every generated frame has a ground-truth frame.

The BVH clips are under the NVIDIA Sample Data Evaluation License (internal evaluation and
testing of NVIDIA technologies). The derived arrays are therefore written to
datasets/soma_eval/ which is git-ignored; only this script and the window manifest are
committed.

Usage: .venv/bin/python datasets/build_eval_set.py [--stride 24] [--per-clip 6]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime"))
from somart.skeleton import load_bvh  # noqa: E402

CLIPS = ROOT / "third_party" / "motionbricks" / "assets" / "motions" / "soma"
OUT = ROOT / "datasets" / "soma_eval"
TOKEN_CYCLE = (8, 10, 12, 14, 9, 11, 13)


def category(rel: str) -> str:
    parts = rel.split("/")
    return parts[1] if parts[0] == "locomotion" else parts[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=24)
    ap.add_argument("--per-clip", type=int, default=6)
    args = ap.parse_args()
    files = sorted(CLIPS.rglob("*.bvh"))
    if not files:
        print("No clips found. Run: python3 tools/fetch_upstream.py --groups rig,motions", file=sys.stderr)
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    ctx, tgt, gt, lens, toks, meta = [], [], [], [], [], []
    # Reference rig: the bone offsets shared by the majority of clips (only Hips, which is
    # position-animated, differs materially in the others).
    ref_skel = load_bvh(CLIPS / "locomotion/idle/idle_loop_001__A268.bvh").skeleton
    max_gt = 4 * max(TOKEN_CYCLE) - 6
    k = 0
    for f in files:
        rel = str(f.relative_to(CLIPS))
        clip = load_bvh(f).resample(30.0)
        if clip.skeleton.names != ref_skel.names:
            print(f"skip {rel}: different joint set")
            continue
        # Largest bone-offset difference to the reference rig, excluding position-animated Hips.
        offs = float(np.abs(clip.skeleton.reference[1:, :3] - ref_skel.reference[1:, :3]).max())
        F = clip.num_frames
        n_here = 0
        t0 = 3
        while n_here < args.per_clip:
            T = TOKEN_CYCLE[k % len(TOKEN_CYCLE)]
            gap = 4 * T - 7
            if t0 + gap + 3 >= F:
                break
            c = clip.local[t0 - 3:t0 + 1]
            tg = clip.local[t0 + gap:t0 + gap + 4]
            g = np.zeros((max_gt,) + clip.local.shape[1:])
            n = gap + 1
            g[:n] = clip.local[t0:t0 + n]
            ctx.append(c); tgt.append(tg); gt.append(g); lens.append(n); toks.append(T)
            meta.append({"clip": rel, "category": category(rel), "t0": t0, "gap": gap, "tokens": T,
                         "bone_offset_delta_m": float(offs)})
            n_here += 1
            k += 1
            t0 += args.stride
    np.savez_compressed(OUT / "eval_requests.npz", context=np.array(ctx, np.float32),
                        target=np.array(tgt, np.float32), gt=np.array(gt, np.float32),
                        lengths=np.array(lens), tokens=np.array(toks))
    (OUT / "skeleton.json").write_text(json.dumps(ref_skel.to_dict()))
    manifest = {"source": "third_party/motionbricks/assets/motions/soma (upstream @54fc563, LFS-verified)",
                "license_note": "Derived from NVIDIA Sample Data Evaluation License clips - local use only",
                "fps": 30, "num_requests": len(meta), "stride": args.stride, "per_clip": args.per_clip,
                "windows": meta}
    (ROOT / "datasets" / "eval_manifest.json").write_text(json.dumps(manifest, indent=1))
    cats = {}
    for m in meta:
        cats[m["category"]] = cats.get(m["category"], 0) + 1
    print(f"{len(meta)} requests from {len(files)} clips; categories: {cats}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
