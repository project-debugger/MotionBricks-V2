#!/usr/bin/env python3
"""Ablation of the animation clean-up passes on the held-out eval set (pinned duration).

  A raw            network output, no correction
  B upstream       + endpoint correction (what the NVIDIA engine does)
  C +velocity      + velocity-matched joins (somart)
  D +footlock      + foot locking with the model's contacts (somart)
  E +both          + velocity matching + foot locking
  F contact        endpoint correction with contact-aware root ramp (somart)
  G full           contact-aware correction + velocity matching + foot locking (somart default)
  GT               the held-out mocap itself (reference level for every metric)

Usage: .venv/bin/python benchmarks/ablation_postprocess.py --model soma_int8_pose
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime"))
from somart import GenerationRequest, SomaEngine, codec  # noqa: E402
from somart import contract as C  # noqa: E402
from somart import metrics as M  # noqa: E402
from somart.backends import OrtBackend  # noqa: E402
from somart.postprocess import FootLocker  # noqa: E402
from somart.skeleton import Skeleton  # noqa: E402


def foot_lock(sk, seq, contacts):
    fl = FootLocker(sk, [["LeftLeg", "LeftShin", "LeftFoot"], ["RightLeg", "RightShin", "RightFoot"]], [0, 2])
    return np.array([fl.apply(seq[i], contacts[i], 1 / 30) for i in range(len(seq))])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="soma_int8_pose")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-requests", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "benchmarks/results/latest_ablation.json"))
    args = ap.parse_args()
    d = np.load(ROOT / "datasets/soma_eval/eval_requests.npz")
    sk = Skeleton.from_dict(json.loads((ROOT / "datasets/soma_eval/skeleton.json").read_text()))
    eng = SomaEngine(OrtBackend(ROOT / f"models/soma/optimized/onnx/{args.model}.onnx", intra_threads=args.threads), sk)
    R = len(d["tokens"]) if not args.max_requests else args.max_requests
    foot_idx = [sk.index(n) for n in M.FOOT_JOINTS]
    names = ["A_raw", "B_upstream", "C_velocity", "D_footlock", "E_both", "F_contact", "G_full", "GT"]
    rows = {n: {k: [] for k in ("mpjpe", "slide_h", "slide_pred", "jerk", "join_start", "join_end", "end_err")}
            for n in names}
    for i in range(R):
        ctx, tgt = d["context"][i].astype(np.float64), d["target"][i].astype(np.float64)
        n = int(d["lengths"][i])
        gt = d["gt"][i, :n].astype(np.float64)
        g = eng.generate(ctx, tgt, allowed_tokens=(int(d["tokens"][i]),), correct_endpoints=False)
        raw = g.local[None]
        cnt = np.array([g.num_frames])
        up = codec.endpoint_correction(raw, cnt, ctx[None, 3], tgt[None, 0])
        vel = codec.velocity_matching(up, cnt, ctx[None, 2], ctx[None, 3], tgt[None, 0], tgt[None, 1])
        ca = codec.endpoint_correction(raw, cnt, ctx[None, 3], tgt[None, 0], contacts=g.contacts[None])
        ca_vel = codec.velocity_matching(ca, cnt, ctx[None, 2], ctx[None, 3], tgt[None, 0], tgt[None, 1])
        seqs = {"A_raw": raw[0], "B_upstream": up[0], "C_velocity": vel[0],
                "D_footlock": foot_lock(sk, up[0], g.contacts), "E_both": foot_lock(sk, vel[0], g.contacts),
                "F_contact": ca[0], "G_full": foot_lock(sk, ca_vel[0], g.contacts), "GT": gt}
        cw, tw, gw = sk.fk(ctx), sk.fk(tgt), sk.fk(gt)
        for name, seq in seqs.items():
            w = sk.fk(seq)
            r = rows[name]
            r["mpjpe"].append(M.mpjpe(w, gw))
            r["slide_h"].append(M.foot_slide(w, M.foot_contacts_from_height(w, foot_idx), foot_idx))
            if name != "GT":
                r["slide_pred"].append(M.foot_slide(w, g.contacts, foot_idx))
            r["jerk"].append(M.jerk(w))
            b = M.boundary_accel(cw, w, tw)
            r["join_start"].append(b["start"]); r["join_end"].append(b["end"])
            r["end_err"].append(float(np.linalg.norm(w[-1, :, :3] - tw[0, :, :3], axis=-1).mean()))
    out = {"model": args.model, "requests": R,
           "results": {n: {k: float(np.mean(v)) for k, v in r.items() if v} for n, r in rows.items()}}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"{'variant':12s} {'MPJPE cm':>9s} {'slide_h cm/s':>12s} {'slide_pred':>10s} {'jerk':>7s} "
          f"{'join s/e':>11s} {'end err cm':>10s}")
    for n, r in out["results"].items():
        print(f"{n:12s} {r['mpjpe']*100:9.2f} {r['slide_h']*100:12.2f} {r.get('slide_pred', float('nan'))*100:10.2f} "
              f"{r['jerk']:7.1f} {r['join_start']:5.2f}/{r['join_end']:<5.2f} {r['end_err']*100:10.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
