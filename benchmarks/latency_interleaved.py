#!/usr/bin/env python3
"""Drift-free relative latency: load every ONNX variant in ONE process and interleave runs.

Shared/cloud VMs show large run-to-run variance (CPU frequency, noisy neighbours). Timing
variants back-to-back in separate processes confounds that drift with real differences.
Here each round runs every (variant, threads) pair once in a random order, so drift hits all
of them equally; we report p50/p95 plus each variant's speed-up vs FP32 in the same round.
Thread-pool spinning is disabled so an idle session's workers do not steal CPU from the next one.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime"))
from somart import GenerationRequest, SomaEngine  # noqa: E402
from somart.backends import OrtBackend  # noqa: E402
from somart.skeleton import Skeleton  # noqa: E402

ONNX = ROOT / "models/soma/optimized/onnx"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="fp32,fp32_ortopt,fp16,int8,int8_pose,int4,int4_pose")
    ap.add_argument("--threads", default="4")
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    d = np.load(ROOT / "datasets/soma_eval/eval_requests.npz")
    sk = Skeleton.from_dict(json.loads((ROOT / "datasets/soma_eval/skeleton.json").read_text()))
    reqs = [GenerationRequest(d["context"][i].astype(float), d["target"][i].astype(float))
            for i in range(args.batch)]
    engines = {}
    for v in args.variants.split(","):
        for th in [int(t) for t in args.threads.split(",")]:
            eng = SomaEngine(OrtBackend(ONNX / f"soma_{v}.onnx", intra_threads=th, allow_spinning=False), sk)
            feats = eng.build_inputs(reqs)[0]
            for _ in range(3):
                eng.backend.run(feats)
            engines[(v, th)] = (eng, feats)
            print(f"loaded {v} x{th}", flush=True)
    times = {k: [] for k in engines}
    rng = random.Random(0)
    for _ in range(args.rounds):
        keys = list(engines)
        rng.shuffle(keys)
        for k in keys:
            eng, feats = engines[k]
            t = time.perf_counter()
            eng.backend.run(feats)
            times[k].append((time.perf_counter() - t) * 1e3)
    res = {}
    for (v, th), ts in times.items():
        ts = np.array(ts)
        base = np.array(times.get(("fp32", th), ts))
        res[f"{v}@{th}t"] = {"variant": v, "threads": th, "batch": args.batch, "p50_ms": float(np.median(ts)),
                             "p95_ms": float(np.percentile(ts, 95)), "min_ms": float(ts.min()),
                             "mean_ms": float(ts.mean()),
                             "speedup_vs_fp32_same_round_p50": float(np.median(base / ts))}
    Path(args.out).write_text(json.dumps(res, indent=2))
    for k, r in sorted(res.items(), key=lambda kv: kv[1]["p50_ms"]):
        print(f"{k:22s} p50 {r['p50_ms']:7.1f} ms  p95 {r['p95_ms']:7.1f}  x{r['speedup_vs_fp32_same_round_p50']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
