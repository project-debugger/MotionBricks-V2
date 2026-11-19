#!/usr/bin/env python3
"""Benchmark ONE model variant in a fresh process (clean RSS / peak-memory numbers).

Measures: import + model load time, RSS after load, peak RSS, VRAM (if a GPU backend is used),
end-to-end and network-only latency (mean/p50/p95/p99/min/max), batch throughput, and runs the
evaluation set in two modes:
  pinned  token count fixed to the ground-truth gap -> frame-aligned accuracy vs mocap
  free    model chooses the duration (allowed 8..14) -> token-choice agreement vs FP32

Outputs JSON summary to --out and raw generated motion to --dump (npz) for cross-variant comparison.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

T_START = time.perf_counter()
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime"))


def rss_mb() -> tuple[float, float]:
    cur = peak = 0.0
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                cur = int(line.split()[1]) / 1024
            elif line.startswith("VmHWM:"):
                peak = int(line.split()[1]) / 1024
    return cur, peak


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, help="JSON inference config")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump", default="")
    ap.add_argument("--latency-runs", type=int, default=30)
    ap.add_argument("--batches", default="1,4,16")
    ap.add_argument("--max-requests", type=int, default=0)
    args = ap.parse_args()
    cfg = json.loads(args.variant)

    rss0, _ = rss_mb()
    import numpy as np
    from somart import GenerationRequest, SomaEngine
    from somart import metrics as M
    from somart.backends import make_backend
    from somart.skeleton import Skeleton

    t_imp = time.perf_counter()
    backend = make_backend(cfg)
    t_load = time.perf_counter()
    rss_load, _ = rss_mb()

    data = np.load(ROOT / "datasets/soma_eval/eval_requests.npz")
    skel = Skeleton.from_dict(json.loads((ROOT / "datasets/soma_eval/skeleton.json").read_text()))
    eng = SomaEngine(backend, skel)
    R = len(data["tokens"]) if not args.max_requests else min(args.max_requests, len(data["tokens"]))
    ctx, tgt, gt, lens, toks = (data["context"][:R].astype(np.float64), data["target"][:R].astype(np.float64),
                                data["gt"][:R].astype(np.float64), data["lengths"][:R], data["tokens"][:R])

    # --- warm-up (first runs allocate arenas / JIT kernels) and time-to-first-generation
    t = time.perf_counter()
    eng.generate(ctx[0], tgt[0])
    first_gen_ms = (time.perf_counter() - t) * 1e3
    for _ in range(2):
        eng.generate(ctx[0], tgt[0])

    # --- latency, N=1 end-to-end + per stage
    stage = {k: [] for k in ("encode", "inference", "decode", "post_process", "total")}
    for i in range(args.latency_runs):
        r = i % R
        eng.generate(ctx[r], tgt[r])
        for k in stage:
            stage[k].append(eng.last_timings[k])
    latency = {k: M.summarize(v) for k, v in stage.items()}

    # --- batch throughput (network + codec), requests/s
    throughput = {}
    for n in [int(b) for b in args.batches.split(",") if b]:
        reqs = [GenerationRequest(ctx[i % R], tgt[i % R]) for i in range(n)]
        eng.generate_batch(reqs)
        reps = max(2, 8 // n)
        t = time.perf_counter()
        for _ in range(reps):
            eng.generate_batch(reqs)
        dt = (time.perf_counter() - t) / reps
        throughput[str(n)] = {"ms_per_batch": dt * 1e3, "requests_per_s": n / dt}

    # --- quality: pinned & free modes
    foot_idx = [skel.index(n) for n in M.FOOT_JOINTS]
    dump = {}
    q = {"pinned": {}, "free": {}}
    for mode in ("pinned", "free"):
        rows = {"mpjpe": [], "root_err": [], "rot_err_deg": [], "foot_slide": [], "jerk": [], "b_start": [],
                "b_end": [], "tokens": [], "gt_foot_slide": [], "gt_jerk": [], "foot_slide_h": []}
        locals_, contacts_ = [], []
        for i in range(R):
            allowed = (int(toks[i]),) if mode == "pinned" else tuple(range(8, 15))
            g = eng.generate(ctx[i], tgt[i], allowed_tokens=allowed)
            w = skel.fk(g.local)
            rows["tokens"].append(g.tokens)
            fs = M.foot_slide(w, g.contacts, foot_idx)
            rows["foot_slide"].append(fs)
            rows["foot_slide_h"].append(M.foot_slide(w, M.foot_contacts_from_height(w, foot_idx), foot_idx))
            rows["jerk"].append(M.jerk(w))
            b = M.boundary_accel(skel.fk(ctx[i]), w, skel.fk(tgt[i]))
            rows["b_start"].append(b["start"]); rows["b_end"].append(b["end"])
            if mode == "pinned":
                gw = skel.fk(gt[i, :lens[i]])
                rows["mpjpe"].append(M.mpjpe(w, gw))
                rows["root_err"].append(M.root_error(w, gw))
                rows["rot_err_deg"].append(M.rot_error_deg(g.local, gt[i, :lens[i]]))
                gc = M.foot_contacts_from_height(gw, foot_idx)
                rows["gt_foot_slide"].append(M.foot_slide(gw, gc, foot_idx))
                rows["gt_jerk"].append(M.jerk(gw))
            pad = np.zeros((58,) + g.local.shape[1:], np.float32)
            pad[:g.num_frames] = g.local
            cpad = np.zeros((58, 4), np.float32)
            cpad[:g.num_frames] = g.contacts
            locals_.append(pad); contacts_.append(cpad)
        q[mode] = {k: M.summarize(v) for k, v in rows.items() if v}
        dump[f"{mode}_local"] = np.array(locals_)
        dump[f"{mode}_contacts"] = np.array(contacts_)
        dump[f"{mode}_tokens"] = np.array(rows["tokens"])
        dump[f"{mode}_foot_slide"] = np.array(rows["foot_slide"])
        if mode == "pinned":
            dump["pinned_mpjpe"] = np.array(rows["mpjpe"])
    rss_end, peak = rss_mb()

    vram = None
    if cfg.get("backend") == "torch" and str(cfg.get("device", "cpu")).startswith("cuda"):
        import torch
        vram = torch.cuda.max_memory_allocated() / 2**20

    result = {
        "variant": cfg, "backend_device": getattr(backend, "device", "cpu"),
        "model_bytes": Path(ROOT / cfg["model"]).stat().st_size if "model" in cfg else None,
        "import_seconds": t_imp - T_START, "load_seconds": t_load - t_imp,
        "first_generation_ms": first_gen_ms,
        "rss_mb": {"before_import": rss0, "after_load": rss_load, "end": rss_end, "peak": peak,
                   "model_resident_delta": rss_load - rss0},
        "vram_mb": vram, "latency_ms": latency, "throughput": throughput, "quality": q, "num_requests": R,
        "threads": cfg.get("threads", 0), "pid": os.getpid(),
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    if args.dump:
        np.savez_compressed(args.dump, **dump)
    print(f"[bench] {cfg.get('name')} load {result['load_seconds']:.2f}s  e2e p50 {latency['total']['p50']:.1f}ms "
          f"peak RSS {peak:.0f}MB  mpjpe {q['pinned']['mpjpe']['mean']*100:.2f}cm", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
