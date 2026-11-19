#!/usr/bin/env python3
"""Run the full model-variant benchmark matrix and write benchmarks/REPORT_models.md.

  .venv/bin/python benchmarks/run_benchmarks.py            # everything
  .venv/bin/python benchmarks/run_benchmarks.py --quick    # 40 requests, fewer runs
  .venv/bin/python benchmarks/run_benchmarks.py --only fp32,int8

Each variant runs in its own process (clean memory numbers). Relative latency is also measured
interleaved in one process (benchmarks/latency_interleaved.py) to cancel VM drift.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime"))
PY = sys.executable
ONNX = "models/soma/optimized/onnx"

VARIANTS = [
    {"name": "fp32", "backend": "onnxruntime", "model": f"{ONNX}/soma_fp32.onnx", "precision": "fp32"},
    {"name": "fp32_ortopt", "backend": "onnxruntime", "model": f"{ONNX}/soma_fp32_ortopt.onnx", "precision": "fp32"},
    {"name": "fp16", "backend": "onnxruntime", "model": f"{ONNX}/soma_fp16.onnx", "precision": "fp16"},
    {"name": "int8", "backend": "onnxruntime", "model": f"{ONNX}/soma_int8.onnx", "precision": "int8"},
    {"name": "int8_pose", "backend": "onnxruntime", "model": f"{ONNX}/soma_int8_pose.onnx", "precision": "int8"},
    {"name": "int4", "backend": "onnxruntime", "model": f"{ONNX}/soma_int4.onnx", "precision": "int4"},
    {"name": "int4_pose", "backend": "onnxruntime", "model": f"{ONNX}/soma_int4_pose.onnx", "precision": "int4"},
    {"name": "int8_lowmem", "backend": "onnxruntime", "model": f"{ONNX}/soma_int8.onnx", "precision": "int8",
     "memory_arena": False, "memory_pattern": False},
    {"name": "torch_fp32", "backend": "torch", "precision": "fp32"},
    {"name": "torch_bf16", "backend": "torch", "precision": "bf16"},
]


def hardware() -> dict:
    info = {"platform": platform.platform(), "python": platform.python_version(), "machine": platform.machine()}
    try:
        cpu = Path("/proc/cpuinfo").read_text()
        info["cpu"] = next(l.split(":", 1)[1].strip() for l in cpu.splitlines() if l.startswith("model name"))
        flags = next(l for l in cpu.splitlines() if l.startswith("flags")).split()
        info["simd"] = sorted(f for f in flags if f.startswith(("avx512", "amx", "avx2", "avx_vnni", "f16c")))
        info["logical_cpus"] = cpu.count("processor\t:")
        mem = Path("/proc/meminfo").read_text().splitlines()[0]
        info["ram_gb"] = round(int(mem.split()[1]) / 2**20, 1)
    except (OSError, StopIteration):
        pass
    try:
        import onnxruntime as ort
        info["onnxruntime"] = ort.__version__
        info["ort_providers"] = ort.get_available_providers()
    except ImportError:
        pass
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
    except ImportError:
        pass
    info["gpu"] = "none detected (no nvidia-smi / CUDA / ROCm / DirectML provider)" if not info.get(
        "cuda_available") and not any("CUDA" in p or "Tensorrt" in p or "DML" in p for p in info.get(
            "ort_providers", [])) else "present"
    return info


def compare_to_reference(dumps: dict[str, Path], ref: str = "fp32") -> dict:
    from somart import metrics as M
    from somart.skeleton import Skeleton
    sk = Skeleton.from_dict(json.loads((ROOT / "datasets/soma_eval/skeleton.json").read_text()))
    # NpzFile decompresses on every item access: materialise each array exactly once.
    r = {k: v for k, v in np.load(dumps[ref]).items()}
    data = np.load(ROOT / "datasets/soma_eval/eval_requests.npz")
    lens = data["lengths"][:len(r["pinned_tokens"])]
    ref_world = [sk.fk(r["pinned_local"][i, :n].astype(np.float64)) for i, n in enumerate(lens)]
    out = {}
    for name, path in dumps.items():
        d = {k: v for k, v in np.load(path).items()}
        tok_agree = float((d["free_tokens"] == r["free_tokens"]).mean())
        mp = []
        for i, n in enumerate(lens):
            mp.append(M.mpjpe(sk.fk(d["pinned_local"][i, :n].astype(np.float64)), ref_world[i]))
        out[name] = {"free_token_agreement": tok_agree, "pinned_mpjpe_vs_fp32_cm": float(np.mean(mp) * 100),
                     "pinned_mpjpe_vs_fp32_p95_cm": float(np.percentile(mp, 95) * 100)}
    return out


def per_category(dump: Path) -> dict:
    meta = json.loads((ROOT / "datasets/eval_manifest.json").read_text())["windows"]
    errs = np.load(dump)["pinned_mpjpe"]
    cats: dict[str, list] = {}
    for m, e in zip(meta, errs):
        cats.setdefault(m["category"], []).append(e)
    return {k: {"mpjpe_cm": float(np.mean(v) * 100), "n": len(v)} for k, v in sorted(cats.items())}


def fmt(x, nd=1):
    return "n/a" if x is None else f"{x:.{nd}f}"


def write_report(res: dict, path: Path) -> None:
    hw = res["hardware"]
    L = [f"# SOMA model-variant benchmark\n", f"Generated {res['generated_at']} by `benchmarks/run_benchmarks.py`.\n",
         "## Hardware\n", f"- CPU: {hw.get('cpu')} ({hw.get('logical_cpus')} logical), RAM {hw.get('ram_gb')} GB",
         f"- SIMD: {', '.join(hw.get('simd', []))}", f"- GPU: {hw.get('gpu')}  ->  **VRAM: not measurable here**",
         f"- onnxruntime {hw.get('onnxruntime')} providers {hw.get('ort_providers')}; torch {hw.get('torch')}",
         f"- Python {hw.get('python')} on {hw.get('platform')}\n",
         f"Eval set: {res['num_requests']} held-out inbetweening windows from the 38 upstream SOMA sample clips "
         "(see `datasets/build_eval_set.py`). Threads = 4 unless stated.\n",
         "## Performance (separate process per variant)\n",
         "| variant | file MB | load s | 1st gen ms | e2e p50 ms | e2e p95 ms | network p50 ms | codec ms | "
         "N=4 req/s | N=16 req/s | RSS after load MB | peak RSS MB |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in res["variants"].items():
        if "error" in r:
            L.append(f"| {name} | error: {r['error'][:80]} |")
            continue
        lat = r["latency_ms"]
        codec = lat["encode"]["p50"] + lat["decode"]["p50"] + lat["post_process"]["p50"]
        mb = r["model_bytes"] / 1e6 if r.get("model_bytes") else None
        L.append(f"| {name} | {fmt(mb, 0)} | {r['load_seconds']:.2f} | {r['first_generation_ms']:.0f} | "
                 f"{lat['total']['p50']:.1f} | {lat['total']['p95']:.1f} | {lat['inference']['p50']:.1f} | "
                 f"{codec:.1f} | {r['throughput'].get('4', {}).get('requests_per_s', 0):.1f} | "
                 f"{r['throughput'].get('16', {}).get('requests_per_s', 0):.1f} | "
                 f"{r['rss_mb']['after_load']:.0f} | {r['rss_mb']['peak']:.0f} |")
    if res.get("interleaved"):
        L += ["\n## Relative latency, interleaved in one process (drift-cancelled)\n",
              "| variant @ threads | p50 ms | p95 ms | min ms | speed-up vs FP32 (same threads) |", "|---|---|---|---|---|"]
        for k, r in sorted(res["interleaved"].items(), key=lambda kv: (kv[1]["threads"], kv[1]["p50_ms"])):
            L.append(f"| {k} | {r['p50_ms']:.1f} | {r['p95_ms']:.1f} | {r['min_ms']:.1f} | "
                     f"x{r['speedup_vs_fp32_same_round_p50']:.2f} |")
    L += ["\n## Animation quality vs held-out mocap (pinned duration)\n",
          "Lower is better except where noted. *GT* rows are the same metric on the ground-truth clip.\n",
          "| variant | MPJPE cm | root err cm | rot err deg | foot slide (pred. contacts) cm/s | "
          "foot slide (height contacts) cm/s | jerk m/s^3 | join accel ratio start/end |",
          "|---|---|---|---|---|---|---|---|"]
    gt_row = None
    for name, r in res["variants"].items():
        if "error" in r:
            continue
        q = r["quality"]["pinned"]
        L.append(f"| {name} | {q['mpjpe']['mean']*100:.2f} | {q['root_err']['mean']*100:.2f} | "
                 f"{q['rot_err_deg']['mean']:.2f} | {q['foot_slide']['mean']*100:.2f} | "
                 f"{q['foot_slide_h']['mean']*100:.2f} | {q['jerk']['mean']:.1f} | "
                 f"{q['b_start']['mean']:.2f} / {q['b_end']['mean']:.2f} |")
        gt_row = gt_row or q
    if gt_row:
        L.append(f"| *ground truth* | 0 | 0 | 0 | - | {gt_row['gt_foot_slide']['mean']*100:.2f} | "
                 f"{gt_row['gt_jerk']['mean']:.1f} | - |")
    if res.get("agreement"):
        L += ["\n## Agreement with the FP32 reference\n",
              "| variant | same duration chosen (free mode) | MPJPE vs FP32 output cm (mean / p95) |", "|---|---|---|"]
        for k, a in res["agreement"].items():
            L.append(f"| {k} | {a['free_token_agreement']*100:.1f}% | {a['pinned_mpjpe_vs_fp32_cm']:.2f} / "
                     f"{a['pinned_mpjpe_vs_fp32_p95_cm']:.2f} |")
    if res.get("per_category"):
        L += ["\n## FP32 accuracy by motion category\n", "| category | n | MPJPE cm |", "|---|---|---|"]
        for k, c in res["per_category"].items():
            L.append(f"| {k} | {c['n']} | {c['mpjpe_cm']:.2f} |")
    L += ["\n## Notes\n",
          "- *e2e* = encode + network + decode + endpoint correction for one character (N=1).",
          "- Foot slide uses horizontal foot speed while in contact; *pred. contacts* uses the model's own "
          "contact output, *height contacts* the same heuristic (low + slow) for generated and GT motion.",
          "- *join accel ratio*: acceleration at the context->generated and generated->target joins divided by "
          "the median in-sequence acceleration (1.0 = no visible pop).",
          "- VRAM and TensorRT cannot be measured on this machine (no GPU).",
          ]
    path.write_text("\n".join(L) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--skip-interleaved", action="store_true")
    ap.add_argument("--aggregate", default="", help="reuse per-variant JSON in this results dir; skip re-running")
    args = ap.parse_args()
    stamp = Path(args.aggregate).name if args.aggregate else time.strftime("%Y%m%d-%H%M%S")
    out_dir = ROOT / "benchmarks" / "results" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_dir = ROOT / "benchmarks" / "results" / "_dumps"  # large, git-ignored
    dump_dir.mkdir(parents=True, exist_ok=True)
    only = set(args.only.split(",")) if args.only else None
    res = {"generated_at": stamp, "hardware": hardware(), "variants": {}, "num_requests": 0}
    dumps = {}
    for v in VARIANTS:
        if only and v["name"] not in only:
            continue
        cfg = dict(v, threads=args.threads)
        jpath, dpath = out_dir / f"{v['name']}.json", dump_dir / f"{v['name']}.npz"
        cmd = [PY, str(ROOT / "benchmarks/bench_variant.py"), "--variant", json.dumps(cfg), "--out", str(jpath),
               "--dump", str(dpath)]
        if args.quick:
            cmd += ["--max-requests", "40", "--latency-runs", "10", "--batches", "1,4"]
        if args.aggregate and jpath.exists() and dpath.exists():
            res["variants"][v["name"]] = json.loads(jpath.read_text())
            res["num_requests"] = res["variants"][v["name"]]["num_requests"]
            dumps[v["name"]] = dpath
            continue
        print(f"=== {v['name']}", flush=True)
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if p.returncode != 0:
            res["variants"][v["name"]] = {"error": (p.stderr or p.stdout).strip().splitlines()[-1]}
            print(p.stderr[-2000:], flush=True)
            continue
        print(p.stdout.strip().splitlines()[-1], flush=True)
        res["variants"][v["name"]] = json.loads(jpath.read_text())
        res["num_requests"] = res["variants"][v["name"]]["num_requests"]
        dumps[v["name"]] = dpath
    if "fp32" in dumps:
        res["agreement"] = compare_to_reference(dumps)
        res["per_category"] = per_category(dumps["fp32"])
    if not args.skip_interleaved:
        names = ",".join(v["name"] for v in VARIANTS if v["backend"] == "onnxruntime" and "lowmem" not in v["name"]
                         and (not only or v["name"] in only))
        rounds = "10" if args.quick else "30"
        res["interleaved"] = {}
        for th in (1, 2, 4):  # one process per thread count keeps memory < ~6 GB
            il = out_dir / f"interleaved_{th}t.json"
            print(f"=== interleaved latency, {th} thread(s)", flush=True)
            subprocess.run([PY, str(ROOT / "benchmarks/latency_interleaved.py"), "--variants", names,
                            "--threads", str(th), "--rounds", rounds, "--out", str(il)], cwd=ROOT, check=False)
            if il.exists():
                res["interleaved"].update(json.loads(il.read_text()))
    (out_dir / "results.json").write_text(json.dumps(res, indent=2))
    write_report(res, out_dir / "REPORT.md")
    (ROOT / "benchmarks" / "REPORT_models.md").write_text((out_dir / "REPORT.md").read_text())
    (ROOT / "benchmarks" / "results" / "latest_models.json").write_text(json.dumps(res, indent=2))
    print(f"report: {out_dir / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
