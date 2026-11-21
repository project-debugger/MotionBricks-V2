# Model optimization: methods, measurements, recommendations

All numbers: Intel Xeon (Sapphire-Rapids-class: AVX-512 VNNI/BF16/FP16, AMX-INT8/BF16), 4 vCPU,
15.7 GB RAM, **no GPU**, onnxruntime 1.32 CPU EP, torch 2.14 CPU. Full tables:
[`benchmarks/REPORT_models.md`](../benchmarks/REPORT_models.md). Re-run: `benchmarks/run_benchmarks.py`.

## Baseline and correctness

`tools/export_onnx.sh` runs NVIDIA's *unmodified* exporter (pinned commit 54fc563, pinned
torch/onnx/numpy versions) on the checksum-verified checkpoints.

- All 434 initializers are byte-identical to NVIDIA's official `mmm_v6_soma_fp32.onnx`, and so is
  the graph topology (6,993 nodes). 8 constant-folded normalisation constants differ by 1 ULP
  (float rounding of `sqrt(std² + eps)` on this CPU).
- On real requests the outputs match the official ONNX to 2.8e-5 (max abs) with the same token
  choice (`tests/test_codec_engine.py::test_matches_official_upstream_onnx`).
- The PyTorch eager graph built from the SafeTensors matches ORT FP32 to 1.7e-5.

## Where the time goes (FP32, ORT profiler)

| Part | Share |
|---|---|
| Pose transformer (16 × 1024, 134 M params in weight matrices) | 56 % |
| Root model transformers (shared 6 layers + token 2 layers) | 16 % |
| Root conv head | 8 % |
| Geometry / feature glue (thousands of tiny ops) | 11 % |
| VQ-VAE decoder | ~5 % |

MatMul is 42 % of kernel time → weight quantization of the pose transformer is the main lever.

## Variants (all in `models/soma/optimized/onnx/`, originals untouched)

| Variant | How | File | Peak RSS | Network p50 (4 thr, interleaved) | MPJPE vs mocap | Same duration as FP32 | Verdict |
|---|---|---|---|---|---|---|---|
| `soma_fp32` | upstream exporter | 758 MB | 1316 MB | 79.4 ms | 6.96 cm | 100 % | reference |
| `soma_fp32_ortopt` | ORT offline graph optimization (EXTENDED) | 757 MB | 1312 MB | ≈ fp32 (×0.93–1.01) | 6.96 cm | 100 % | **27–35 % faster load** (1.8–2.0 s vs 2.5–3.1 s) → ULTRA (CPU) |
| `soma_fp16` | upstream mixed FP16 (LayerNorm/Softmax/geometry FP32) | 380 MB | 1209 MB | 81.1 ms | 6.94 cm | 100 % | no CPU speed-up; GPU choice |
| `soma_int8_pose` | INT8 dynamic quant, **pose transformer only** | 350 MB | 671 MB | **56.2 ms (×1.42)** | 7.08 cm | 98.9 % | **BALANCED / CPU_ONLY** |
| `soma_int8` | INT8 dynamic quant, all network MatMul/Gemm | 262 MB | 575 MB | 48.3 ms (×1.60) | 8.67 cm | 91.4 % | EXTREME PERFORMANCE only |
| `soma_int4_pose` | INT4 weight-only (MatMulNBits, block 32), pose transformer | 305 MB | 644 MB | 62.5 ms (×1.26) | 7.30 cm | 100 % | LOW VRAM (smallest good-quality) |
| `soma_int4` | INT4 weight-only, all networks (Gemm→MatMul rewrite first) | 199 MB | 529 MB | 61.6 ms (×1.31) | 13.28 cm | 78.6 % | **not usable** — quality collapses |
| torch BF16 | autocast BF16 (AMX) | (SafeTensors) | 1797 MB | 160 ms (separate process) | 7.13 cm | 97.9 % | slower than FP32 on CPU at N=1 |

Speed-ups were measured in two independent full runs; the second (above) gave INT8-pose ×1.42 /
×1.59 / ×1.98 at 4 / 2 / 1 threads, the first ×1.54 / ×1.69 / ×2.04. Absolute latencies drift
±20 % on this VM; the interleaved ratios are the reliable numbers.

Single-thread speed-ups are larger (INT8 ×2.5–2.6, INT8-pose ×2.0): INT8-pose on **1 thread
(62–70 ms)** is faster than FP32 on **4 threads (65–79 ms)** — it frees three cores for the game.

### Key findings

1. **The root model and small heads are quantization-sensitive; the pose transformer is not.**
   INT8 on the pose transformer alone costs +0.12 cm MPJPE; extending INT8 to the root model
   costs +1.6 cm and changes the chosen duration in 8.6 % of requests, for only 10–20 % extra speed.
   The root model decides the segment duration (token count) — a discrete choice that amplifies
   small numeric errors.
2. **INT4 everywhere is not viable** (13.3 cm, 79 % duration agreement). INT4 restricted to the
   pose transformer is fine but slower than INT8 on this CPU (MatMulNBits dequantization overhead
   at N=1).
3. **FP16/BF16 give no CPU speed-up at batch 1** (small matrices, cast overhead), but FP16 is
   accuracy-neutral and halves file size → the GPU model of every preset.
4. **Batching** raises throughput: INT8-pose 44–46 requests/s at N=16 vs ~21–24 req/s at N=1
   (end-to-end incl. codec), i.e. one network call can serve many characters.
5. **CPU-side overhead** (encode/decode/endpoint correction) was cut from ~14 ms to ~5–7 ms by
   depth-batched FK (one vectorised op per tree level), and a log/exp fractional-rotation instead
   of generic slerp.

## Runtime optimizations (beyond the network)

| Technique | Where | Effect |
|---|---|---|
| Asynchronous generation on a worker thread (ORT releases the GIL) | `SomaEngine.submit`, controller | game loop never waits for the network: frame p99 60 ms (sync) → 2 ms (async), frames over a 60 FPS budget 118 → 0 per 20 s |
| Core reservation for the game thread | `inference.reserve_cores` | removed frame-budget overruns caused by ORT using every core (pilot: 64 → 0 frames over budget) |
| Latency-compensated, receding-horizon streaming | controller | segments are requested ahead of need; ≤ 1 late result per 20 s session |
| Batching many characters per network call | `SomaEngine.generate_batch` | 2× requests/s at N=16 vs N=1 |
| Depth-batched FK (≈12 vectorised steps instead of 77 per-joint steps) | `math3d.forward_kinematics` | encode 8.3 → 1.9 ms |
| Log-map fractional rotations instead of generic slerp | `math3d.quat_fraction` | endpoint correction 3.7 → 2.4 ms |
| Scalar-math IK + chain-only FK in foot locking | `postprocess` | foot locking 2.2 → 0.22 ms per character per frame |
| Cached config sections (invalidated on change) | controller | no per-frame deep copies |
| ORT memory arena / memory pattern (pooling), spinning control | `inference.*` | tunable latency vs RAM vs CPU sharing |
| Pre-optimized ORT graph on disk | `soma_fp32_ortopt.onnx` | session start 3.1 → 2.0 s |

## Memory

- Memory arena off (`memory_arena: false`) lowers peak RSS slightly (INT8: 573 → 552 MB).
- `inference.memory_limit_mb` makes the loader refuse a model that would not fit (estimate =
  1.6 × file size, from the measured RSS/file ratio).
- VRAM could not be measured: there is no GPU on this machine.

## Not possible here (recorded blockers)

- **TensorRT** requires an NVIDIA GPU; none is present. The presets list
  `TensorrtExecutionProvider`/`CUDAExecutionProvider` first and fall back automatically, so the
  same configs use them on an NVIDIA machine with `onnxruntime-gpu`. Upstream reports 2.5 ms FP32 /
  1.28 ms FP16 on an RTX 4090 with TensorRT (not reproduced here).
- **Efficient-attention kernels / fused attention**: the exported graph expresses attention as
  MatMul + Softmax; ORT's transformer fusions did not match these patterns. Sequence length is
  only ~20–30 tokens, so attention is a small share of time; the MatMul weights dominate.
- **Static (calibrated) INT8 QDQ** for the conv-heavy root/decoder parts was not attempted: the
  root model is the accuracy-sensitive part, and ConvInteger kernels are slow on ORT CPU.

## Web model: one split GGUF, Q5_0 storage (INT8 compute)

`tools/gguf/build_gguf.py` packs the browser's SOMA model and the animal network into one GGUF model split into 13
shards (llama.cpp `split.*` keys, each shard a valid GGUF; `gguf-py` reads them). The 136 M dynamic-INT8 weights of the
pose transformer's matrix multiplies are re-coded with ggml's Q5_0 blocks for storage and decoded back onto the INT8
grid in the browser (integer math, ~0.5 s for all of them on this CPU), so the onnxruntime graph, speed and memory are
those of the INT8 model. The duration head (512x12) and the quantization-sensitive root/decoder weights are left as
they were.

| Storage | Download | Eval MPJPE (pinned) | Same duration as the INT8 file (free / pinned) |
|---|---|---|---|
| INT8 ONNX (before) | 192.8 MB | 7.10 cm | - |
| Q4_0 pose weights | ~133 MB | 7.27 cm | 88 % / 100 % |
| **Q5_0 pose weights (shipped, as rebuilt in the browser)** | **151.8 MB** | **7.11 cm** | **98.4 % / 100 %** |
| Q5_0 on every INT8 weight | ~134 MB | 12.93 cm | 84 % (not usable: root and decoder are sensitive) |

Runtime memory is unchanged: the pose weights stay INT8 in `MatMulInteger`, so RAM is set by the INT8 model, not by
a float copy. VRAM is not used (onnxruntime-web runs on the CPU through WebAssembly).
