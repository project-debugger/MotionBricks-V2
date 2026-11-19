# SOMA model-variant benchmark

Generated 20261119-141733 by `benchmarks/run_benchmarks.py`.

## Hardware

- CPU: Intel(R) Xeon(R) Processor @ 2.10GHz (4 logical), RAM 15.7 GB
- SIMD: amx_bf16, amx_int8, amx_tile, avx2, avx512_bf16, avx512_bitalg, avx512_fp16, avx512_vbmi2, avx512_vnni, avx512_vpopcntdq, avx512bw, avx512cd, avx512dq, avx512f, avx512ifma, avx512vbmi, avx512vl, avx_vnni, f16c
- GPU: none detected (no nvidia-smi / CUDA / ROCm / DirectML provider)  ->  **VRAM: not measurable here**
- onnxruntime 1.32.0 providers ['AzureExecutionProvider', 'CPUExecutionProvider']; torch 2.14.0+cpu
- Python 3.12.3 on Linux-6.18.44-fc-v653-x86_64-with-glibc2.39

Eval set: 187 held-out inbetweening windows from the 38 upstream SOMA sample clips (see `datasets/build_eval_set.py`). Threads = 4 unless stated.

## Performance (separate process per variant)

| variant | file MB | load s | 1st gen ms | e2e p50 ms | e2e p95 ms | network p50 ms | codec ms | N=4 req/s | N=16 req/s | RSS after load MB | peak RSS MB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fp32 | 758 | 2.53 | 96 | 76.4 | 93.4 | 64.8 | 11.0 | 21.7 | 31.1 | 1213 | 1316 |
| fp32_ortopt | 757 | 1.84 | 63 | 52.1 | 65.4 | 45.3 | 6.2 | 29.9 | 30.4 | 1207 | 1312 |
| fp16 | 380 | 2.51 | 61 | 58.7 | 75.7 | 51.9 | 7.2 | 23.6 | 28.7 | 983 | 1209 |
| int8 | 262 | 1.48 | 54 | 42.2 | 51.5 | 34.3 | 7.5 | 36.7 | 41.8 | 449 | 575 |
| int8_pose | 350 | 1.68 | 49 | 41.5 | 46.8 | 34.0 | 7.3 | 31.3 | 45.5 | 544 | 671 |
| int4 | 199 | 1.37 | 63 | 54.6 | 64.8 | 45.6 | 7.7 | 27.6 | 30.8 | 399 | 529 |
| int4_pose | 305 | 1.23 | 59 | 51.3 | 61.8 | 44.0 | 6.8 | 21.7 | 29.3 | 513 | 644 |
| int8_lowmem | 262 | 1.58 | 52 | 44.0 | 58.2 | 35.5 | 7.6 | 34.3 | 40.7 | 447 | 549 |
| torch_fp32 | n/a | 2.66 | 105 | 100.4 | 141.2 | 93.9 | 6.7 | 20.8 | 40.1 | 1033 | 1797 |
| torch_bf16 | n/a | 3.15 | 819 | 168.9 | 228.0 | 160.1 | 9.0 | 1.8 | 7.6 | 1033 | 1797 |

## Relative latency, interleaved in one process (drift-cancelled)

| variant @ threads | p50 ms | p95 ms | min ms | speed-up vs FP32 (same threads) |
|---|---|---|---|---|
| int8@1t | 57.0 | 81.9 | 46.9 | x2.52 |
| int8_pose@1t | 69.5 | 89.8 | 63.2 | x1.98 |
| int4@1t | 109.7 | 139.5 | 92.0 | x1.29 |
| int4_pose@1t | 115.8 | 154.8 | 96.4 | x1.25 |
| fp32_ortopt@1t | 140.4 | 170.7 | 129.3 | x1.01 |
| fp32@1t | 149.4 | 168.9 | 122.4 | x1.00 |
| fp16@1t | 149.6 | 171.7 | 126.9 | x0.97 |
| int8@2t | 52.4 | 65.7 | 39.5 | x1.97 |
| int8_pose@2t | 63.0 | 76.8 | 45.8 | x1.59 |
| int4@2t | 75.3 | 106.7 | 59.9 | x1.31 |
| int4_pose@2t | 80.8 | 95.5 | 66.2 | x1.21 |
| fp32@2t | 100.3 | 121.7 | 74.2 | x1.00 |
| fp16@2t | 101.2 | 129.7 | 81.5 | x0.97 |
| fp32_ortopt@2t | 102.0 | 121.6 | 79.6 | x0.95 |
| int8@4t | 48.3 | 59.7 | 40.0 | x1.60 |
| int8_pose@4t | 56.2 | 73.1 | 43.4 | x1.42 |
| int4@4t | 61.6 | 72.5 | 45.9 | x1.31 |
| int4_pose@4t | 62.5 | 74.9 | 52.8 | x1.26 |
| fp32@4t | 79.4 | 110.5 | 60.0 | x1.00 |
| fp16@4t | 81.1 | 110.4 | 67.3 | x0.94 |
| fp32_ortopt@4t | 82.9 | 105.6 | 56.9 | x0.93 |

## Animation quality vs held-out mocap (pinned duration)

Lower is better except where noted. *GT* rows are the same metric on the ground-truth clip.

| variant | MPJPE cm | root err cm | rot err deg | foot slide (pred. contacts) cm/s | foot slide (height contacts) cm/s | jerk m/s^3 | join accel ratio start/end |
|---|---|---|---|---|---|---|---|
| fp32 | 6.96 | 2.65 | 2.09 | 10.79 | 12.59 | 119.1 | 3.68 / 3.76 |
| fp32_ortopt | 6.96 | 2.65 | 2.09 | 10.79 | 12.59 | 119.1 | 3.68 / 3.76 |
| fp16 | 6.94 | 2.66 | 2.09 | 10.80 | 12.57 | 119.4 | 3.66 / 3.77 |
| int8 | 8.67 | 3.05 | 2.24 | 11.66 | 13.32 | 139.2 | 3.69 / 3.74 |
| int8_pose | 7.08 | 2.67 | 2.09 | 10.63 | 12.49 | 122.7 | 3.68 / 3.72 |
| int4 | 13.28 | 4.80 | 3.29 | 18.60 | 18.71 | 150.2 | 4.77 / 4.27 |
| int4_pose | 7.30 | 2.67 | 2.12 | 10.68 | 12.59 | 123.2 | 3.76 / 3.82 |
| int8_lowmem | 8.67 | 3.05 | 2.24 | 11.66 | 13.32 | 139.2 | 3.69 / 3.74 |
| torch_fp32 | 6.96 | 2.65 | 2.09 | 10.79 | 12.59 | 119.1 | 3.68 / 3.76 |
| torch_bf16 | 7.13 | 2.75 | 2.12 | 11.67 | 13.43 | 166.7 | 2.67 / 2.87 |
| *ground truth* | 0 | 0 | 0 | - | 5.16 | 158.0 | - |

## Agreement with the FP32 reference

| variant | same duration chosen (free mode) | MPJPE vs FP32 output cm (mean / p95) |
|---|---|---|
| fp32 | 100.0% | 0.00 / 0.00 |
| fp32_ortopt | 100.0% | 0.00 / 0.00 |
| fp16 | 100.0% | 0.54 / 2.02 |
| int8 | 91.4% | 5.67 / 18.73 |
| int8_pose | 98.9% | 1.96 / 4.81 |
| int4 | 78.6% | 11.88 / 32.57 |
| int4_pose | 100.0% | 3.02 / 8.66 |
| int8_lowmem | 91.4% | 5.67 / 18.73 |
| torch_fp32 | 100.0% | 0.00 / 0.00 |
| torch_bf16 | 97.9% | 2.39 / 6.40 |

## FP32 accuracy by motion category

| category | n | MPJPE cm |
|---|---|---|
| avoidance | 12 | 6.91 |
| bs_1d_linear | 19 | 4.26 |
| bs_2d_directional | 3 | 8.48 |
| bs_2d_directional_crouch | 22 | 5.38 |
| idle | 30 | 2.90 |
| jump | 22 | 15.01 |
| sitting | 16 | 3.39 |
| variations | 63 | 8.28 |

## Notes

- *e2e* = encode + network + decode + endpoint correction for one character (N=1).
- Foot slide uses horizontal foot speed while in contact; *pred. contacts* uses the model's own contact output, *height contacts* the same heuristic (low + slow) for generated and GT motion.
- *join accel ratio*: acceleration at the context->generated and generated->target joins divided by the median in-sequence acceleration (1.0 = no visible pop).
- VRAM and TensorRT cannot be measured on this machine (no GPU).
