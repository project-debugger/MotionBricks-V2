# SOMA model-variant benchmark

Generated 20261119-133535 by `benchmarks/run_benchmarks.py`.

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
| fp32 | 758 | 3.13 | 86 | 81.3 | 103.8 | 71.6 | 10.8 | 19.8 | 25.7 | 1213 | 1316 |
| fp32_ortopt | 757 | 2.04 | 97 | 66.1 | 80.7 | 58.3 | 6.6 | 21.1 | 32.4 | 1207 | 1312 |
| fp16 | 380 | 2.70 | 138 | 72.6 | 100.3 | 63.6 | 7.7 | 17.7 | 39.0 | 975 | 1209 |
| int8 | 262 | 1.33 | 46 | 48.7 | 67.0 | 38.0 | 9.9 | 34.4 | 39.4 | 447 | 573 |
| int8_pose | 350 | 1.53 | 42 | 46.4 | 57.3 | 37.0 | 7.2 | 37.2 | 44.2 | 547 | 674 |
| int4 | 199 | 1.20 | 85 | 42.9 | 51.0 | 35.8 | 6.7 | 30.3 | 31.9 | 399 | 529 |
| int4_pose | 305 | 1.42 | 67 | 66.7 | 82.8 | 55.8 | 10.4 | 21.9 | 30.1 | 510 | 641 |
| int8_lowmem | 262 | 1.71 | 74 | 45.8 | 68.8 | 37.4 | 7.4 | 35.2 | 40.0 | 449 | 552 |
| torch_fp32 | n/a | 2.63 | 141 | 118.9 | 151.5 | 110.2 | 7.8 | 14.2 | 29.2 | 1033 | 1797 |
| torch_bf16 | n/a | 2.36 | 598 | 103.0 | 132.2 | 96.3 | 6.2 | 3.0 | 11.6 | 1033 | 1796 |

## Relative latency, interleaved in one process (drift-cancelled)

| variant @ threads | p50 ms | p95 ms | min ms | speed-up vs FP32 (same threads) |
|---|---|---|---|---|
| int8@1t | 47.7 | 57.5 | 43.9 | x2.64 |
| int8_pose@1t | 62.1 | 80.6 | 57.8 | x2.04 |
| int4@1t | 91.3 | 108.6 | 86.4 | x1.38 |
| int4_pose@1t | 98.7 | 108.3 | 90.9 | x1.31 |
| fp32_ortopt@1t | 127.1 | 143.4 | 118.2 | x0.99 |
| fp32@1t | 127.9 | 144.6 | 114.7 | x1.00 |
| fp16@1t | 128.6 | 156.5 | 117.6 | x0.98 |
| int8@2t | 40.9 | 50.7 | 35.0 | x2.04 |
| int8_pose@2t | 48.9 | 62.2 | 41.1 | x1.69 |
| int4@2t | 62.3 | 74.4 | 54.6 | x1.32 |
| int4_pose@2t | 66.7 | 87.9 | 59.2 | x1.24 |
| fp32@2t | 82.1 | 106.3 | 71.0 | x1.00 |
| fp32_ortopt@2t | 83.2 | 96.7 | 71.0 | x0.98 |
| fp16@2t | 85.5 | 98.7 | 76.8 | x0.94 |
| int8@4t | 39.7 | 49.2 | 33.4 | x1.62 |
| int8_pose@4t | 43.6 | 52.4 | 36.9 | x1.54 |
| int4@4t | 49.2 | 59.9 | 43.1 | x1.31 |
| int4_pose@4t | 55.1 | 70.4 | 45.1 | x1.14 |
| fp32@4t | 65.5 | 82.6 | 51.0 | x1.00 |
| fp16@4t | 67.1 | 89.0 | 55.6 | x0.99 |
| fp32_ortopt@4t | 69.5 | 86.6 | 50.3 | x0.97 |

## Animation quality vs held-out mocap (pinned duration)

Lower is better except where noted. *GT* rows are the same metric on the ground-truth clip.

| variant | MPJPE cm | root err cm | rot err deg | foot slide (pred. contacts) cm/s | foot slide (height contacts) cm/s | jerk m/s^3 | join accel ratio start/end |
|---|---|---|---|---|---|---|---|
| fp32 | 6.96 | 2.65 | 2.10 | 10.79 | 12.59 | 119.1 | 3.68 / 3.76 |
| fp32_ortopt | 6.96 | 2.65 | 2.10 | 10.79 | 12.59 | 119.1 | 3.68 / 3.76 |
| fp16 | 6.94 | 2.66 | 2.10 | 10.80 | 12.57 | 119.4 | 3.66 / 3.77 |
| int8 | 8.67 | 3.05 | 2.25 | 11.66 | 13.32 | 139.2 | 3.69 / 3.74 |
| int8_pose | 7.08 | 2.67 | 2.10 | 10.63 | 12.49 | 122.7 | 3.68 / 3.72 |
| int4 | 13.28 | 4.80 | 3.29 | 18.60 | 18.71 | 150.2 | 4.77 / 4.27 |
| int4_pose | 7.30 | 2.67 | 2.13 | 10.68 | 12.59 | 123.2 | 3.76 / 3.82 |
| int8_lowmem | 8.67 | 3.05 | 2.25 | 11.66 | 13.32 | 139.2 | 3.69 / 3.74 |
| torch_fp32 | 6.96 | 2.65 | 2.10 | 10.79 | 12.59 | 119.1 | 3.68 / 3.76 |
| torch_bf16 | 7.13 | 2.75 | 2.13 | 11.67 | 13.43 | 166.7 | 2.67 / 2.87 |
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
