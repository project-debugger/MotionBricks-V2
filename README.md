# MotionBricks SOMA — optimized, configurable runtime

A fast, configurable CPU/GPU runtime and real-time character controller for
[NVIDIA MotionBricks](https://huggingface.co/nvidia/MotionBricks) **SOMA** (human, 30-joint),
built on the untouched released checkpoints.

**What it adds on top of the release**

- **Verified download** of exactly the `soma/` folder (parallel, resumable, SHA-256 against both
  upstream checksum files), and an **ONNX export proven equal to NVIDIA's official one**.
- **Optimized model variants** (ORT-pre-optimized FP32, mixed FP16, INT8, INT4, pose-only
  INT8/INT4) with a measured accuracy/speed/memory trade-off. The recommended **INT8-pose**
  variant runs **1.4–1.5× (4 threads) to 2.0× (1 thread) faster** than FP32 (two independent
  benchmark runs), with **half the RAM**, and
  picks the same motion duration as FP32 in 98.9 % of held-out requests.
- A **NumPy runtime** (`runtime/somart`) — no Warp/TensorRT/CUDA needed — with single, batched and
  asynchronous generation; any ONNX Runtime execution provider; optional PyTorch backend.
- **Animation-quality passes** measured on held-out mocap: velocity-matched joins and foot locking
  cut foot sliding by **37 %** and join pops by **40–49 %**, and improve accuracy 5 % over upstream.
- A **real-time controller**: spring-damper locomotion, latency-compensated receding-horizon
  planning, duration pinned to the prediction horizon (body response **0.41 s vs 1.38 s**),
  inertialization, actions (jump, duck, dodge, sit…); 0.65 ms per frame on the game thread,
  0 dropped 60 FPS frames on a 4-vCPU machine while generating in the background.
- **Everything configurable** in YAML (live changes, validation, per-character / per-action /
  per-skeleton, save/load), 5 hardware-aware **presets**, and a **multi-skeleton architecture** for
  future animal models (templates only — the released model is human-only).

Measured on: Intel Xeon (AVX-512 VNNI, AMX), 4 vCPU, 15.7 GB RAM, **no GPU**.

## Results at a glance

| Model (CPU, ORT 1.32) | File | Peak RAM | Network p50, 4 thr / 1 thr | Accuracy vs mocap (MPJPE) | Same duration as FP32 |
|---|---|---|---|---|---|
| FP32 (reference) | 758 MB | 1316 MB | 79.4 / 149.4 ms | 6.96 cm | 100 % |
| **INT8 pose transformer** (balanced) | 350 MB | 671 MB | **56.2 / 69.5 ms** (×1.42 / ×1.98) | 7.08 cm | 98.9 % |
| INT4 pose transformer (low memory) | 305 MB | 644 MB | 62.5 / 115.8 ms | 7.30 cm | 100 % |
| INT8 everything (extreme) | 262 MB | 575 MB | 48.3 / 57.0 ms | 8.67 cm | 91.4 % |
| Mixed FP16 (GPU choice) | 380 MB | 1209 MB | 81.1 / 149.6 ms | 6.94 cm | 100 % |

Latencies are drift-cancelled (all variants interleaved in one process); absolute values vary
±20 % between runs on this shared VM, speed-ups are stable.

| Animation clean-up (INT8-pose, 187 held-out windows) | MPJPE | Foot slide | Join pop start / end |
|---|---|---|---|
| Upstream endpoint correction | 7.08 cm | 12.5 cm/s | 3.68 / 3.72 |
| **+ velocity matching + foot locking** | **6.69 cm** | **7.8 cm/s** | **1.89 / 2.23** |
| Mocap ground truth | — | 5.2 cm/s | 1.15 / 1.29 |

| Real-time controller (60 Hz scripted session, mean of 3 runs) | Frame p50 / p99 | Frames > 16.7 ms | Body t90 | Tracking err | Foot slide |
|---|---|---|---|---|---|
| balanced, async (default) | 0.65 / 2.0 ms | 0 / 1200 | 0.41 s | 0.13 m | 8.7 cm/s |
| balanced, synchronous generation | 0.63 / 60.1 ms | 118 / 1200 | 0.41 s | 0.13 m | 8.5 cm/s |
| balanced, model-chosen duration (upstream style) | 0.66 / 2.1 ms | 0 / 1200 | 1.38 s | 1.31 m | 8.8 cm/s |
| balanced, no foot locking | 0.44 / 1.7 ms | 0 / 1200 | 0.41 s | 0.13 m | 15.0 cm/s |

Full reports: [`benchmarks/REPORT_models.md`](benchmarks/REPORT_models.md),
[`benchmarks/REPORT_postprocess.md`](benchmarks/REPORT_postprocess.md),
[`benchmarks/REPORT_controller.md`](benchmarks/REPORT_controller.md).

## Install

```bash
# 1. Python env (Python 3.12 + pinned torch/onnx/numpy, same as the upstream exporter)
bash tools/setup_env.sh                       # uses uv if present

# 2. Download + verify the SOMA package (only soma/, never g1/)  -> models/soma/original/
python3 tools/download_soma.py                # resumable; --verify-only re-hashes

# 3. Upstream software at a pinned commit + SOMA LFS assets (rig, sample clips)  -> third_party/
python3 tools/fetch_upstream.py --groups rig,motions        # add ,onnx for NVIDIA's reference ONNX

# 4. Export + optimize                                       -> models/soma/optimized/onnx/
bash tools/export_onnx.sh                     # FP32 + mixed FP16 (upstream exporter, unmodified)
.venv/bin/python tools/optimize_models.py all # ORT-opt, INT8, INT8-pose, INT4, INT4-pose

# 5. Check
.venv/bin/python -m pytest -q                 # 42 tests
.venv/bin/python examples/quickstart.py       # writes benchmarks/output/quickstart_*.bvh
```

GPU: install the matching onnxruntime build (`onnxruntime-gpu`, `-directml`, …); presets pick
GPU providers and the FP16/FP32 model automatically when they are available.

## Use

```python
from somart.config import Config
from somart.hardware import detect, recommend_profile, resolve_inference
from somart.backends import make_backend
from somart.clips import ClipLibrary
from somart.engine import SomaEngine
from somart.controller import CharacterController, Command

hw  = detect()
cfg = Config.load(recommend_profile(hw))            # or "balanced", "custom/my_game.yaml", ...
lib = ClipLibrary(cfg.get("clips.root")).load()
lib.index_windows(["locomotion/bs_1d_linear", "locomotion/bs_2d_directional",
                   "locomotion/bs_2d_directional_crouch", "locomotion/idle"])
engine = SomaEngine(make_backend(resolve_inference(cfg.section("inference"), hw)), lib.skeleton)
hero = CharacterController(engine, lib, cfg)

while running:                                       # your game loop, any frame rate
    pose = hero.update(Command(move=(x, z), gait="run"), dt)   # local transforms [J, 7]
    if jump_pressed:
        hero.trigger("jump")
    cfg.set("locomotion.run_speed", 4.0)             # live tuning
```

Lower level: `engine.generate(context_4_poses, target_1to4_poses)` returns a 30 Hz
`GeneratedMotion` (poses + foot contacts); `generate_batch` serves many characters in one call;
`submit` runs on a worker thread. Export anything with `somart.bvh_writer.write_bvh`.

## Repository layout

```
models/soma/original/      untouched HF soma/ folder (downloaded, verified; not in git)
models/soma/optimized/     ONNX variants (generated; not in git) + manifest.json
models/soma/checksums/     upstream SHA256SUMS, verification report, optimized-file hashes
runtime/somart/            the runtime package (engine, codec, controller, config, ...)
configs/                   default.yaml, presets/, custom/, skeletons/
tools/                     download, fetch upstream, export, optimize, setup, profiles
benchmarks/                benchmark scripts, reports (REPORT_*.md), results/, logs/
datasets/                  eval-set builder + window manifest (derived data not in git)
docs/                      architecture, optimization, animation, configuration, animals,
                           training, limitations
examples/                  quickstart.py
tests/                     pytest suite
third_party/               upstream MotionBricks clone (not in git)
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — model, pipeline, runtime modules, controller
- [Optimization](docs/OPTIMIZATION.md) — every variant, how it was made, measurements, blockers
- [Animation quality & actions](docs/ANIMATION.md) — what improves the feel, action coverage, tuning
- [Configuration](docs/CONFIGURATION.md) — schema, presets, runtime changes, overrides
- [Animals](docs/ANIMALS.md) — multi-skeleton architecture, templates, what a real animal model needs
- [Training](docs/TRAINING.md) — package contents, fine-tuning feasibility (none performed)
- [Limitations](docs/LIMITATIONS.md) — remaining limitations and recorded blockers

## Honesty notes

- No training or fine-tuning was performed; all variants are export/precision/quantization of
  NVIDIA's weights. The released model is **human-only**; animal files are templates.
- No GPU was available: TensorRT and VRAM were not measured.
- Sample clips are evaluation-licensed and stay local (`third_party/`, `datasets/soma_eval/`).

## Licences

Weights: [NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/).
Upstream MotionBricks software: Apache-2.0 (used unmodified from `third_party/`).
Sample motions: NVIDIA Sample Data Evaluation License (not redistributed).
Code in this repository: no licence has been chosen yet (add a `LICENSE` file). Parts of
`runtime/somart` are NumPy ports of Apache-2.0 upstream code — see [NOTICE](NOTICE).
