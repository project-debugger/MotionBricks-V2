# Remaining limitations and blockers

## Hardware / environment blockers (this machine)

- **No GPU**: TensorRT, CUDA/DirectML/CoreML execution and **VRAM measurement were not possible**.
  Presets list GPU providers first and fall back to CPU; GPU behaviour is unverified here. The
  only GPU numbers are NVIDIA's own (RTX 4090 + TensorRT: 2.50 ms FP32, 1.28 ms FP16, N=1).
- **Noisy shared VM**: run-to-run latency varies by ±20–30 %. Relative comparisons use the
  interleaved benchmark; absolute numbers will differ on other machines.
- **4 vCPUs**: the game loop and generation share cores; generation latency in the controller
  benchmark (~60–70 ms) is higher than in isolation (~44 ms).

## Model limitations (inherited from the released SOMA model)

- Human only, one 30-joint SOMA skeleton. **No animal capability** (see `ANIMALS.md`).
- Conditioning is poses + root constraints only: no text, no scene geometry, no physics.
  Obstacles, contacts with objects and balance are the application's job.
- Each generation covers 0.6–1.9 s (6–16 tokens); longer motion is chained segments.
- Motion quality depends on target poses being in-distribution; jumps are the weakest
  category on the eval set (15 cm MPJPE vs 3–5 cm for idle/locomotion).
- Output can differ across precisions/backends because token choice is discrete (`argmax`);
  differences accumulate when generated motion is fed back as context.

## Optimization limitations

- INT8/INT4 quantization of the **root model** degrades motion measurably; only the pose
  transformer is quantized in the recommended variants. No speed-up beyond ~1.6× (4 threads) /
  ~2.6× (1 thread) was found on this CPU while keeping FP32-level accuracy.
- FP16/BF16 bring no CPU speed-up at batch size 1.
- Attention is not fused (exported as MatMul+Softmax); static INT8 (QDQ) not attempted.
- `fp32_ortopt` is tied to the onnxruntime version that produced it (regenerate after upgrades).

## Animation-system limitations

- Foot locking is ankle-only two-bone IK; no toe pivoting, ground-height probing or slope
  adaptation; residual sliding is ~1.5× the mocap level on the eval set.
- Join smoothness after velocity matching is still ~1.6× the mocap level.
- Action set is limited by available **licensed clips**: falling, rolling, crawling, climbing,
  fighting, hit reactions, carrying and object interactions have no clips here and are untested.
- Dodge uses a combat quick-step as an approximation; sprint is capped by the fastest sample run
  (~5.5 m/s).
- Target matching uses root velocity only (plus crouch flag); no foot-phase matching.
- Single character per controller; batching several characters into one network call is
  supported by `SomaEngine.generate_batch` but not yet scheduled automatically by the controller.

## Training

- **No training or fine-tuning was performed.** The package has no training code, optimizer
  state, or training data; the only motion data here is evaluation-licensed. See `TRAINING.md`
  for what would be needed.

## Data / licensing

- The sample BVH clips and everything derived from them (eval set, generated BVH previews) are
  under the NVIDIA Sample Data Evaluation License and are **not committed**; they are fetched
  locally by `tools/fetch_upstream.py`.
- Model weights (NVIDIA Open Model Agreement) are downloaded, not committed (size); exact
  checksums are committed in `models/soma/checksums/`.
