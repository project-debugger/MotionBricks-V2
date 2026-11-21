# Remaining limitations and blockers

## Hardware / environment blockers (this machine)

- **No GPU**: TensorRT, CUDA/DirectML/CoreML execution and **VRAM measurement were not possible**.
  Presets list GPU providers first and fall back to CPU; GPU behaviour is unverified here. The
  only GPU numbers are NVIDIA's own (RTX 4090 + TensorRT: 2.50 ms FP32, 1.28 ms FP16, N=1).
- **Noisy shared VM**: run-to-run latency varies by ±20–30 %. Relative comparisons use the
  interleaved benchmark; absolute numbers will differ on other machines.
- **4 vCPUs**: the game loop and generation share cores; generation latency in the controller
  benchmark (49–93 ms p50 depending on preset) is higher than in isolation (~44 ms for INT8-pose).

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
  transformer is quantized in the recommended variants. On this CPU the speed-up at FP32-level
  accuracy is ~1.4–1.5× (4 threads) / ~2.0× (1 thread); full INT8 reaches ~1.6× / ~2.5× with worse motion.
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
- Fingers, toes beyond their learned 1–2 axes, and the jaw are not predicted by the model: hands
  use a procedural layer (`FingerAnimator`), toes and jaw are clamped to mocap ranges. No hand
  contact, gripping or facial animation.
- Joint limits are fitted to the 38 sample clips (plus 3° margin); poses outside that range
  (deep squats, splits, acrobatics) would be clamped.
- Web playground (`site/`): phone performance is unmeasured (no device here). Measured on this
  machine with 0.3-0.6 s of added delay per generation: every start, stop, turn, gait change and
  jump/dodge/duck shows its first visible change within 0.03-0.17 s and follows the command within
  0.13-0.58 s (`site/build/test/sim.html`); no freezes or lurches at up to 1.2 s of added delay.
- The instant response comes from a bank of 294 transitions generated ahead of time by the same
  SOMA network (discrete directions every 45 deg, three speeds, four stride phases); the live model
  refines the exact direction and speed from inside the bank segment. An action tapped while
  another action is still playing waits for it. Sitting down and standing up are live only.
- The body layer (muscle bulges, breathing, soft tissue, mouth opening when out of breath) is
  procedural and tuned by eye; the model and the mocap contain no muscle, breath or face data.
  The mesh has no eyelids or separately skinned eyes, so there is no blinking or eye motion.
- Animals in the web playground are driven by a small network trained here on 41 minutes of dog
  motion capture (CC BY-NC 4.0, non-commercial). Only the dog is real data; the other species reuse
  it re-proportioned and time-scaled, so a horse, deer or elephant moves like a large dog (no
  hoofed gaits, no columnar elephant legs, no rabbit hopping). The data has lying down, rearing up
  and about ten forward leaps, which are learned; it has almost no sitting, so Sit lies down, and
  the elephant rears instead of jumping. Bodies are built from primitive shapes, with no fur or
  skin deformation. The SOMA model itself was not retrained: it was released without training code.
- Fighting moves: the key poses are hand-built by IK, the motion into and out of them is SOMA; the
  model was not trained on fighting, so strikes are less crisp than captured martial arts.
- Target windows come from 4,737 mocap frames (with mirrored copies); straight running near
  3.4 m/s and sprinting are still thinly covered (about 20 and 14 frames before mirroring).

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
- Exception for the web playground: `site/models/` (INT8 web copy of the weights) and
  `site/data/windows.*` (motion windows derived from the evaluation-licensed sample clips) are
  committed so the site can be served. The repository is private; enabling GitHub Pages would
  make both public (see `site/README.md`).
