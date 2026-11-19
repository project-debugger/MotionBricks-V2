# SOMA Motion Showcase

Thirteen 15–20 second videos (MP4, H.264 High, 1920×1080, 30 fps) of NVIDIA MotionBricks SOMA
running in this repository's optimized runtime. Open [`index.html`](index.html) for the gallery,
or download [`MotionBricks_SOMA_showcase.zip`](MotionBricks_SOMA_showcase.zip) (all videos +
gallery, works offline).

## Is this real model output?

Yes, every video. For each clip a scripted gameplay input (move direction, gait, actions) drives
`runtime/somart`'s `CharacterController`, which calls the SOMA network (ONNX Runtime, the model
file named in the clip) for every transition. Nothing is keyframed, interpolated by hand or played
back from mocap. [`videos/MANIFEST.json`](videos/MANIFEST.json) lists for every panel the model
file, its SHA-256 prefix, the number of SOMA generations and the settings that differ from the
balanced preset.

What is *not* model output:

- The **body mesh** is NVIDIA's SOMA character from the MotionBricks software, skinned to the
  generated joints (linear blend skinning, 4 weights per vertex).
- **Hurdles and the chair** are props placed after generation, at the generated jump apex and
  under the generated seated pose. The model never sees them.
- **Camera, floor, lights, labels** are rendering. Green rings under the feet show the model's
  own foot-contact prediction.

## Videos

| File | What it shows |
|---|---|
| [01_walk_run_sprint.mp4](videos/01_walk_run_sprint.mp4) | Idle → walk → run → 5.5 m/s sprint → hard stop |
| [02_turns_and_reversals.mp4](videos/02_turns_and_reversals.mp4) | 90° turns, a 180° reversal and a zig-zag at running speed |
| [03_crouch.mp4](videos/03_crouch.mp4) | Walk → crouch walk → crouched turn → crouch idle → stand and walk |
| [04_jump.mp4](videos/04_jump.mp4) | Two running jumps over hurdle props |
| [05_sit_and_stand.mp4](videos/05_sit_and_stand.mp4) | Walk → sit → seated → stand up → walk away |
| [06_transitions_medley.mp4](videos/06_transitions_medley.mp4) | 20 s of chained moves: walk, run, turn, sprint, stop, crouch, run, jump, walk |
| [07_animation_speed.mp4](videos/07_animation_speed.mp4) | Same turns at 0.6×, 1.0× and 1.5× `animation.playback_rate` |
| [cmp_fp32_vs_int8.mp4](videos/cmp_fp32_vs_int8.mp4) | Original FP32 model vs the INT8 model of the balanced preset |
| [cmp_responsiveness.mp4](videos/cmp_responsiveness.mp4) | Default vs high responsiveness (orange arrow = player input). Measured: 0.31 s vs 0.28 s to follow an input change; the high setting mainly gives harsher poses |
| [cmp_smoothness.mp4](videos/cmp_smoothness.mp4) | Default vs maximum-smoothness profile. Measured: 0.32 s vs 1.03 s to follow an input change |
| [cmp_model_variants_quality.mp4](videos/cmp_model_variants_quality.mp4) | FP32, mixed FP16, INT8-pose, INT4-pose |
| [cmp_model_variants_aggressive.mp4](videos/cmp_model_variants_aggressive.mp4) | FP32 and INT8-pose vs full INT8 and full INT4 |
| [cmp_foot_locking.mp4](videos/cmp_foot_locking.mp4) | Foot locking off vs on during a walk with turns and stops; a dot is stamped every frame the model reports a planted foot (teal left, orange right), so sliding shows as streaks |

Settings in the comparisons (everything else is the balanced preset, and both panels get the
same input and random seed):

- **High responsiveness**: `locomotion.responsiveness 0.05`, `facing_responsiveness 0.05`,
  `acceleration 22`, `deceleration 28`, `inertia 0.8`, `generation.prediction_horizon 0.6`,
  `replan_interval 5`.
- **Maximum smoothness**: `responsiveness 0.30`, `facing_responsiveness 0.30`, `acceleration 5`,
  `deceleration 6`, `inertia 1.4`, `prediction_horizon 1.2`, `replan_interval 14`,
  `animation.smoothing 0.6`, `transition_duration 0.4`.

Generation runs synchronously at 30 Hz so every run is frame-exact and repeatable; the planning
logic is the same as in real-time use.

## Rebuild

Needs the models (`tools/download_soma.py`, `tools/export_onnx.sh`, `tools/optimize_models.py`),
the upstream assets (`python3 tools/fetch_upstream.py --groups rig,motions` plus the SOMA mesh)
and `usd-core` in `.venv`.

```bash
bash showcase/build/make_all.sh      # motions -> render (3 parallel Chromium) -> gallery + zip
```

Rendering takes roughly 40 minutes on 4 CPU cores with software WebGL; a GPU machine is much
faster.

## Licences

Model weights: NVIDIA Open Model Agreement. SOMA mesh and MotionBricks software: see the upstream
repository's licence terms. The sample motion clips used for target poses are under the NVIDIA
Sample Data Evaluation License (evaluation and testing) and are not included here.
