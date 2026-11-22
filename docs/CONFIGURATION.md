# Configuration

Everything is driven by YAML. `configs/default.yaml` is the complete, commented schema; every
other file only lists what it changes.

```
configs/
  default.yaml              complete schema with defaults and comments
  presets/                  ultra_quality | balanced | low_vram | extreme_performance | cpu_only
  custom/                   your profiles (example_heavy_character.yaml)
  skeletons/                skeleton profiles (soma; dog/cat/... topology templates), joint limits, hinge axes
```

## Loading, layering, saving

```python
from somart.config import Config
cfg = Config.load("balanced")                         # preset name
cfg = Config.load("custom/example_heavy_character.yaml")  # any path; supports `extends:`
cfg = Config.load("balanced", overrides={"locomotion": {"run_speed": 4.0}})
cfg.save("configs/custom/my_game.yaml")               # writes only the keys that differ from defaults
cfg.save("configs/custom/my_game_full.yaml", full=True)
```

Layer order: `default.yaml` ← `extends` chain ← the file ← `overrides`.

## Runtime changes

```python
cfg.set("locomotion.responsiveness", 0.08)    # validated: unknown keys, wrong types, out-of-range
cfg.set("animation.foot_locking.enabled", False)  #   values and invalid choices raise ConfigError
cfg.on_change(lambda key, value: print(key, value))
```

`CharacterController` listens for changes and applies them on the next tick (inertialization
half-life, smoothing, foot-lock thresholds, all locomotion/generation values). Changing
`inference.*` requires recreating the engine (a new model must be loaded).

## Per-character, per-action, per-skeleton

```yaml
characters:
  heavy_guard:
    locomotion: {run_speed: 2.8, acceleration: 5.0, inertia: 1.6, momentum: 0.35}
actions:
  sprint: {source: locomotion/bs_1d_linear, kind: locomotion, generation: {prediction_horizon: 0.6}}
  roll:   {source: my_clips/roll.bvh, kind: oneshot, key: min_height}   # your own licensed clip
skeletons:
  soma: {foot_joints: [...], leg_chains: [[LeftLeg, LeftShin, LeftFoot], [RightLeg, RightShin, RightFoot]]}
```

```python
hero = cfg.for_character("heavy_guard")   # base merged with characters.heavy_guard
ctrl = CharacterController(engine, lib, cfg, name="heavy_guard")  # picks it up automatically
cfg.for_action("sprint")                  # base merged with the action's nested sections
cfg.skeleton_settings("soma")
```

## Settings reference

| Key | Meaning |
|---|---|
| `inference.backend / model / precision` | onnxruntime or torch; model file; fp32/fp16/int8/int4/bf16 |
| `inference.models_by_device` | `{gpu: ..., cpu: ...}` — hardware detection picks one |
| `inference.providers` | ORT execution providers in priority order (unavailable ones are skipped) |
| `inference.threads / reserve_cores / inter_threads` | thread pool sizing (0 = physical cores − reserve) |
| `inference.memory_arena / memory_pattern / allow_spinning` | ORT memory and CPU behaviour |
| `inference.async / max_batch / memory_limit_mb` | worker thread; batch size; refuse models that do not fit |
| `generation.min_tokens / max_tokens` | allowed segment lengths (6–16 tokens, 1 token = 4 frames) |
| `generation.duration_mode` | `horizon` (length from prediction horizon, responsive) or `model` (network chooses) |
| `generation.prediction_horizon` | seconds ahead the target is placed |
| `generation.replan_interval / replan_lead / input_replan_threshold` | receding-horizon cadence, latency margin, input sensitivity |
| `generation.correct_endpoints / velocity_matching / velocity_matching_tau` | pose accuracy at joins, velocity continuity |
| `locomotion.walk/run/sprint/crouch_speed` | gait speeds (m/s) |
| `locomotion.acceleration / deceleration` | caps (m/s²) |
| `locomotion.responsiveness / inertia / momentum` | velocity spring half-life, its multiplier, carry-over on reversals |
| `locomotion.turn_rate / facing_responsiveness / strafe` | facing behaviour |
| `animation.playback_rate` | animation speed |
| `animation.transition_duration / inertialization` | splice blending |
| `animation.smoothing / intensity / variation / seed` | low-pass, motion amplitude, target randomness |
| `animation.foot_locking.*` | contact thresholds, max correction, blend times |
| `skeleton.*`, `skeletons.<name>.*` | rig selection and per-skeleton IK/contacts |
| `clips.root`, `actions.<name>.*` | clip library and action definitions |

## Presets (measured with the earlier benchmark scripts)

| Preset | CPU model | GPU model | Intent |
|---|---|---|---|
| `ultra_quality` | FP32 (ORT-pre-optimized) | FP32 | reference accuracy, longest horizon |
| `balanced` *(recommended)* | INT8 pose transformer | mixed FP16 | FP32-level motion, ~1.5–2× faster, half the RAM |
| `low_vram` | INT4 pose transformer, no arena, 2 threads | mixed FP16 | smallest footprint keeping FP32-level motion |
| `extreme_performance` | full INT8, no IK, fewer replans | mixed FP16 | crowds / LOD; accuracy traded for throughput |
| `cpu_only` | INT8 pose transformer, never a GPU, no spinning | — | CPU-only deployments sharing cores |

`somart.hardware.recommend_profile()` picks one automatically from the detected hardware.
