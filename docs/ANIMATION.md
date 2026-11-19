# Animation quality and actions

The goal is game-quality feel: immediate response, believable acceleration and weight, no pops,
no foot skating. "AAA feel" is not a single metric, so every claim below is tied to a measurement
in `benchmarks/` (held-out mocap comparisons, and a real-time 60 Hz gameplay script).

## What improves the feel, and how much (measured)

| Technique | Where | Effect (measured) |
|---|---|---|
| **Duration pinned to the prediction horizon**, target placed where gameplay will be when the segment ends | `controller.py` | body reaches 90 % of a velocity change in **0.40–0.48 s** vs 1.45 s with model-chosen duration; tracking error 0.13–0.21 m vs 1.45 m (`REPORT_controller.md`) |
| **Latency-compensated receding-horizon replanning** (every 8–16 frames + immediately on input change) | `controller.py` | 0 late results in the scripted session; inputs are acted on within one generation (~60 ms) |
| **Asynchronous generation**, core reservation | `engine.py`, `hardware.py` | main-thread frame p50 ≈ 2.3 ms, 0–4 frames over the 16.7 ms budget per 1200-frame session (vs. ~60 ms stalls in sync mode — see report) |
| **Velocity-matched joins** | `codec.velocity_matching` | join acceleration spikes −49 % / −40 % (start / end) vs upstream correction; MPJPE −5 % (`REPORT_postprocess.md`) |
| **Foot locking** (model contacts → two-bone IK) | `postprocess.FootLocker` | foot sliding −37 % (12.5 → 7.8 cm/s; mocap 5.2) |
| **Inertialization** of any spliced discontinuity | `postprocess.Inertializer` | hides late results / runtime overrides without cross-fade "ghosting" |
| **Spring-damper locomotion** with accel/decel caps, inertia, momentum, turn-rate limit | `controller._simulate` | natural acceleration/deceleration curves; heavier/lighter characters via config |
| **Target variation band** | `clips.match(variation=…)` | non-repeating choice among near-equal mocap windows |

Not done: physics-based balance or contact dynamics (MotionBricks is kinematic); pose-space
foot-ground penetration correction beyond ankle locking; hand IK for object interaction.

## Supported actions

Status with the data available in this environment (the 38 upstream SOMA sample clips):

| Action | Status | Source |
|---|---|---|
| Idle | ✅ | `locomotion/idle` (5 clips incl. looking around, relaxed legs) |
| Walk, jog, run | ✅ | `locomotion/bs_1d_linear` (0–5.5 m/s) |
| Sprint | ✅ (≤ 5.5 m/s) | fastest run-loop windows; no dedicated sprint capture in the samples |
| Turning, 180° reversal, strafing | ✅ | 2D directional clips + model inbetweening |
| Crouch idle / crouch walk | ✅ | `locomotion/idle/crouch_idle`, `bs_2d_directional_crouch` (≤ 1.6 m/s) |
| Jump (over obstacle) incl. take-off/landing | ✅ one-shot | `jump/*` (0.5 m, 0.75 m, with/without hands) |
| Duck / bend down | ✅ one-shot | `avoidance/*` |
| Dodge / side-step | ⚠️ approximation | `variations/combat_quickstep*` |
| Sit down / stand up | ✅ one-shot | `sitting/*` |
| Styles: injured, zombie, armed walk, skipping, dance | ✅ as clips/targets | `variations/*` |
| Falling, rolling, crawling, climbing, fighting/strikes, hit reactions, carrying, object interaction | ❌ **no clips available here** | supported by the architecture once you add licensed clips (below); model quality on them is **untested** |

The SOMA model was trained on ~313 k clips of the Bones RigPlay 1 corpus, so it has seen far more
motion types than the samples here; whether it inbetweens a given new action well can only be
established by testing with real target poses for that action.

## Adding an action

```python
lib.add_clip("mine/roll_forward", load_bvh("roll_forward.bvh"), tags={"category": "roll"})
cfg.set("actions.roll", {"source": "mine/roll_forward", "kind": "oneshot", "key": "min_height"})
ctrl.trigger("roll")
```

or in YAML (`actions:` section). Clips must be on the SOMA skeleton (same joint names), at any
frame rate (resampled to 30 Hz). `key` chooses the target pose: `min_height`, `max_height`,
`middle`, `end`, or a fraction 0–1 of the clip. One-shot actions keep the model-chosen duration
and suppress replanning until the segment has nearly played.

## Transitions between actions

Every transition — gait change, turn, action entry/exit — is a MotionBricks inbetween from the
4 poses on screen to target poses, followed by velocity matching, inertialization (if late) and
foot locking. `animation.transition_duration` sets the inertialization length,
`generation.prediction_horizon` the transition length for locomotion.

## Tuning cheat-sheet

| Want | Change |
|---|---|
| Snappier | `locomotion.responsiveness` ↓ (0.08), `generation.prediction_horizon` ↓ (0.65), `replan_interval` ↓ |
| Heavier, more momentum | `locomotion.inertia` ↑ (1.4–1.8), `momentum` ↑ (0.3), `acceleration`/`deceleration` ↓ |
| Smoother, less reactive | `prediction_horizon` ↑ (1.0), `animation.smoothing` 0.1–0.2 |
| More variety | `animation.variation` ↑ (0.3–0.5) |
| Less foot sliding | keep `foot_locking.enabled`, raise `max_correction` (0.15), lower `contact_threshold` |
| Bigger/smaller gestures | `animation.intensity` 0.8–1.2 |
| Faster/slower playback | `animation.playback_rate` |
