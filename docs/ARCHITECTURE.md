# Technical architecture

```
                 gameplay input (direction, gait, actions)
                                 |
                 +---------------v----------------+        configs/*.yaml  (live-editable,
                 | CharacterController (60+ Hz)    |<------- validated, per character /
                 |  spring-damper trajectory       |         action / skeleton)
                 |  receding-horizon planner       |
                 |  target selection (ClipLibrary) |
                 +---+-------------------------^---+
         request:    |                         | GeneratedMotion (30 Hz local poses, contacts)
  4 context poses +  |   worker thread (async) |
  1-4 target poses   v                         |
                 +---------------------------------+
                 | SomaEngine                      |
                 |  FK -> align -> encode (NumPy)  |
                 |  backend: ONNX Runtime / torch  |   models/soma/optimized/onnx/*.onnx
                 |  decode -> endpoint correction  |   (FP32 / FP16 / INT8 / INT4)
                 |  -> velocity matching           |
                 +---------------------------------+
                                 |
       per frame: sample 30 Hz buffer -> inertialization -> foot locking (IK) -> smoothing -> render
```

## The SOMA model (as released by NVIDIA)

Three networks combined into one ONNX graph by the upstream exporter:

1. **Root model** (40.4 M params) — predicts the duration (6–16 tokens of 4 frames) and the root
   trajectory from the 4 context + up-to-4 target poses.
2. **Pose model** (136.5 M) — 16-layer masked transformer predicting 5 × 16-way pose-token heads
   per token in one pass (argmax, `num_inference_step = 1`).
3. **Pose VQ-VAE decoder** (23.4 M incl. encoder) — decodes tokens + root conditioning into
   366-dim per-frame features (root, joint positions, 6D rotations, velocities, 4 foot contacts).

Interface (`runtime/somart/contract.py`): inputs `[N,8,5] [N,8,4] [N,8,267]` + 3 masks + allowed
token counts; outputs `[N,64,366]` + token count. 30 Hz, Y-up, metres.

## Runtime package `somart` (`runtime/somart/`)

| Module | Responsibility |
|---|---|
| `math3d.py` | Vectorised quaternion/transform maths (XYZW, matches upstream Warp), depth-batched FK |
| `skeleton.py` | `Skeleton`, `Clip`, BVH loader (NumPy port of upstream, drops the static `Root` wrapper) |
| `contract.py` | Tensor contract and timing constants |
| `codec.py` | NumPy port of the upstream Warp kernels: guide alignment, input encoding, masks, output decoding, endpoint correction; plus **velocity matching** and contact-aware correction |
| `backends.py` | `OrtBackend` (any ORT execution provider, thread/arena/spinning/graph-opt options), `TorchBackend` (eager graph from SafeTensors, BF16 autocast) |
| `engine.py` | `SomaEngine`: single, batched and asynchronous generation with per-stage timings |
| `clips.py` | `ClipLibrary`: clip loading/tagging, canonical 4-frame windows, velocity-based target matching, placement |
| `controller.py` | `CharacterController`: trajectory simulation, latency-compensated receding-horizon planning, splicing, actions |
| `postprocess.py` | Inertialization, two-bone IK foot locking, smoothing, intensity scaling |
| `config.py` | Layered YAML config, validation, runtime changes, save/load, per-character/action/skeleton |
| `hardware.py` | Hardware detection (CPU SIMD, ORT providers, CUDA), profile recommendation, device model choice |
| `skeletons.py` | Multi-skeleton profiles, generalised feature layout, model-family registry, retargeting |
| `metrics.py` | MPJPE, foot sliding, jerk, join acceleration, contacts heuristic |
| `bvh_writer.py` | BVH export for inspection in DCC tools |

## Generation pipeline details

1. **FK** of the 8 guide poses on the runtime skeleton (77 joints for the SOMA BVH rig; joints are
   batched by tree depth: ~12 vectorised steps instead of 77).
2. **Alignment** (upstream): the last context pose is moved to the origin and turned to face +X;
   heading is derived from the leg-root joints.
3. **Encoding** (upstream): root position/heading, heading rate and ground velocity (from the next
   pose), 29 joint positions relative to the root ground point and 30 6D rotations.
4. **Masks** (upstream): the last context slot and last given target slot carry no root velocity.
5. **Network** on the selected backend.
6. **Decode** (upstream): drop 3 boundary frames at each end, Gram–Schmidt 6D → rotation, local
   rotations from parent-relative global rotations, undo alignment.
7. **Endpoint correction** (upstream): blend so frame 0 equals the last context pose and the last
   frame equals the first target pose.
8. **Velocity matching** (somart, default on in presets): additive offset `dv · t · e^(−t/τ)` at both
   ends so velocities are continuous across the joins.

## Controller

- **Trajectory**: critically-damped velocity spring (`responsiveness × inertia` half-life),
  acceleration/deceleration caps, momentum on reversals, turn-rate-limited facing spring.
- **Planning**: every `replan_interval` frames, or immediately when the goal velocity changes by
  more than `input_replan_threshold`, a request is built from the 4 buffered frames at the
  **switch frame** `k = now + measured latency + replan_lead`, so the result is ready before it is
  needed (late results are hidden by inertialization and counted).
- **Duration pinning** (`duration_mode: horizon`): the segment length is set from the prediction
  horizon and the target is placed where the trajectory will be when the segment ends — this is
  what brought body response from 1.38 s to 0.41 s and tracking error from 1.31 m to 0.13 m (earlier benchmark report `REPORT_controller.md`).
- **Targets**: `ClipLibrary.match` picks the window whose local root velocity best matches the
  predicted velocity (crouch-aware, optional random variation band), placed at the predicted
  position/facing. One-shot actions use a configured key pose of their clip and commit the segment.
- **Per frame**: sample the 30 Hz buffer at the display rate (lerp/slerp), inertialization, foot
  locking (one FK per frame, analytic IK per leg), optional smoothing.

## Threading and determinism

`SomaEngine.submit` uses one worker thread; ONNX Runtime releases the GIL during inference. With
`threads: 0` the runtime uses physical cores minus `reserve_cores` (default 1) for the network,
leaving a core for the game loop. Use `synchronous=True` (or `inference.async: false`) for
deterministic offline generation.
