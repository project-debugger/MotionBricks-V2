# SOMA Playground (web)

A playable third-person demo in which every movement of the character is generated live by
NVIDIA's MotionBricks SOMA network, running in the browser with onnxruntime-web (WebAssembly).
It works with touch controls on phones and with keyboard and mouse on desktop, and the controller
can be tuned in the settings panel.

Nothing is pre-baked: the page sends the last poses and a target pose window to the network,
then decodes and post-processes the result in JavaScript. The pipeline is a port of the Python
runtime in `runtime/somart/`, checked against it to float precision.

> **Measurements below** were taken while the motion targets still came from NVIDIA's MotionBricks sample clips
> (evaluation licence). The targets, clips, transition bank and joint limits have since been rebuilt from freely
> licensed motion capture only (see "Motion data"). The rebuilt page was checked for every move in headless
> Chromium (walk 1.4, run 3.4, sprint 5.5 m/s, crouch, jump, duck, dodge, fighting, sit and stand up) and with
> `build/test/sim.html` (no failures, NaN frames or spikes); the per-metric tables were not re-measured.

## Controls

| Phone | Desktop | Action |
|---|---|---|
| Left joystick (push far to run) | `WASD` / arrow keys | Move (camera-relative) |
| Hold Sprint | `Shift` | Sprint |
| | `Z` | Walk / run toggle |
| Crouch button | `C` | Crouch / stand |
| Jump button | `Space` | Jump |
| Dodge button | `F` | Dodge (quick side-step) |
| Duck button | `X` | Duck |
| Sit button | `E` | Sit on a chair / stand up |
| Punch, Kick, Block | `J`, `K`, `L` | Fighting |
| Drag the scene | Drag, mouse wheel | Orbit the camera (desktop: wheel zooms) |

## Model and settings

| Model | Network | Download (cached after the first visit) |
|---|---|---|
| Balanced | INT8 pose transformer; matrix weights stored as ggml Q5_0 in a split GGUF (12 shards), computed in INT8 | 150 MB |

Built from `models/soma/optimized/onnx/soma_int8_pose.onnx` by `tools/build_web_models.py` and packed by
`tools/gguf/build_gguf.py`: the INT8 weights of the matrix multiplies are re-coded as Q5_0 blocks (5.5 bits per
weight) and decoded back onto the INT8 grid when the page loads (`js/gguf.js`), so the runtime graph, speed and
memory are those of the INT8 model. Held-out eval set (187 windows): joint error 7.07 cm vs 7.10 cm for the INT8
file (Q4_0 would be 7.27 cm).

Earlier versions also offered **Light (INT4)**. It was removed: in the browser its 4-bit kernels
are about 3x slower than INT8 on one WebAssembly thread (0.75-1.2 s vs 0.23-0.31 s per generation
in headless Chromium on this machine), which made slow phones fall behind. An all-INT8 model was
also tried: same download, no measurable speed gain in the browser, higher joint error (8.96 cm).

The settings panel (gear icon) has presets (Responsive, Balanced, Smooth), sliders for speeds,
responsiveness, planning horizon, animation speed, smoothing, motion variety, the finger layer
and the (off by default) body-rotation assist, and toggles for foot locking, anatomical joint
limits, procedural fingers, velocity-matched joins, head-leads-turns, instant response and the
body layer. Settings are kept in the browser's local storage.

## What runs per generation

1. The controller predicts where the character will be at the end of the next segment, starting
   from the root position and velocity of the frame the segment starts from. It then picks the
   target pose from continuous mocap (every frame of the walk/jog/run/crouch/idle clips at 30 fps,
   plus left/right mirrored copies): matching velocity, turn rate, posture, and the **stride phase**
   the character will be in when the segment ends (from the left/right foot separation), so the
   network continues the current stride instead of being pulled to a pose from another part of the
   cycle. An action uses its key pose (jump, sit, dodge, ...).
2. `codec.js` encodes 4 context + 4 target frames, the worker runs the ONNX model, then decodes
   frames and contacts, applies endpoint correction, velocity matching and the learned
   anatomical joint limits (knees and elbows as pure hinges, toe and jaw ranges).
3. Every displayed frame then gets foot locking (two-bone IK on model contacts),
   inertialization of any splice (on the base motion, before the procedural layers), the
   procedural finger layer, the head/chest look-ahead and the body layer (below).

## Instant response (0 s)

A live generation takes 0.25 s here and longer on a phone, so the first frames of a reaction can
never come from a request made at that moment. The playground therefore ships a **transition bank**:
294 starts (8 directions x walk/run/sprint), stops (from 4 stride phases at each speed), turns
(+-45/90/135/180 deg), gait changes, crouch moves and jump/dodge/duck from idle, walk and run,
21,450 frames in total. Every bank frame is output of the same SOMA network and pipeline, generated
ahead of time by `build/tools/gen_bank.html` from mocap contexts, as chains of short segments
(see "Starts and turns").

When the input changes, the controller scores every bank frame against the character's current
feet, root velocity and the path the new command asks for (motion matching), restricted to the
right intent (stop entries for a stop, entries moving at about the current speed), and jumps there
on the next frame with inertialization, unless the current motion already follows the new path.
Live generation continues from inside the bank segment; results planned before the jump are
discarded. Turns are real SOMA turning motion (lean, steps); the earlier whole-body rotation assist
is off by default.

Measured with `build/test/sim.html` (real model, scripted 21 s input, one WASM thread, 0.3-0.6 s of
random delay added to every generation to emulate a phone; "onset" is the first visible change in
smoothed root velocity, "reached" is when the motion follows the command):

| Input | Before: motion follows the command | Now: first visible change / motion follows |
|---|---|---|
| Start running | 1.3 s | 0.09 s / 0.45 s |
| 90 deg turn | 0.2 s (rotation assist) | 0.07 s / 0.39 s |
| 180 deg reversal | 1.5 s | 0.17 s / 0.49 s |
| Stop | 1.7 s | 0.04 s / 0.34 s |
| Run to walk diagonally | 1.8 s | 0.07 s / 0.23 s |
| Sprint | 0.9 s | 0.07 s / 0.58 s |
| Stop from sprint | 1.4 s | 0.03 s / 0.47 s |

No freezes or lurches; top speed 5.7-6.0 m/s (sprint 5.5); joint angular speed p99 14 rad/s and the
displayed per-joint maxima within ~1.5x of the mocap's. Arm swing after a 90 deg turn: hand
travel 0.48 / 0.59 m and elbow range 48-54 deg (before: 0.28 / 0.21 m, 19-20 deg; mocap running:
0.46-0.52 m, 52 deg). Walking reaches its 1.4 m/s target (before 0.84 m/s) with even arm swing.

## Starts and turns: upright, no flips

Recorded frame by frame (`SOMA_DEBUG.step(dt)` advances the game exactly one frame), starting to jog used
to play a sprinter's block start: the torso pitched up to 50 deg forward with the arms hanging, for about
half a second; some 180 deg reversals flipped the body through one frame. Both came from the transition
bank, and both are fixed:

- **Cause of the lunge**: each bank start or turn was one 16-token (1.9 s) request. Given that much time,
  the SOMA network plans a block start, even with every post pass off (torso pitch 38-54 deg idle to run,
  60-97 deg idle to sprint). 6-token steps reach the same speed (3.4 m/s after about 1 s) with 7-9 deg of
  lean, as people start a jog. Bank entries are now chains of 6-token segments (12 frames kept from each),
  the way live play requests motion.
- **Flips**: the network sometimes jumps a 180 deg reversal through one or two frames (70-170 deg).
  `maxBodyStep` flags any body joint turning more than 60 deg in a frame; the generator then turns
  through one side first (90 deg, then 180), and the live controller discards such a result and replans
  through 90 deg (at most twice in a row, so playback never waits).
- **Human acceleration**: planned motion speeds up and brakes no faster than people in the mocap
  (walk 2.5, jog 4.5, sprint 6 m/s^2; braking 3.5-7), measured on the 100STYLE and NVIDIA clips.
  Turning keeps the preset's rate.
- **Bank matching**: an entry of another gait family costs 2 (was 0.25), so a jog never starts from a
  walking turn.

| Peak torso pitch (hips to neck) | Before | After |
|---|---|---|
| bank, idle to run (all directions) | 38-54 deg | 9-14 deg |
| bank, idle to sprint | 60-97 deg | 6-11 deg |
| bank, walk to run / run turns | 20-66 / up to 58 deg | 11-14 / up to 17 deg |
| live jog start straight / 90 deg / 180 deg | 49 / 17-29 / 17-36 deg | 9 / 9-12 / 13-15 deg |
| live, 450 ms extra latency (phone) | - | 8-14 deg |
| real people starting a jog (100STYLE) | 24 deg (99th pct), 32 max | |

Worst shoulder turn in one frame across the bank: 145 -> 25 deg. Response (sim, desktop): start to
run 0.23 s (was 0.38), 90 deg turn 0.08 s, sprint 0.11 s; a 180 deg reversal takes 0.6 s (was 0.5)
and a stop from sprint 0.5-0.64 s (was 0.43), because the body now brakes like a person.

### Turns and reversals at a run

Planned motion now also turns within human limits (sideways acceleration: walk 3, run 6, sprint 7 m/s^2, from
the mocap), so a sharp turn slows down and curves and a reversal brakes (about 0.5 s from a run), plants, pivots
and pushes off. Before, the plan braked at 15-25 m/s^2 and turned at 16 m/s^2, which the network could only
follow with a deep lunge (29 deg) or a 45 deg bank. Measured (frame by frame, run then reverse): peak lean
29 -> 20 deg, no speed surge; 90 deg turn: lean 16-17 deg while slowing to about 1.7 m/s. A reversal now takes
0.8 s to follow (was 0.6); starts, 90 deg turns and stops are unchanged. The bank's reversal from a run no
longer borrows a sprint entry (same-family cost 0.25 -> 0.8).

A result that arrives after the playhead has already passed its whole segment (stalled worker, tab in the
background) is now discarded; splicing it collapsed the buffer and crashed the controller.

## Shooter-style movement (default preset)

The default preset, **Shooter**, aims at the feel of third-person action games (movement only):

- quick replans and short blends (the earlier Responsive values) plus a steering assist that turns the
  motion already playing toward the joystick at up to 220 deg/s;
- **snappy transitions**: the first 0.4 s after an instant start, stop or turn play up to 35 % faster,
  easing back to normal speed (slider "Snappy starts, stops and turns");
- **leaning into turns**: the body banks toward the inside of a curve about a pivot on the ground, by
  atan(speed x turn rate / g), smoothed (0.15 s) and capped at 16 deg (slider "Lean into turns", default
  0.8; `build/test/lean_test.mjs` checks the direction on circular paths).

Measured with `build/test/sim.html` (same script, desktop): time to reach the new direction or speed

| Event | Balanced (before) | Shooter |
|---|---|---|
| start to run | 0.45 s | 0.38 s |
| 90 deg turn while running | 0.30 s | 0.10 s |
| 180 deg reversal | 0.55 s | 0.52 s |
| stop | 0.47-0.55 s | 0.43 s |

Joint-speed p99 is unchanged (13.8 vs 14.2 rad/s); with 300 ms extra generation latency (slow phone) there are
still no freezes, lurches or failures. The mocap has no real sidestep or backpedal clips, so the character
always faces where it runs (no strafing).

## Motion data (freely licensed motion capture)

The network generates every frame; motion capture only decides where each segment should end up (targets), and
seeds the instant-transition bank. All of it is free to redistribute:

| Source | Licence | Used for |
|---|---|---|
| **100STYLE** (Mason, Starke, Komura 2022) | CC BY 4.0 | walking forward/backward/sideways, slow jogs, idle, starts and stops; the six walk styles |
| **CMU Motion Capture Database** | free for all uses | running 2-4.5 m/s with turns and sudden stops (subjects 9, 16, 35), crouch walking and sneaking (136, 139), jump (16_01), sit and stand up (143_18), duck (77_09), side-step dodge (127_13) |

- `tools/mocap/style100.py fetch` range-downloads just the files used from the 1.5 GB zip; `tools/mocap/cmu.py fetch`
  downloads the listed ASF/AMC trials (14 MB). `retarget` maps them onto the SOMA rig: the bind pose (an A-pose) is
  swung onto each source's rest pose bone by bone, joint world rotations are copied, feet stay flat, the pelvis follows
  the performer's hip sockets scaled by leg length, each clip is lowered so the standing foot is at the rig's stance
  height, and the arms are re-solved with two-bone IK so the elbows bend about the network's own hinge axis
  (`tools/fit_hinges.py`, measured on its raw output; the source rigs' elbow axes are 16-30 deg off it). Bone
  directions agree within 2-3 deg (`tests/test_style100.py`, `tests/test_cmu.py`).
- `tools/export_web_data.py` builds the clips (58, 16k frames, 4 MB), 782 target windows, the 8 action key poses
  and the joint limits; `build/tools/gen_bank.cjs` regenerates the transition bank (166 entries, all SOMA output).
- **Gait-aware matching** (`js/library.js`): jog and run clips are tagged as running and crouch clips as crouching;
  other candidates are labelled walk or jog from their stride period. Walking never targets a jog (100STYLE's walks
  top out near 1 m/s, its slow jogs reach 1.7 m/s), and crouching is a tag, not a pelvis height (a crouch walk's pelvis
  is ~0.85 m because the torso leans).
- **Walk style** (Settings): Natural, Proud, Strutting, Rushed, Sad, Elderly or Hands behind back. Each style file
  (1.3-2 MB) is downloaded only when chosen; the character walks at the style's own pace (0.35 m/s for Elderly up to
  1.95 m/s for Rushed). Running and sprinting stay on the default library.
- **Joint limits**: knees, elbows and toes are hinges about the network's axes, with ranges covering the clips, the
  network's output and normal clinical range of motion; joints the retargeters do not drive stay at the rig's
  reference; the relaxed hand pose is authored (the captures have no finger motion).
- Gaps: sprinting (5.5 m/s) is faster than any capture (4.5 m/s), so the network extrapolates; the crouch is one
  performer's moderate crouch, with no deep crouch walk.

## Upper body and hands

- **Calm torso** (`TorsoCalm` in `js/controller.js`, slider "Calm torso twist when running", default 0.4):
  the chest's turn relative to the hips is split into a slow part (turning, looking around), which is kept,
  and the per-stride side-to-side twist, which is reduced and spread over the three spine joints; a 30 ms
  filter smooths the spine. Measured while jogging (same run, before vs after the pass): twist standard
  deviation 11.4 deg -> 7.1 deg (real jogging capture: 10-22 deg; 0.7 gives 5.6 deg).
- **Jogging hands**: fingers close to a loose fist with speed (running curl 55 deg by default), graded
  from index to pinky, knuckles and thumb included, with follow-through from the arm swing.

## Body layer

Procedural, in the vertex shader (the model and the mocap contain no muscle or breathing data):

- **Muscles**: 24 regions (biceps, triceps, forearm flexors, deltoids, pectorals, lats, trapezius,
  abdominals, glutes, quadriceps, hamstrings, calves, tibialis) found on the bind-pose mesh from the
  skin weights and bone geometry; elbow and knee flexion sides come from the learned hinge axes.
  Each bulges up to 0.4-1.2 cm along the skin normal and tilts the normals along its slope so the
  shape catches the light. Activation follows joint angles, foot contact and effort, with ~60 ms
  contraction and ~160 ms relaxation.
- **Breathing**: chest and belly expansion plus a small thoracic lift; rate rises from ~13 to ~46
  breaths per minute with exertion (which builds over seconds and recovers slowly), the abdominals
  tighten on the exhale, and the mouth opens slightly when breathing hard.
- **Soft tissue**: belly, glutes, chest, thighs, calves and upper arms follow a damped spring
  driven by their bone's acceleration (footfalls, landings), up to 5 mm.
- **Skin**: physically based material with a soft sheen at grazing angles.

## Fighting

Punch (J / Punch button: jab, cross, hook, uppercut in turn), Kick (K: front kick, roundhouse) and
Block (L). Each move is a hand-built key pose (made by IK on the mocap idle pose: guard, extended
arm, chambered and extended leg, forearms up) that the SOMA network moves into from the current
motion and back out of to the stance; from standing (and jab / cross / front kick from walking)
they are in the instant-transition bank, otherwise they are generated live.

## Slow devices (phones)

- **No catch-up lurches**: targets are predicted from the animation's own momentum, never from a
  free-running gameplay position, so the model is never asked to cover an impossible distance.
- **No freezes**: if a result is late, the buffer is extended by looping the last gait cycle
  (re-rooted, seam blended) until it arrives; the late result is then inertialized in.
- **Free worker**: while the input is steady, segments are long and replanned only when the
  buffer runs low.
- The joystick switches walk/run with hysteresis, so a wobbling thumb does not flip the gait.
- **Feet**: soft foot locking. A planted foot stays where it landed but the hold point follows the
  animated foot within 2.5 cm, planting needs a slow foot as well as the model's contact, and a
  released foot carries its offset while the lock fades (the earlier version pulled the foot back
  toward its old spot during the fade: up to 18 cm in this test, the "legs stuck, then pop back"
  glitch). Blends after an instant transition are softer (0.14 s half-life).

## Phones and old PCs

- **Adaptive quality** (Settings, on by default): when the frame rate stays below 40 fps the resolution,
  then the shadow quality, then the shadows step down (4 levels, "lite n" in the fps chip); above
  57 fps they step back up, one change per 4-8 s. On this machine's software renderer (a stand-in
  for a weak GPU): 12 -> 23-27 fps.
- The controller costs 1.2-1.4 ms per frame on the main thread; the SOMA network runs in a
  worker (0.19 s per generation with 4 threads here, about 4 generations per second while moving,
  1.5 when idle). Slow devices get fewer, longer generations automatically (one request in flight).

## Performance (measured here, not on phones)

| Where | Network time per generation | Plan-to-result latency |
|---|---|---|
| Headless Chromium, 1 WASM thread | 0.23-0.31 s | ~0.3 s |
| Headless Chromium, 4 WASM threads, CPU slowed 4x, phone layout | 0.2-0.43 s | 0.41-0.56 s |

No phone was available to test. A recent phone is likely 2-3x slower per thread than this machine,
i.e. about 0.5-0.8 s per generation on one thread (claude.ai copy) and less with threads (GitHub
Pages). The first visit downloads 150 MB.

## Run locally

```bash
node site/build/serve.mjs 8123 --isolate   # COOP/COEP headers -> multi-threaded WASM; open http://localhost:8123/
```

Rebuild the generated pieces when their sources change:

```bash
python3 tools/build_web_models.py && python3 tools/gguf/build_gguf.py   # site/models/ (needs models/soma/optimized/onnx/)
python3 -I tools/mocap/style100.py fetch && python3 -I tools/mocap/style100.py retarget
python3 -I tools/mocap/cmu.py fetch && python3 -I tools/mocap/cmu.py retarget
python3 tools/export_web_data.py                                        # site/data/
node site/build/tools/gen_bank.cjs 8123                                 # site/data/bank.* (dev server running)
python3 site/build/make_index.py                                        # site/index.html from index.src.html
```

`site/build/package.json` pins the vendored libraries (`npm install` there, then copy into `vendor/`).

## Deploy to GitHub Pages

`.github/workflows/pages.yml` publishes `site/` (without `build/`) on every push to `main` or the development
branch that touches `site/`, once Pages is enabled. It checks the model shards against `models/models.json` first.

1. Settings, Pages, Build and deployment, Source: **GitHub Actions**.
2. Actions, **Deploy SOMA Playground**, **Run workflow** (or push a change under `site/`).
3. The site appears at `https://<owner>.github.io/<repo>/`.

Notes:

- GitHub Pages for a private repository needs a paid plan; the published site is public.
- The `github-pages` environment only accepts deployments from allowed branches (by default the default branch).
  Deploy from `main`, or add the branch under Settings, Environments, github-pages.
- GitHub Pages has a soft limit of 100 GB of traffic per month, roughly 600 first visits (150 MB each).
- Pages cannot set COOP/COEP headers, so `coi-serviceworker.js` adds them through a service
  worker (one automatic reload on the first visit) to enable multi-threaded WASM. Without it
  the page still works on one thread.

## Files

```
index.src.html          page source (HUD, controls, panel); build/make_index.py -> index.html
js/main.js              scene, input, settings panel, frame loop, adaptive quality
js/controller.js        planning, splicing, instant transitions, gait-cycle extension,
                        inertialization, foot locking, fingers, head look-ahead
js/library.js           continuous mocap clips with stride phase and gait, walk styles (target selection)
js/bank.js              transition bank + motion-matching features
js/body.js              muscles, breathing, soft tissue (vertex shader)
js/fight.js             fighting key poses (IK on the idle pose)
js/settings.js          presets and defaults
js/codec.js             SOMA encode/decode, endpoint correction, velocity matching, joint limits
js/gguf.js              GGUF reader; rebuilds the ONNX file from the split model
js/math3d.js            quaternion and pose helpers
js/worker.js            model download and cache, onnxruntime-web inference
data/                   skeleton (SOMA rig), target windows + action poses, clips (100STYLE + CMU),
                        styles/ (optional walk styles, loaded on demand), transition bank,
                        joint limits, SOMA body mesh
models/                 models.json + the split GGUF model (12 shards, 150 MB)
vendor/                 onnxruntime-web 1.31.1, three.js 0.170.0 (MIT, unmodified)
coi-serviceworker.js    COOP/COEP shim for static hosts (MIT)
build/                  dev server, index builder, bank generator (tools/gen_bank.*), controller
                        simulation (test/sim.html + run_sim.cjs; controller_baseline.js is an older
                        controller for comparison), lean test - not deployed
```

## Licences and attribution

- MotionBricks SOMA model: licensed by NVIDIA Corporation under the
  [NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/).
  `models/` holds an INT8-quantized derivative of it.
- SOMA body mesh and rig (`data/soma_mesh.*`, `data/skeleton.json`): extracted from the upstream MotionBricks asset
  `soma_base_skel_minimal_no_root.usd`, NVIDIA, CC BY 4.0.
- Clips named `100STYLE/*` and everything in `data/styles/`: the 100STYLE dataset by Ian Mason, Sebastian Starke and
  Taku Komura ([doi:10.5281/zenodo.8127870](https://doi.org/10.5281/zenodo.8127870)), CC BY 4.0, retargeted.
- Clips named `CMU/*`: the data used in this project was obtained from [mocap.cs.cmu.edu](http://mocap.cs.cmu.edu);
  the database was created with funding from NSF EIA-0196217. Retargeted.
- Target windows (`data/windows.*`) and joint limits come from those clips; the transition bank (`data/bank.*`) is
  SOMA model output conditioned on them.
- onnxruntime-web (Microsoft), three.js, coi-serviceworker: MIT, licence files in `vendor/`.
