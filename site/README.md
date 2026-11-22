# SOMA Playground (web)

A playable third-person demo in which every movement of the character is generated live by
NVIDIA's MotionBricks SOMA network, running in the browser with onnxruntime-web (WebAssembly).
It works with touch controls on phones and with keyboard and mouse on desktop, and the controller
can be tuned in the settings panel.

Nothing is pre-baked: the page sends the last poses and a target pose window to the network,
then decodes and post-processes the result in JavaScript. The pipeline is a port of the Python
runtime in `runtime/somart/`, checked against it to float precision (`build/test/parity.html`).

## Controls

| Phone | Desktop | Action |
|---|---|---|
| Left joystick (push far to run) | `WASD` / arrow keys | Move (camera-relative) |
| Hold Sprint | `Shift` | Sprint |
| | `Z` | Walk |
| Crouch button | `C` | Crouch / stand |
| Jump button | `Space` | Jump |
| Dodge button | `F` | Dodge (quick side-step) |
| Duck button | `X` | Duck |
| Sit button | `E` | Sit on a chair / stand up |
| Drag the scene | Drag, mouse wheel | Orbit the camera (desktop: wheel zooms) |

## Model and settings

| Model | Network | Download (cached after the first visit) |
|---|---|---|
| Balanced | INT8 pose transformer; other weights stored as int8 per channel, computed in FP32 | 193 MB |

Built from `models/soma/optimized/onnx/soma_int8_pose.onnx` by `tools/build_web_models.py`. The
float MatMul/Conv weights are stored as int8 with one scale per output channel and expanded back
to float32 when the model loads, so the arithmetic is unchanged. Held-out eval set (187 windows):
joint error 7.10 cm vs 7.08 cm for the source model, foot slide 11.1 vs 10.6 cm/s.

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
bank, and both are fixed (`build/test/startlab.html` reproduces the experiments):

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

## Real walking (100STYLE motion capture)

Walking targets also come from real human motion capture: the **100STYLE** dataset (Mason, Starke and
Komura, CC BY 4.0), one performer walking, jogging, idling, starting and stopping in many styles. The SOMA
network still generates every frame; the capture only decides where each segment should end up.

- `tools/mocap/style100.py fetch` downloads just the files used (about 120 MB of a 1.5 GB zip, via HTTP
  range requests); `retarget` maps them onto the SOMA rig: the SOMA bind pose (an A-pose) is swung onto the
  performer's T-pose bone by bone, feet and toes stay flat, the pelvis follows the performer's hip sockets
  scaled by leg length (0.984), and each clip is lowered so the standing foot is at the same height as in the
  NVIDIA clips. Bone directions agree within 1 deg; stance-foot pitch 21-23 deg vs 19 deg in NVIDIA walks;
  pelvis height within 1 cm at the same speed (`check`, `tests/test_style100.py`).
- The default library adds 5 minutes of neutral walking, a slow jog, idling and starts/stops (31 clips, 3.5 MB).
- **Gait-aware matching** (`js/library.js`): each candidate is labelled walk or jog from its stride period
  (walks take 31-52 frames, jogs 15-29); walking prefers walking strides and running prefers running ones.
  Walk and jog overlap in speed (both libraries have jogs at 0.8-2 m/s), so this was ambiguous before.
- **Walk style** (Settings): Natural, Proud, Strutting, Rushed, Sad, Elderly or Hands behind back. Each style
  file (1.3-2 MB) is downloaded only when chosen; its clips win wherever they cover the requested speed, and
  the character walks at the style's own pace (0.35 m/s for Elderly up to 1.95 m/s for Rushed). Running and
  sprinting stay on the default library. Elated was dropped after looking at it on SOMA (an exaggerated
  crouched bounce); styles with hands in pockets would sink the hands into the unclothed mesh.

Measured with `build/test/sim.html?script=walk` (walking, turns, a half-pushed stick, stop, jog start),
3 runs each, NVIDIA clips only vs with 100STYLE:

| | NVIDIA only | + 100STYLE |
|---|---|---|
| toe slide while on the ground, median | 0.099 m/s | 0.085 m/s |
| ... 90th percentile | 1.79 m/s | 1.51 m/s |
| ground frames sliding > 0.2 m/s | 37 % | 33 % |
| stop from walking | 0.17-0.24 s | 0.04-0.05 s |
| slow walk (half stick) reached | 0.09-0.42 s | 0.11-0.12 s |
| joint speed p99 | 11.7 rad/s | 12.1 rad/s |

On the running script nothing changed beyond noise, except walk-to-sprint, which became about 0.1 s slower
(0.24 -> 0.34 s): from the new walking pose the transition bank picks a 90 deg entry for a 45 deg change.
No NaNs, failures, lurches or freezes in any run, also with 450 ms of added delay. Building the library
takes 159 ms here (73 ms before); one target search 0.8 ms (0.3 ms before).

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

## Animals (trained network, realistic models)

Settings, Character: Dog, Wolf, Fox, Cat, Horse, Brown bear, Elephant, Tiger, Jaguar, Rabbit (or "Play an
animal" on the start screen; each model is 0.2-1.9 MB and loads when first picked). The SOMA network only knows
human motion and was released without training code, so the animals have their own network, trained here from
scratch, driving realistic skinned 3D models:

- **Data**: 41 minutes of dog motion capture from Zhang, Starke, Komura and Saito, "Mode-Adaptive
  Neural Networks for Quadruped Motion Control" (SIGGRAPH 2018), CC BY-NC 4.0 (non-commercial use
  only). 51 clips at 60 fps, mirrored left/right; about 3 minutes where the dog stands on a raised
  surface are dropped. Three actions are labelled automatically from the posture: lying down, leaping
  (flight phases at speed) and rearing up on the hind legs (about 4 minutes of the data).
- **Model** (`tools/animal/train_mann.py`): a gating network blends 4 expert MLPs (2 hidden layers
  of 256) per frame. Input: 1 s of past path, 1 s of future path (the network's own prediction
  blended with the player's joystick) with the action labels, the current joint positions and
  velocities. Output: the next pose, root motion, the future path and paw contacts. 0.77 M weights,
  1.5 MB as float16; trained for 100 epochs (16 minutes) on 253,336 samples on this machine's CPU;
  validation loss 0.24 on 5 held-out clips (training loss 0.08).
- **One model file**: both networks ship as one GGUF model, split like llama.cpp's `gguf-split` into 13 shards of
  at most 14 MB (`models/balanced/motionbricks-000NN-of-00013.gguf`, `tools/gguf/build_gguf.py`, read by `js/gguf.js`).
  Shard 1 (1.6 MB) holds the animal network and the metadata, so the animal mode downloads only that shard. GGUF is
  the ggml / llama.cpp container; llama.cpp cannot run these networks, the browser does.
- **Runtime** (`js/animalnet.js`): evaluated every frame in a WebAssembly SIMD kernel (0.2 ms; plain
  JavaScript fallback; matches PyTorch to 1e-6), bone lengths kept. Paws the network marks as in contact stay planted (up to
  10 cm of lag before they slide, never beyond the leg's reach).
- **Bodies** (`js/animalmesh.js`): Creative Commons models from Sketchfab, taken from the Objaverse
  mirror on Hugging Face and packed with `tools/animal/pack_models.py` (animations removed, textures at
  most 1024 px; credits in the settings panel and `NOTICE`). Each model gets a small bone map (pelvis,
  spine, neck, head, four legs, tail). Every frame each mapped bone is turned rigidly so its part of the
  body lines up with the network's joints (the body's left axis fixes the twist); the spine and tail bend
  progressively; a two-bone IK puts every paw exactly on the network's planted paw. The network's legs are
  scaled to each model's leg lengths and the body is pitched so front and hind paws stand level. Bones the
  map does not name (ears, jaw, toes, trunk, mane) follow their parents. Network plus bone update: about
  1.5 ms per frame.
- **Measured** (`build/test/animal_sim.html`, dog; `&model=1` includes the bone update): frame-to-frame
  joint acceleration of the network output is at or below the real mocap's in every gait (walk 1.7 vs
  2.5 mm/frame², trot 5.3 vs 5.5, gallop 6.7 vs 7.6), so the motion is as smooth as the recording.
  Planted paws slide 0.15 m/s while moving (the recording: 0.10 m/s; before planting: 0.38 m/s).
- **Other species** move with the dog-trained motion on their own body proportions and are time-scaled
  by size (larger animals step more slowly). Per-species styles sit on top (`style` in `js/animalmesh.js`):
  rabbits bound with each pair of legs together, elephants never gallop and move slower, bears are slower,
  cats and foxes step quicker, horses run faster. Planted paws keep their toes at the standing angle.
  Bigger animals get straighter legs (`legReach`: the network's legs are 5-10% longer than the model's,
  so the model's legs extend more), the horse keeps its cannon bones near vertical (`metaStiff`) and
  carries its head 25 deg higher than a dog (`headLift`).
- **Bigger is not better here**: a 6-expert, 320-unit network trained for 150 epochs reached a lower
  validation loss (0.221 vs 0.242) but drove worse in closed loop (stops 4x slower, more paw sliding,
  jerkier trot, twice the cost), so it was not shipped. A run with noise on the input pose
  (`--noise 0.1 --keep-best`) was worse still (walk too fast, horse never fully lay down, 4x paw sliding).
- **Rig fixes** (audit: `build/test/animal_rig.html`, `bind_check.html`, frame-exact captures with `SOMA_DEBUG.step`):
  - *Reference pose = the skin's bind pose.* The node pose stored in each file is whatever frame the author
    exported: a fox lying curled with its head on the ground, a galloping horse, mid-stride cats. Rest
    directions, twists and toe angles were measured on it, so the fox's head pointed at the ground and paws
    turned sideways. The driven chains (and toes) are now put in the bind pose, rebuilt from the inverse bind
    matrices joint by joint (independent of each file's units and axes), and stood upright. Skin in the
    reference pose is now undeformed (edge-length spread 1.00-1.06; it was up to 1.64). Faces, ears and manes
    keep the authored expression.
  - *Paws*: a planted paw stands flat in its bind orientation turned to the body's heading; a lifted paw
    flexes like the network's, at most 100 deg and without twist; the leg IK reaches the wrist that puts the
    toe tip on the network's contact point. (Before, the paw bone was aimed at the toe tip with a minimal
    rotation, which tipped paws 80-140 deg and twisted them 20-60 deg.)
  - *No candy-wrapper*: limb bones twist at most 12 deg about their own axis, the head rolls at most 10 deg.
  - *Misparented helpers*: skin bones that sit on a driven joint but hang off another bone (the horse's hip
    bone on the second spine bone) follow the joint they sit on.
  - Dual-quaternion skinning was measured (CPU) and did not reduce skin stretch on these models, so the
    meshes keep three.js linear blend skinning.
- **Muscles, breathing, soft tissue**: a vertex-shader layer on each animal mesh (per-bone displacement along
  the normal, from a small float texture): loaded legs bulge (more when the joint below is flexed), the chest
  breathes 0.35 times a second at rest up to ~2.2 when sprinting (panting), and the belly rides the body's
  bounce on a 3.2 Hz spring. Up to ~1 cm on a dog; follows the "Muscles, breathing and soft tissue" setting.
- **Response and stability**: the network's inputs are kept within 8 standard deviations of its training
  data. Before, harsh input (180 deg reversals, direction changes every 0.4 s; `animal_sim.html?stress=1`)
  made every species except the horse diverge, even with the original settings. With that guard the
  player's path steers the network much harder (`RESPONSE` in `js/animalnet.js`):

  | 90 deg turn / stop | Before | Now |
  |---|---|---|
  | dog | 0.7-1.4 s / 1.17 s | 0.5 s / 0.82 s |
  | cat, fox, rabbit | 0.6-0.9 s / 0.6 s | 0.23-0.38 s / 0.33-0.5 s |
  | horse | 1.6-2.5 s / 1.8 s | 0.83-0.95 s / 1.08 s |
  | elephant | 1.5-2.4 s / did not stop in 2 s | 0.85-1.02 s / 1.23 s |

  Sprint speeds are unchanged; paw sliding while moving is within about +/-25 % of before per species.
- Controls: move, Sprint (fast gait), Jump (leap), Punch / Kick (pounce), Dodge (side jump),
  Block (rear up; from a run the animal pulls up first), Duck or Sit (lie down / get up). The
  elephant rears instead of jumping. Settings can switch back to the earlier procedural animation
  (simple shapes).

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

- **Animal network in WebAssembly SIMD**: the expert-blended layers run in a 400-byte SIMD kernel
  (`build/tools/mix_simd.wat`, embedded in `js/animalnet.js`): 0.19 ms per step instead of 1.32 ms
  (6.9x), outputs equal to the JS path within 1e-6, identical rollout metrics. Browsers without SIMD
  use the JS path. Main-thread cost of an animal: 5.3 -> 1.6 ms per frame here.
- **Adaptive quality** (Settings, on by default): when the frame rate stays below 40 fps the resolution,
  then the shadow quality, then the shadows step down (4 levels, "lite n" in the fps chip); above
  57 fps they step back up, one change per 4-8 s. On this machine's software renderer (a stand-in
  for a weak GPU): 12 -> 23-27 fps.
- Human: the controller costs 1.2-1.4 ms per frame on the main thread; the SOMA network runs in a
  worker (0.19 s per generation with 4 threads here, about 4 generations per second while moving,
  1.5 when idle). Slow devices get fewer, longer generations automatically (one request in flight).
- The human and animal networks are one model file (split GGUF) but never run at the same time, so
  fusing their computation would not lower the CPU load; it would make animal-only play download and
  start the 152 MB human network.

## Performance (measured here, not on phones)

| Where | Network time per generation | Plan-to-result latency |
|---|---|---|
| Headless Chromium, 1 WASM thread | 0.23-0.31 s | ~0.3 s |
| Headless Chromium, 4 WASM threads, CPU slowed 4x, phone layout | 0.2-0.43 s | 0.41-0.56 s |

No phone was available to test. A recent phone is likely 2-3x slower per thread than this machine,
i.e. about 0.5-0.8 s per generation on one thread (claude.ai copy) and less with threads (GitHub
Pages). The first visit downloads 193 MB.

## Run locally

```bash
cd site/build && npm install            # onnxruntime-web, three, coi-serviceworker (dev only)
node serve.mjs 8123 --isolate           # COOP/COEP headers -> multi-threaded WASM
# open http://localhost:8123/
```

Rebuild the generated pieces when their sources change:

```bash
python3 tools/build_web_models.py       # site/models/  (needs models/soma/optimized/onnx/)
python3 tools/export_web_data.py        # site/data/    (needs third_party/ sample clips)
node site/build/tools/gen_bank.cjs 8124 # site/data/bank.* (with `node site/build/serve.mjs 8124 --isolate`)
python3 site/build/make_index.py        # site/index.html from index.src.html
```

## Deploy to GitHub Pages

`.github/workflows/pages.yml` publishes `site/` (without `build/`) on every push to this
branch that touches `site/`, once Pages is enabled:

1. Settings, Pages, Build and deployment, Source: **GitHub Actions**.
2. Actions, **Deploy SOMA Playground**, **Run workflow** (or push a change under `site/`).
3. The site appears at `https://<owner>.github.io/<repo>/`.

Notes before enabling it:

- GitHub Pages for a private repository needs a paid plan (Pro, Team or Enterprise), and the
  published site is public unless the organization uses Enterprise Cloud private Pages.
- `site/data/windows.*`, `clips.*` and `bank.*` are derived from the MotionBricks sample clips, which are under the
  NVIDIA Sample Data Evaluation License (internal evaluation and testing). Publishing a public
  site redistributes that derived data. Check that license, or replace the windows with motion
  you have the rights to (`tools/export_web_data.py` builds them from any SOMA BVH clips),
  before making the site public.
- GitHub Pages has a soft limit of 100 GB of traffic per month, roughly 500 first visits.
- Pages cannot set COOP/COEP headers, so `coi-serviceworker.js` adds them through a service
  worker (one automatic reload on the first visit) to enable multi-threaded WASM. Without it
  the page still works on one thread.

## claude.ai copy

`build/stage_artifact.py` stages a copy for a claude.ai artifact, which accepts
standard web media types only: the `.bin` parts are copied byte-for-byte under `.wasm` names
and the page is told so by `<meta name="soma-bin-ext">`. Service workers are not available
there, so it runs single-threaded.

## Files

```
index.src.html          page source (HUD, controls, panel); make_index.py -> index.html
js/main.js              scene, input, settings panel, frame loop
js/controller.js        planning, splicing, instant transitions, gait-cycle extension,
                        inertialization, foot locking, fingers, head look-ahead
js/library.js           continuous mocap clips with stride phase and gait, walk styles (target selection)
js/bank.js              transition bank + motion-matching features
js/body.js              muscles, breathing, soft tissue (vertex shader)
js/fight.js             fighting key poses (IK on the idle pose)
js/animals.js           species shapes + the earlier procedural animal animation
js/animalnet.js         trained animal network: JS inference, control loop, paw planting
js/animalmesh.js        realistic animal models: bone maps, retargeting onto skinned meshes, credits
js/settings.js          presets and defaults
js/codec.js             SOMA encode/decode, endpoint correction, velocity matching, joint limits
js/math3d.js            quaternion and pose helpers
js/worker.js            model download and cache, onnxruntime-web inference
data/                   skeleton, action windows, clips (30 fps + mirrored; 100STYLE walking),
                        styles/ (optional walk styles, loaded on demand), transition bank,
                        joint limits, SOMA body mesh; animal/: trained network,
                        models/*.glb (10 animals, CC licences, see NOTICE)
models/                 models.json + model parts (14 x 14 MB)
vendor/                 onnxruntime-web 1.31.1, three.js 0.170.0 + GLTFLoader (MIT, unmodified)
coi-serviceworker.js    COOP/COEP shim for static hosts (MIT)
build/                  dev server, index builder, artifact stager, parity and timing tests
                        (test/sim.html + run_sim.cjs; controller_baseline.js is the pre-fix
                        controller for comparison), bank generator (tools/gen_bank.*) - not deployed
```

## Licences and attribution

- MotionBricks SOMA model: licensed by NVIDIA Corporation under the
  [NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/).
  `models/` holds an INT8-quantized derivative of it (int8/FP16 weight storage).
- SOMA body mesh (`data/soma_mesh.*`): extracted from the upstream MotionBricks asset
  `soma_base_skel_minimal_no_root.usd`, NVIDIA, CC BY 4.0 (upstream assets licence).
- Motion windows and clips (`data/windows.*`, `data/clips.*`, mirrored copies included): derived
  from MotionBricks sample clips, NVIDIA Sample Data Evaluation License.
- Clips named `100STYLE/*` in `data/clips.*` and everything in `data/styles/`: the 100STYLE dataset by Ian Mason,
  Sebastian Starke and Taku Komura ([doi:10.5281/zenodo.8127870](https://doi.org/10.5281/zenodo.8127870)),
  CC BY 4.0, retargeted onto the SOMA rig and resampled to 30 fps by `tools/mocap/style100.py`.
- Transition bank (`data/bank.*`): SOMA model output conditioned on 4-frame windows of those clips.
- onnxruntime-web (Microsoft), three.js, coi-serviceworker: MIT, licence files in `vendor/`.
