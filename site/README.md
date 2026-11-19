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
and instant steering, and toggles for foot locking, anatomical joint limits, procedural fingers,
velocity-matched joins and head-leads-turns. Settings are kept in the browser's local storage.

## What runs per generation

1. The controller predicts where the character will be at the end of the next segment, starting
   from the root position and velocity of the frame the segment starts from, and picks a matching
   4-frame target window from a library of 709 mocap windows (by velocity and crouch state), or an
   action's key pose (jump, sit, dodge, ...).
2. `codec.js` encodes 4 context + 4 target frames, the worker runs the ONNX model, then decodes
   frames and contacts, applies endpoint correction, velocity matching and the learned
   anatomical joint limits (knees and elbows as pure hinges, toe and jaw ranges).
3. Every displayed frame then gets foot locking (two-bone IK on model contacts),
   inertialization of any splice, the procedural finger layer and the head/chest look-ahead.

## Slow devices (phones)

A phone generates more slowly than a desktop, so the controller is built not to depend on speed:

- **No catch-up lurches**: targets are predicted from the animation's own momentum, never from a
  free-running gameplay position, so the model is never asked to cover an impossible distance.
- **No freezes**: if a result is late, the buffer is extended by looping the last gait cycle
  (re-rooted, seam blended) until it arrives; the late result is then inertialized in.
- **Free worker**: while the input is steady, segments are long and replanned only when the
  buffer runs low, so a new input usually starts generating at once.
- **Instant turns**: direction changes rotate the buffered motion right away (up to 200 deg/s,
  up to 90 deg); the rotation is replayed onto any result still in flight so it stays continuous.
  Reversals over 135 deg are left to the model.
- **Head leads**: the head and chest turn toward the new direction immediately.
- The joystick switches walk/run with hysteresis, so a wobbling thumb does not flip the gait.

Measured with `build/test/sim.html` (real model, scripted 21 s input sequence, extra delay added
to every generation to emulate slower devices; Balanced, one WASM thread, ~0.26 s per generation
here; one run each, so response times vary by about 0.3 s between runs):

| Delay added | Controller | Frozen | Lurching (> 8 m/s) | Top speed | Joint speed p99 | Turn 90 deg | Start | Stop |
|---|---|---|---|---|---|---|---|---|
| 0 | before | 0.2 s | 0.5 s | 9.9 m/s | 28 rad/s | 0.45 s | 0.76 s | 0.58 s |
| 0 | after | 0 | 0 | 6.0 m/s | 15 rad/s | 0.20 s | 0.78 s | 0.77 s |
| 0.3-0.6 s (random) | before | 0.95 s | 1.8 s | 11.0 m/s | 80 rad/s | 0.99 s | 1.31 s | 1.26 s |
| 0.3-0.6 s (random) | after | 0 | 0 | 6.0 m/s | 15 rad/s | 0.21 s | 1.31 s | 1.48 s |
| 1.2 s | before | 3.0 s | 3.5 s | 16.5 m/s | 89 rad/s | 1.74 s | 2.77 s | 1.87 s |
| 1.2 s | after | 0 | 0 | 6.8 m/s | 16 rad/s | 0.23 s | 2.70 s | not reached in 2 s |

Starting and stopping still take about one generation plus a natural human acceleration or
deceleration: the first new frames cannot exist before the network has produced them. Stops are
~0.2 s slower than before because the model now brakes from the real momentum instead of being
pulled toward an unreachable target (the cause of the lurches).

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
- `site/data/windows.*` is derived from the MotionBricks sample clips, which are under the
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
js/controller.js        motion library, planning, splicing, steering, gait-cycle extension,
                        inertialization, foot locking, fingers, head look-ahead
js/settings.js          presets and defaults
js/codec.js             SOMA encode/decode, endpoint correction, velocity matching, joint limits
js/math3d.js            quaternion and pose helpers
js/worker.js            model download and cache, onnxruntime-web inference
data/                   skeleton, motion windows, joint limits, SOMA body mesh
models/                 models.json + model parts (14 x 14 MB)
vendor/                 onnxruntime-web 1.31.1, three.js 0.170.0 (MIT, unmodified)
coi-serviceworker.js    COOP/COEP shim for static hosts (MIT)
build/                  dev server, index builder, artifact stager, parity and timing tests
                        (test/sim.html + run_sim.cjs; controller_baseline.js is the pre-fix
                        controller for comparison) - not deployed
```

## Licences and attribution

- MotionBricks SOMA model: licensed by NVIDIA Corporation under the
  [NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/).
  `models/` holds an INT8-quantized derivative of it (int8/FP16 weight storage).
- SOMA body mesh (`data/soma_mesh.*`): extracted from the upstream MotionBricks asset
  `soma_base_skel_minimal_no_root.usd`, NVIDIA, CC BY 4.0 (upstream assets licence).
- Motion windows (`data/windows.*`): derived from MotionBricks sample clips, NVIDIA Sample Data
  Evaluation License.
- onnxruntime-web (Microsoft), three.js, coi-serviceworker: MIT, licence files in `vendor/`.
