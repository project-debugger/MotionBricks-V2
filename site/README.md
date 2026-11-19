# SOMA Playground (web)

A playable third-person demo in which every movement of the character is generated live by
NVIDIA's MotionBricks SOMA network, running in the browser with onnxruntime-web (WebAssembly).
It works with touch controls on phones and with keyboard and mouse on desktop. You can switch
between two quantized copies of the model and tune the controller in the settings panel.

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

## Models and settings

| Model | Network | Download (cached after the first visit) |
|---|---|---|
| Balanced | INT8 pose transformer, FP16-stored root model and decoder | 245 MB |
| Light | INT4 pose transformer, FP16-stored root model and decoder | 187 MB |

Both come from `models/soma/optimized/` (see `docs/OPTIMIZATION.md`) via
`tools/build_web_models.py`, which stores large float weights as FP16 with a Cast back to
FP32 (identical maths at FP32) and splits each model into 14 MB parts.

The settings panel (gear icon) has presets (Responsive, Balanced, Smooth), sliders for speeds,
responsiveness, planning horizon, animation speed, smoothing, motion variety and the finger
layer, and toggles for the quality passes: foot locking, anatomical joint limits, procedural
fingers and velocity-matched joins. Settings are kept in the browser's local storage.

## What runs per generation

1. The controller predicts where the character will be at the end of the next segment and
   picks a matching 4-frame target window from a library of 709 mocap windows (by velocity,
   turn rate and crouch state), or an action's key pose (jump, sit, dodge, ...).
2. `codec.js` encodes 4 context + 4 target frames, the worker runs the ONNX model, then decodes
   frames and contacts, applies endpoint correction, velocity matching and the learned
   anatomical joint limits (knees and elbows as pure hinges, toe and jaw ranges).
3. Every displayed frame then gets foot locking (two-bone IK on model contacts),
   inertialization of any splice, and the procedural finger layer (relaxed mocap hand pose,
   curl that grows with speed, follow-through on wrist swings).

Segment length adapts to the measured generation latency, so slower devices plan further ahead
instead of freezing.

## Performance (measured here, not on phones)

| Where | Model | Network time per generation | Plan-to-result latency |
|---|---|---|---|
| Headless Chromium, 4 WASM threads (cross-origin isolated) | Balanced | ~0.3 s | ~0.5 s |
| claude.ai artifact viewer, 1 thread (no cross-origin isolation) | Light | ~0.8-1.0 s | ~0.9-1.0 s |

Both were on this cloud machine's CPU with software rendering. No phone was available to test,
so phone timings are unmeasured: expect roughly 1 s per generation on a recent phone, which the
controller hides by planning longer segments. Responsiveness to input changes degrades
accordingly. The first visit downloads 187 or 245 MB.

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
- GitHub Pages has a soft limit of 100 GB of traffic per month, roughly 400-500 first visits.
- Pages cannot set COOP/COEP headers, so `coi-serviceworker.js` adds them through a service
  worker (one automatic reload on the first visit) to enable multi-threaded WASM. Without it
  the page still works on one thread.

## claude.ai copy

`build/stage_artifact.py` stages a Light-only copy for a claude.ai artifact, which accepts
standard web media types only: the `.bin` parts are copied byte-for-byte under `.wasm` names
and the page is told so by `<meta name="soma-bin-ext">`. Service workers are not available
there, so it runs single-threaded.

## Files

```
index.src.html          page source (HUD, controls, panel); make_index.py -> index.html
js/main.js              scene, input, settings panel, frame loop
js/controller.js        motion library, planning, splicing, inertialization, foot locking, fingers
js/codec.js             SOMA encode/decode, endpoint correction, velocity matching, joint limits
js/math3d.js            quaternion and pose helpers
js/worker.js            model download and cache, onnxruntime-web inference
data/                   skeleton, motion windows, joint limits, SOMA body mesh
models/                 models.json + model parts
vendor/                 onnxruntime-web 1.31.1, three.js 0.170.0 (MIT, unmodified)
coi-serviceworker.js    COOP/COEP shim for static hosts (MIT)
build/                  dev server, index builder, artifact stager, parity tests (not deployed)
```

## Licences and attribution

- MotionBricks SOMA model: licensed by NVIDIA Corporation under the
  [NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/).
  `models/` holds INT8/INT4-quantized, FP16-stored derivatives of it.
- SOMA body mesh (`data/soma_mesh.*`): extracted from the upstream MotionBricks asset
  `soma_base_skel_minimal_no_root.usd`, NVIDIA, CC BY 4.0 (upstream assets licence).
- Motion windows (`data/windows.*`): derived from MotionBricks sample clips, NVIDIA Sample Data
  Evaluation License.
- onnxruntime-web (Microsoft), three.js, coi-serviceworker: MIT, licence files in `vendor/`.
