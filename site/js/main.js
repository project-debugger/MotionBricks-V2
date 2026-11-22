// SOMA Playground: real NVIDIA MotionBricks SOMA generation in the browser (onnxruntime-web),
// driven by phone touch controls or keyboard/mouse, rendered on NVIDIA's SOMA body mesh.
import * as THREE from 'three';
import * as M from './math3d.js';
import { bindLimits } from './codec.js';
import { Controller, Library } from './controller.js';
import { ClipLibrary, legIndices } from './library.js';
import { TransitionBank } from './bank.js';
import { BodyFX } from './body.js';
import { registerFightActions } from './fight.js';
import { PRESETS, DEFAULTS } from './settings.js';

const $ = (s) => document.querySelector(s);
const BASE = new URL('.', location.href).href;
const IS_TOUCH = matchMedia('(pointer: coarse)').matches || 'ontouchstart' in window;

// ------------------------------------------------------------------ settings
let S = { ...DEFAULTS };
try { Object.assign(S, JSON.parse(localStorage.getItem('soma-settings') || '{}')); } catch { /* private mode */ }
const save = () => { try { localStorage.setItem('soma-settings', JSON.stringify(S)); } catch { /* ignore */ } };
if ((S.settingsVersion || 1) < 2) Object.assign(S, { fingerCurlRun: DEFAULTS.fingerCurlRun, torsoCalm: DEFAULTS.torsoCalm, settingsVersion: 2 }); // new defaults reach saved settings
if (S.settingsVersion < 3) Object.assign(S, PRESETS.shooter, { preset: 'shooter', settingsVersion: 3 });

// ------------------------------------------------------------------ data
async function getJSON(p) { const r = await fetch(BASE + p); if (!r.ok) throw new Error(`${p}: ${r.status}`); return r.json(); }
// hosts that refuse .bin files (claude.ai artifacts) serve the same bytes under another extension
const BIN_EXT = document.querySelector('meta[name="soma-bin-ext"]')?.content || '.bin';
async function getBin(p) { p = p.replace(/\.bin$/, BIN_EXT); const r = await fetch(BASE + p); if (!r.ok) throw new Error(`${p}: ${r.status}`); return r.arrayBuffer(); }

// ------------------------------------------------------------------ renderer
function gridTexture() {
  const c = document.createElement('canvas'); c.width = c.height = 512;
  const g = c.getContext('2d');
  g.fillStyle = '#c6ccd3'; g.fillRect(0, 0, 512, 512);
  g.fillStyle = '#b9c0c8'; g.fillRect(0, 0, 256, 256); g.fillRect(256, 256, 256, 256);
  g.strokeStyle = '#97a1ab'; g.lineWidth = 3; g.strokeRect(0, 0, 512, 512);
  const t = new THREE.CanvasTexture(c);
  t.wrapS = t.wrapT = THREE.RepeatWrapping; t.repeat.set(200, 200); t.anisotropy = 8; t.colorSpace = THREE.SRGBColorSpace;
  return t;
}
// Adapts rendering to the device: when the frame rate stays low the resolution, then the shadow quality, then the
// shadows step down; with headroom they step back up (at most one change per few seconds, no flicker).
export class QualityGovernor {
  constructor(R, base) { this.R = R; this.base = base; this.level = 0; this.t = 0; this.n = 0; this.acc = 0; this.hold = 3; this.good = 0; }
  static LEVELS = [{ ratio: 1, shadows: 2 }, { ratio: 0.75, shadows: 2 }, { ratio: 0.6, shadows: 1 }, { ratio: 0.5, shadows: 0 }];
  apply(level) {
    const L = QualityGovernor.LEVELS[level], r = this.R.renderer;
    this.level = level;
    r.setPixelRatio(Math.max(0.5, this.base * L.ratio));
    const sm = L.shadows > 0;
    if (r.shadowMap.enabled !== sm || (sm && (r.shadowMap.type === THREE.PCFSoftShadowMap) !== (L.shadows === 2))) {
      r.shadowMap.enabled = sm; r.shadowMap.type = L.shadows === 2 ? THREE.PCFSoftShadowMap : THREE.PCFShadowMap;
      this.R.sun.castShadow = sm; if (this.R.sun.shadow.map) { this.R.sun.shadow.map.dispose(); this.R.sun.shadow.map = null; }
      this.R.scene.traverse((o) => { if (o.material) [].concat(o.material).forEach((m) => { m.needsUpdate = true; }); });
    }
  }
  // dt: real frame time (s); enabled: setting; returns true when the level changed
  update(dt, enabled) {
    if (!enabled) { if (this.level) this.apply(0); return false; }
    if (dt <= 0 || dt > 0.5) return false; // tab switches, stalls
    this.hold -= dt; this.acc += dt; this.n++;
    if (this.acc < 2) return false;
    const fps = this.n / this.acc; this.acc = 0; this.n = 0;
    this.good = fps > 57 ? this.good + 2 : 0;
    if (this.hold > 0) return false;
    if (fps < 40 && this.level < QualityGovernor.LEVELS.length - 1) { this.apply(this.level + 1); this.hold = 4; return true; }
    if (this.good >= 8 && this.level > 0) { this.apply(this.level - 1); this.hold = 8; this.good = 0; return true; }
    return false;
  }
}

function buildScene(mesh) {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  const basePixelRatio = Math.min(devicePixelRatio, IS_TOUCH ? 1.5 : 2);
  renderer.setPixelRatio(basePixelRatio);
  renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.outputColorSpace = THREE.SRGBColorSpace;
  $('#stage').appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color('#dde4ec'); scene.fog = new THREE.Fog('#dde4ec', 18, 70);
  scene.add(new THREE.HemisphereLight('#ffffff', '#7b838c', 1.35));
  const sun = new THREE.DirectionalLight('#fff4e6', 2.5);
  sun.castShadow = true; sun.shadow.mapSize.set(1024, 1024);
  Object.assign(sun.shadow.camera, { left: -4, right: 4, top: 4, bottom: -4, near: 0.5, far: 30 });
  sun.shadow.bias = -0.0004; sun.shadow.normalBias = 0.02; scene.add(sun, sun.target);
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(400, 400), new THREE.MeshLambertMaterial({ map: gridTexture() }));
  ground.rotation.x = -Math.PI / 2; ground.receiveShadow = true; scene.add(ground);
  // landmarks so speed and direction read on screen
  const r = M.rng(3), colors = ['#9aa7b5', '#c18a5a', '#7f9a8a', '#a2849c'], landmarks = [];
  for (let i = 0; i < 60; i++) {
    const h = 0.4 + r() * 2.2, w = 0.4 + r() * 1.0;
    const b = new THREE.Mesh(new THREE.BoxGeometry(w, h, w), new THREE.MeshLambertMaterial({ color: colors[i % 4], transparent: true }));
    const a = r() * 6.283, d = 6 + r() * 40;
    b.position.set(Math.cos(a) * d, h / 2, Math.sin(a) * d); b.castShadow = true; b.receiveShadow = true; scene.add(b); landmarks.push(b);
  }
  // SOMA skinned mesh
  const geo = new THREE.BufferGeometry(), buf = mesh.bin, meta = mesh.meta;
  const view = (k, T) => new T(buf, meta[k].offset, meta[k].length);
  geo.setAttribute('position', new THREE.BufferAttribute(view('positions', Float32Array), 3));
  geo.setIndex(new THREE.BufferAttribute(view('indices', Uint32Array), 1));
  geo.setAttribute('skinIndex', new THREE.Uint16BufferAttribute(view('skinIndex', Uint16Array), 4));
  geo.setAttribute('skinWeight', new THREE.BufferAttribute(view('skinWeight', Float32Array), 4));
  geo.computeVertexNormals();
  const inv = view('inverseBind', Float32Array), inverses = [];
  for (let j = 0; j < 77; j++) inverses.push(new THREE.Matrix4().fromArray(inv, j * 16));
  const bones = inverses.map(() => { const b = new THREE.Bone(); scene.add(b); return b; });
  // skin: soft sheen at grazing angles reads as skin rather than plastic
  const skin = new THREE.MeshPhysicalMaterial({ color: '#c9b3a5', roughness: 0.55, sheen: 0.4, sheenRoughness: 0.6, sheenColor: new THREE.Color('#ffc9b5'), specularIntensity: 0.35 });
  const body = new THREE.SkinnedMesh(geo, skin);
  body.castShadow = true; body.frustumCulled = false; body.bind(new THREE.Skeleton(bones, inverses), new THREE.Matrix4()); scene.add(body);
  const camera = new THREE.PerspectiveCamera(42, 1, 0.05, 120);
  const chair = new THREE.Group(); chair.visible = false; scene.add(chair);
  const wood = new THREE.MeshLambertMaterial({ color: '#b5763f' }), dark = new THREE.MeshLambertMaterial({ color: '#3d4650' });
  chair.userData.build = (seat) => {
    chair.clear();
    const add = (g, m, x, y, z) => { const o = new THREE.Mesh(g, m); o.position.set(x, y, z); o.castShadow = o.receiveShadow = true; chair.add(o); };
    add(new THREE.BoxGeometry(0.5, 0.05, 0.48), wood, 0, seat - 0.025, 0);
    for (const [x, z] of [[-0.22, -0.21], [0.22, -0.21], [-0.22, 0.21], [0.22, 0.21]]) add(new THREE.BoxGeometry(0.045, seat - 0.05, 0.045), dark, x, (seat - 0.05) / 2, z);
    add(new THREE.BoxGeometry(0.5, 0.45, 0.04), wood, 0, seat + 0.25, -0.24);
  };
  const resize = () => { const w = innerWidth, h = innerHeight; renderer.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix(); };
  addEventListener('resize', resize); resize();
  return { renderer, scene, camera, bones, sun, chair, landmarks, geo, inverses, skin, body, basePixelRatio };
}

// ------------------------------------------------------------------ input
const input = { keys: new Set(), joy: [0, 0], sprintHeld: false, walk: false, crouch: false, sitting: false, camYaw: Math.PI * 0.8, camDist: 4.2, pending: [] };
function setupInput(canvas) {
  addEventListener('keydown', (e) => {
    if (e.target.closest && e.target.closest('#panel')) return;
    input.keys.add(e.code);
    const am = { Space: () => 'jump', KeyF: () => 'dodge', KeyX: () => 'duck', KeyJ: nextPunch, KeyK: nextKick, KeyL: () => 'block' }[e.code];
    const a = am && !e.repeat ? am() : null;
    if (a) input.pending.push(a);
    if (e.code === 'KeyC' && !e.repeat) toggleCrouch();
    if (e.code === 'KeyZ' && !e.repeat) { input.walk = !input.walk; }
    if (e.code === 'KeyE' && !e.repeat) toggleSit();
    if (['Space', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.code)) e.preventDefault();
  });
  addEventListener('keyup', (e) => input.keys.delete(e.code));
  addEventListener('blur', () => input.keys.clear());
  // camera orbit: mouse drag / one-finger drag outside the joystick, wheel / pinch zoom
  let drag = null;
  canvas.addEventListener('pointerdown', (e) => { drag = { id: e.pointerId, x: e.clientX }; try { canvas.setPointerCapture(e.pointerId); } catch { /* ignore */ } });
  canvas.addEventListener('pointermove', (e) => { if (drag && e.pointerId === drag.id) { input.camYaw -= (e.clientX - drag.x) * 0.008; drag.x = e.clientX; } });
  canvas.addEventListener('pointerup', () => { drag = null; });
  canvas.addEventListener('wheel', (e) => { input.camDist = Math.min(9, Math.max(2.2, input.camDist * (1 + Math.sign(e.deltaY) * 0.08))); e.preventDefault(); }, { passive: false });
  // virtual joystick
  const pad = $('#joy'), knob = $('#knob');
  let jid = null, c0 = null;
  const R = 56;
  pad.addEventListener('pointerdown', (e) => { jid = e.pointerId; const r = pad.getBoundingClientRect(); c0 = [r.left + r.width / 2, r.top + r.height / 2]; try { pad.setPointerCapture(jid); } catch { /* synthetic or stale pointer */ } move(e); });
  const move = (e) => {
    if (e.pointerId !== jid) return;
    let dx = e.clientX - c0[0], dy = e.clientY - c0[1];
    const d = Math.hypot(dx, dy); if (d > R) { dx *= R / d; dy *= R / d; }
    knob.style.transform = `translate(${dx}px, ${dy}px)`; input.joy = [dx / R, -dy / R];
  };
  pad.addEventListener('pointermove', move);
  const end = (e) => { if (e.pointerId !== jid) return; jid = null; input.joy = [0, 0]; knob.style.transform = ''; };
  pad.addEventListener('pointerup', end); pad.addEventListener('pointercancel', end);
  const hold = $('#b-sprint');
  hold.addEventListener('pointerdown', () => { input.sprintHeld = true; hold.classList.add('on'); });
  for (const ev of ['pointerup', 'pointercancel', 'pointerleave']) hold.addEventListener(ev, () => { input.sprintHeld = false; hold.classList.remove('on'); });
  $('#b-jump').addEventListener('pointerdown', () => input.pending.push('jump'));
  $('#b-dodge').addEventListener('pointerdown', () => input.pending.push('dodge'));
  $('#b-duck').addEventListener('pointerdown', () => input.pending.push('duck'));
  $('#b-punch').addEventListener('pointerdown', () => input.pending.push(nextPunch()));
  $('#b-kick').addEventListener('pointerdown', () => input.pending.push(nextKick()));
  $('#b-block').addEventListener('pointerdown', () => input.pending.push('block'));
  $('#b-crouch').addEventListener('pointerdown', toggleCrouch);
  $('#b-sit').addEventListener('pointerdown', toggleSit);
}
let punchN = 0, kickN = 0;
const PUNCHES = ['jab', 'cross', 'hook', 'uppercut'], KICKS = ['front_kick', 'roundhouse'];
function nextPunch() { return PUNCHES[punchN++ % PUNCHES.length]; }
function nextKick() { return KICKS[kickN++ % KICKS.length]; }
function toggleCrouch() { input.crouch = !input.crouch; $('#b-crouch').classList.toggle('on', input.crouch); }
function toggleSit() { input.pending.push(input.sitting ? 'stand_up' : 'sit'); input.sitting = !input.sitting; input.sitT = 0; $('#b-sit').classList.toggle('on', input.sitting); }

function command(dt) {
  const k = input.keys;
  let ix = (k.has('KeyD') || k.has('ArrowRight') ? 1 : 0) - (k.has('KeyA') || k.has('ArrowLeft') ? 1 : 0);
  let iy = (k.has('KeyW') || k.has('ArrowUp') ? 1 : 0) - (k.has('KeyS') || k.has('ArrowDown') ? 1 : 0);
  let mag = Math.hypot(ix, iy);
  if (mag > 0) { ix /= mag; iy /= mag; mag = 1; }
  let gait = input.crouch ? 'crouch' : input.walk ? 'walk' : 'run';
  const jm = Math.hypot(input.joy[0], input.joy[1]);
  if (jm > 0.15) {
    ix = input.joy[0] / jm; iy = input.joy[1] / jm;
    // walk/run with hysteresis so a wobbling thumb near the boundary does not flip the gait every frame
    input.joyRun = input.joyRun ? jm > 0.5 : jm > 0.7;
    if (!input.crouch) gait = input.joyRun ? 'run' : 'walk';
    mag = input.crouch || !input.joyRun ? Math.min(1, (jm - 0.15) / 0.45 + 0.35) : 1;
  } else input.joyRun = false;
  if ((k.has('ShiftLeft') || k.has('ShiftRight') || input.sprintHeld) && !input.crouch) gait = 'sprint';
  // camera-relative direction (camera sits at yaw, looking at the character)
  const fx = -Math.sin(input.camYaw), fz = -Math.cos(input.camYaw);
  // right = (-fz, fx), forward = (fx, fz) on the ground plane
  let move = [-fz * ix + fx * iy, fx * ix + fz * iy];
  if (input.sitting) {
    move = [0, 0];
    input.sitT += dt;
    if (input.sitT > 1.6) { input.pending.push('sit_idle'); input.sitT = 0; }
  }
  const action = input.pending.shift() || null;
  return { move: [move[0] * mag, move[1] * mag], gait, action };
}

// ------------------------------------------------------------------ UI
const SLIDERS = [
  ['runSpeed', 'Run speed', 2, 5, 0.1, 'm/s'], ['sprintSpeed', 'Sprint speed', 4, 6.5, 0.1, 'm/s'],
  ['responsiveness', 'Responsiveness (lower = snappier)', 0.04, 0.4, 0.01, 's'],
  ['predictionHorizon', 'Planning horizon', 0.5, 1.4, 0.05, 's'], ['playbackRate', 'Animation speed', 0.5, 1.5, 0.05, 'x'],
  ['smoothing', 'Extra smoothing', 0, 0.6, 0.05, ''], ['variation', 'Motion variety', 0, 0.5, 0.05, ''],
  ['snap', 'Snappy starts, stops and turns', 0, 0.6, 0.05, ''], ['lean', 'Lean into turns', 0, 1.5, 0.05, ''], ['fingerCurlRun', 'Finger curl when running', 0, 80, 1, 'deg'], ['fingerFollow', 'Finger follow-through', 0, 20, 1, ''], ['torsoCalm', 'Calm torso twist when running', 0, 1, 0.05, ''],
  ['steering', 'Body-rotation assist (0 = off)', 0, 360, 10, 'deg/s'],
];
const TOGGLES = [['footLocking', 'Foot locking (IK)'], ['anatomical', 'Anatomical joint limits'], ['fingers', 'Procedural fingers'], ['velocityMatching', 'Velocity-matched joins'], ['lookAt', 'Head leads turns'],
  ['instant', 'Instant response (pre-generated SOMA transitions)'], ['bodyFX', 'Muscles, breathing and soft tissue'],
  ['autoQuality', 'Adapt resolution and shadows to this device (keeps the frame rate up on phones and old PCs)']];
let WALK_STYLES = [];
function buildPanel(models, onModel, onWalkStyle) {
  const p = $('#panel-body');
  const mrow = Object.entries(models).map(([k, m]) => `<button class="seg" data-model="${k}" ${m.available ? '' : 'disabled'}>${m.short}<small>${(m.bytes / 1e6).toFixed(0)} MB</small></button>`).join('');
  p.innerHTML = `<h3>Walk style</h3><div class="segs" id="walkstyles"><button class="seg" data-walkstyle="natural">Natural<small>default</small></button>${WALK_STYLES.map((w) => `<button class="seg" data-walkstyle="${w.key}">${w.label}<small>${w.pace.toFixed(2)} m/s</small></button>`).join('')}</div>
    <p class="note" style="margin-top:6px" id="walkstyle-note">Real walking motion capture (100STYLE, CC BY 4.0) used as targets for the SOMA network; a style walks at its own pace. Running is unchanged.</p>
    <h3>Model</h3><div class="segs" id="models">${mrow}</div>
    <h3>Preset</h3><div class="segs" id="presets">${Object.keys(PRESETS).map((k) => `<button class="seg" data-preset="${k}">${k[0].toUpperCase() + k.slice(1)}</button>`).join('')}</div>
    <h3>Tuning</h3>${SLIDERS.map(([k, l, a, b, st, u]) => `<label class="sl"><span>${l} <b id="v-${k}"></b></span><input id="s-${k}" type="range" min="${a}" max="${b}" step="${st}" data-unit="${u}"></label>`).join('')}
    <h3>Quality passes</h3>${TOGGLES.map(([k, l]) => `<label class="tg"><input id="t-${k}" type="checkbox"><span>${l}</span></label>`).join('')}
    <label class="tg"><input id="t-dur" type="checkbox"><span>Let the model choose segment length (smoother, slower to react)</span></label>
    <button class="reset" id="reset">Reset to defaults</button>`;
  const sync = () => {
    p.querySelectorAll('[data-model]').forEach((b) => b.classList.toggle('on', b.dataset.model === S.model));
    p.querySelectorAll('[data-preset]').forEach((b) => b.classList.toggle('on', b.dataset.preset === S.preset));
    p.querySelectorAll('[data-walkstyle]').forEach((b) => b.classList.toggle('on', b.dataset.walkstyle === (S.walkStyle || 'natural')));
    for (const [k] of SLIDERS) { const el = $(`#s-${k}`); el.value = S[k]; $(`#v-${k}`).textContent = `${(+S[k]).toFixed(el.step < 1 ? 2 : 0)} ${el.dataset.unit}`; }
    for (const [k] of TOGGLES) $(`#t-${k}`).checked = !!S[k];
    $('#t-dur').checked = S.durationMode === 'model';
  };
  p.addEventListener('input', (e) => {
    const id = e.target.id;
    if (id.startsWith('s-')) { S[id.slice(2)] = +e.target.value; S.preset = 'custom'; }
    else if (id.startsWith('t-') && id !== 't-dur') S[id.slice(2)] = e.target.checked;
    else if (id === 't-dur') S.durationMode = e.target.checked ? 'model' : 'horizon';
    save(); sync();
  });
  p.addEventListener('click', (e) => {
    const b = e.target.closest('button'); if (!b) return;
    if (b.dataset.preset) { Object.assign(S, PRESETS[b.dataset.preset]); S.preset = b.dataset.preset; }
    if (b.dataset.model && b.dataset.model !== S.model) { S.model = b.dataset.model; onModel(S.model); }
    if (b.dataset.walkstyle) { S.walkStyle = b.dataset.walkstyle; if (onWalkStyle) onWalkStyle(S.walkStyle); }
    if (b.id === 'reset') S = { ...DEFAULTS, model: S.model };
    save(); sync();
  });
  sync();
}

// ------------------------------------------------------------------ boot
async function main() {
  const status = (t) => { $('#status').textContent = t; };
  status('Loading skeleton, motion targets and body mesh...');
  const [skel, lim, wmeta, wbin, mmeta, mbin, models, cmeta, cbin, bmeta, bbin] = await Promise.all([getJSON('data/skeleton.json'), getJSON('data/joint_limits.json'),
    getJSON('data/windows.json'), getBin('data/windows.bin'), getJSON('data/soma_mesh.json'), getBin('data/soma_mesh.bin'), getJSON('models/models.json'),
    getJSON('data/clips.json'), getBin('data/clips.bin'), getJSON('data/bank.json'), getBin('data/bank.bin')]);
  const sk = { names: skel.names, parents: skel.parents, reference: Float64Array.from(skel.reference.flat()) };
  const lib = new Library(sk, wmeta, wbin), limits = bindLimits(lim, sk);
  const fightPoses = registerFightActions(sk, lib, wmeta.actions, limits);
  status('Indexing mocap strides and the instant-transition bank...');
  await new Promise((r) => setTimeout(r, 0));
  const clips = new ClipLibrary(sk, cmeta, cbin, limits), bank = new TransitionBank(sk, bmeta, bbin, limits, legIndices(sk));
  // real walk styles (100STYLE): each style's clips are fetched only when it is chosen
  WALK_STYLES = cmeta.styles || [];
  const loadWalkStyle = async (key) => {
    const w = WALK_STYLES.find((x) => x.key === key);
    if (!w || clips.styles[key]) return;
    const note = $('#walkstyle-note'), was = note?.textContent;
    if (note) note.textContent = `Loading ${w.label} walking (${(w.bytes / 1e6).toFixed(1)} MB)...`;
    try {
      const [m, b] = await Promise.all([getJSON(`${w.file}.json`), getBin(`${w.file}.bin`)]);
      clips.addStyle(key, m, b, w.pace);
    } catch (e) { console.warn('walk style unavailable', e); S.walkStyle = 'natural'; save(); }
    if (note) note.textContent = was;
  };
  if (S.walkStyle && S.walkStyle !== 'natural') loadWalkStyle(S.walkStyle);
  const R = buildScene({ meta: mmeta, bin: mbin });
  const bodyFX = new BodyFX(THREE, R.geo, sk, R.inverses, limits);
  bodyFX.install(R.skin);
  setupInput(R.renderer.domElement);
  const MODELS = {
    balanced: { short: 'Balanced', label: 'Balanced - INT8 pose transformer' },
  };
  for (const [k, m] of Object.entries(MODELS)) Object.assign(m, { bytes: models.models[k]?.bytes || 0, available: !!models.models[k] });
  if (!MODELS[S.model]?.available) { S.model = Object.keys(MODELS).find((k) => MODELS[k].available); save(); } // e.g. the removed Light model

  // worker
  const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
  let ready = false, nextId = 1;
  const waiting = new Map();
  const J = sk.names.length;
  worker.onmessage = (e) => {
    const m = e.data;
    if (m.type === 'progress') { $('#bar').style.width = `${(100 * m.done / m.total).toFixed(1)}%`; status(`Downloading ${MODELS[S.model].short} model: ${(m.done / 1e6).toFixed(0)} / ${(m.total / 1e6).toFixed(0)} MB (cached after the first visit)`); }
    else if (m.type === 'status') status(m.text);
    else if (m.type === 'ready') { ready = true; $('#loading').hidden = true; $('#hud-model').textContent = MODELS[S.model].label; $('#hud-threads').textContent = `${m.threads} thread${m.threads > 1 ? 's' : ''}`; }
    else if (m.type === 'result') {
      const w = waiting.get(m.id); waiting.delete(m.id);
      const frames = [], contacts = [];
      for (let i = 0; i < m.count; i++) { frames.push(m.frames.slice(i * J * 7, (i + 1) * J * 7)); contacts.push(Array.from(m.contacts.slice(i * 4, i * 4 + 4))); }
      $('#hud-net').textContent = `${m.ms.network.toFixed(0)} ms`; PERF.net += m.ms.total; PERF.gens++;
      const done = () => w.resolve({ frames, contacts, tokens: m.tokens });
      if (window.SOMA_EXTRA_MS) setTimeout(done, window.SOMA_EXTRA_MS); else done(); // test hook: emulate a slow phone
    } else if (m.type === 'error') {
      if (m.id && waiting.has(m.id)) { waiting.get(m.id).reject(new Error(m.error)); waiting.delete(m.id); }
      else { status(`Could not start the model: ${m.error}`); $('#loading').hidden = false; }
    }
  };
  const loadModel = (key) => {
    ready = false; $('#loading').hidden = false; $('#bar').style.width = '0%';
    $('#start').hidden = true; $('#go').hidden = true;
    worker.postMessage({ type: 'load', base: BASE + 'models/', model: key, info: models.models[key], skeleton: sk, jointLimits: lim,
      threads: Math.min(4, navigator.hardwareConcurrency || 2) });
  };
  const generate = (req) => new Promise((resolve, reject) => {
    const id = nextId++; waiting.set(id, { resolve, reject });
    worker.postMessage({ type: 'generate', id, ...req });
  });
  const ctrl = new Controller({ sk, lib, clips, bank, limits, settings: () => S, generate, actions: wmeta.actions });
  buildPanel(MODELS, (k) => loadModel(k), loadWalkStyle);

  // start screen: pick a model, then download
  const startRow = $('#start-models');
  startRow.innerHTML = Object.entries(MODELS).filter(([, m]) => m.available).map(([k, m]) => `<button class="seg${k === S.model ? ' on' : ''}" data-start="${k}">${m.short}<small>${(m.bytes / 1e6).toFixed(0)} MB</small></button>`).join('');
  startRow.addEventListener('click', (e) => { const b = e.target.closest('[data-start]'); if (!b) return; S.model = b.dataset.start; save(); startRow.querySelectorAll('button').forEach((x) => x.classList.toggle('on', x === b)); });
  $('#start').hidden = false;
  status(`The real SOMA network runs on this device. First download: about ${(models.models[S.model].bytes / 1e6).toFixed(0)} MB.`);
  $('#go').addEventListener('click', () => loadModel(S.model));

  $('#gear').addEventListener('click', () => { $('#panel').hidden = !$('#panel').hidden; });
  $('#close').addEventListener('click', () => { $('#panel').hidden = true; });
  if (!IS_TOUCH) document.body.classList.add('desktop');

  // per-frame cost counters (main thread) and generation cost (worker); read by the performance governor and tests
  const governor = new QualityGovernor(R, R.basePixelRatio);
  const PERF = window.SOMA_PERF = { frames: 0, frameMs: 0, update: 0, post: 0, render: 0, net: 0, gens: 0 };
  // loop
  const camT = new THREE.Vector3(0, 0.95, 0); let last = performance.now(), fpsAcc = 0, fpsN = 0;
  const ray = new THREE.Raycaster(), tmp = new THREE.Vector3();
  function frame(now) {
    // test hook: with SOMA_MANUAL set, the game only advances through SOMA_DEBUG.step(dt) (frame-exact captures)
    if (window.SOMA_MANUAL && !frame.stepping) { last = now; requestAnimationFrame(frame); return; }
    const realDt = (now - last) / 1000;
    const dt = frame.stepping ? frame.stepping : Math.min(0.1, Math.max(0, realDt)); last = now; // stays real-time down to 10 fps
    if (!frame.stepping && ready) governor.update(realDt, S.autoQuality !== false);
    let pose;
    const tu = performance.now();
    if (ready && !window.SOMA_FREEZE) pose = ctrl.update(command(dt), dt);
    else if (ready) pose = ctrl.display;
    else pose = ctrl.display || ctrl.buffer[3];
    const tp = performance.now(); PERF.update += tp - tu;
    if (S.bodyFX !== false) pose = bodyFX.posePass(pose, dt);
    const world = M.fk(pose, sk.parents);
    if (S.bodyFX !== false) bodyFX.update(pose, world, ready ? ctrl.sampleContacts(ctrl.time) : null, Math.hypot(ctrl.simVel[0], ctrl.simVel[2]), S.sprintSpeed, dt);
    else bodyFX.off();
    for (let j = 0; j < J; j++) { R.bones[j].position.set(world[j * 7], world[j * 7 + 1], world[j * 7 + 2]); R.bones[j].quaternion.set(world[j * 7 + 3], world[j * 7 + 4], world[j * 7 + 5], world[j * 7 + 6]); }
    const hx = world[0], hz = world[2], k = 1 - Math.exp(-dt * 6);
    camT.x += (hx - camT.x) * k; camT.z += (hz - camT.z) * k;
    R.camera.position.set(camT.x + Math.sin(input.camYaw) * input.camDist, 1.45 + input.camDist * 0.12, camT.z + Math.cos(input.camYaw) * input.camDist);
    R.camera.lookAt(camT.x, 0.95, camT.z);
    R.sun.position.set(hx + 3.5, 7.5, hz + 2.5); R.sun.target.position.set(hx, 0, hz);
    // third-person occlusion: blocks between the camera and the character fade out
    ray.set(R.camera.position, tmp.set(hx, world[1], hz).sub(R.camera.position).normalize());
    ray.far = R.camera.position.distanceTo(tmp.set(hx, world[1], hz));
    const hits = new Set(ray.intersectObjects(R.landmarks, false).map((h) => h.object));
    for (const b of R.landmarks) {
      const goal = hits.has(b) ? 0.18 : 1;
      b.material.opacity += (goal - b.material.opacity) * Math.min(1, dt * 10);
      b.material.depthWrite = b.material.opacity > 0.95; b.castShadow = b.material.opacity > 0.5;
    }
    for (const ev of ctrl.events.splice(0)) if (ev.type === 'sit') {
      R.chair.userData.build(ev.seat); R.chair.position.set(ev.pos[0] - Math.sin(ev.yaw) * 0.08, 0, ev.pos[2] - Math.cos(ev.yaw) * 0.08); R.chair.rotation.y = ev.yaw; R.chair.visible = true;
    }
    const tr = performance.now(); PERF.post += tr - tp;
    R.renderer.render(R.scene, R.camera);
    PERF.render += performance.now() - tr; PERF.frames++; PERF.frameMs += dt * 1000;
    fpsAcc += dt; fpsN++;
    if (fpsAcc > 0.5) {
      $('#hud-fps').textContent = `${(fpsN / fpsAcc).toFixed(0)} fps${governor.level ? ' · lite ' + governor.level : ''}`;
      $('#hud-speed').textContent = `${Math.hypot(ctrl.simVel[0], ctrl.simVel[2]).toFixed(1)} m/s`;
      $('#hud-gens').textContent = `${ctrl.stats.generations}`;
      $('#hud-lat').textContent = `${(ctrl.latency * 1000).toFixed(0)} ms`;
      if (ctrl.stats.failures) $('#hud-err').textContent = ctrl.stats.error.slice(0, 80);
      fpsAcc = 0; fpsN = 0;
    }
    window.SOMA_STATE = { ready, generations: ctrl.stats.generations, late: ctrl.stats.late, latency: ctrl.latency, speed: Math.hypot(ctrl.simVel[0], ctrl.simVel[2]), hips: [hx, world[1], hz], failures: ctrl.stats.failures, error: ctrl.stats.error };
    window.SOMA_LAST = { pose, world }; // test hook: last displayed pose
    if (!frame.stepping) requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
  const step = (dt = 1 / 30) => { frame.stepping = dt; try { frame(performance.now()); } finally { frame.stepping = 0; } };
  window.SOMA_DEBUG = { input, ctrl, S, loadModel, bodyFX, fightPoses, clips, sk, M, loadWalkStyle, step, R, governor };
}
main().catch((e) => { $('#status').textContent = `Error: ${e.message}`; console.error(e); });
