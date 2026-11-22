// Trained animal motion: a mode-adaptive neural network (gating network + 4 expert MLPs) trained on
// 41 minutes of dog motion capture (tools/animal/train_mann.py), evaluated every frame in plain JS.
// The network predicts the next pose, the root motion, the future path and paw contacts from the
// current pose and a 2 s path (1 s of history, 1 s of blended prediction and player input).
// Other species reuse the dog-trained motion, re-proportioned to their bodies and time-scaled by size
// (larger animals step more slowly, as in nature). Bodies are smooth primitives placed on the joints.
import { SPECIES } from './animals.js';

const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
const lerp = (a, b, t) => a + (b - a) * t;
const wrapA = (a) => Math.atan2(Math.sin(a), Math.cos(a));

function half2float(buf, offset, n) {
  const h = new Uint16Array(buf, offset, n), out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const x = h[i], s = x & 0x8000 ? -1 : 1, e = (x >> 10) & 0x1f, m = x & 0x3ff;
    out[i] = e === 0 ? s * m * 5.960464477539063e-8 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * Math.pow(2, e - 15);
  }
  return out;
}

// WebAssembly SIMD kernel for the expert-blended layers (build/tools/mix_simd.wat, built by build_simd.cjs):
// the same sums 4 floats per instruction, about 3x faster than plain JS. Browsers without SIMD use the JS path.
const SIMD_MIX = 'AGFzbQEAAAABDAFgCH9/f39/f39/AAMCAQAFAwEAAQcNAgNtZW0CAANtaXgAAArmAgHjAgQDfwF9AXsDfyAGQQJ0IQ5BACEKAkADQCAKIA5PDQEgBCAKav0MAAAAAAAAAAAAAAAAAAAAAP0LBAAgCkEQaiEKDAALC0EAIQgCQANAIAggB08NASACIAhBAnRqKgIAIQsgC/0TIQwgASAIIA5saiEPQQAhCgJAA0AgCiAOTw0BIAQgCmogBCAKav0ABAAgDCAPIApq/QAEAP3mAf3kAf0LBAAgCkEQaiEKDAALCyAAIAggBWwgDmxqIQ1BACEJAkADQCAJIAVPDQEgCyADIAlBAnRqKgIAlP0TIQxBACEKAkADQCAKIA5PDQEgBCAKaiAEIApq/QAEACAMIA0gCmr9AAQA/eYB/eQB/QsEACAEIApBEGpqIAQgCkEQamr9AAQAIAwgDSAKQRBqav0ABAD95gH95AH9CwQAIApBIGohCgwACwsgDSAOaiENIAlBAWohCQwACwsgCEEBaiEIDAALCws=';
function simdKernel(net) {
  try {
    if (typeof WebAssembly === 'undefined' || globalThis.SOMA_NO_SIMD) return null;
    const bytes = Uint8Array.from(atob(SIMD_MIX), (c) => c.charCodeAt(0));
    if (!WebAssembly.validate(bytes)) return null;
    const inst = new WebAssembly.Instance(new WebAssembly.Module(bytes), {}), mem = inst.exports.mem;
    const pad = (n) => Math.ceil(n / 8) * 8, K = net.K;
    let off = 0; const alloc = (floats) => { const o = off; off += Math.ceil(floats / 4) * 16; return o; };
    const layers = net.layers.map((L) => ({ ...L, noP: pad(L.o), w: alloc(K * L.i * pad(L.o)), b: alloc(K * pad(L.o)) }));
    const maxN = pad(Math.max(net.nx, ...net.layers.map((L) => L.o))), al = alloc(K), a = alloc(maxN), b = alloc(maxN);
    mem.grow(Math.max(0, Math.ceil(off / 65536) - mem.buffer.byteLength / 65536));
    const F = new Float32Array(mem.buffer);
    for (const L of layers) for (let k = 0; k < K; k++) {
      for (let i = 0; i < L.i; i++) F.set(L.W.subarray((k * L.i + i) * L.o, (k * L.i + i + 1) * L.o), (L.w >> 2) + (k * L.i + i) * L.noP);
      F.set(L.B.subarray(k * L.o, (k + 1) * L.o), (L.b >> 2) + k * L.noP);
    }
    return { mix: inst.exports.mix, F, layers, al, a, b };
  } catch { return null; }
}

// Inputs are kept within 8 standard deviations of the training data (5 also capped sprint speed): harsh input (reversals, direction changes every
// few tenths of a second) otherwise fed the network states it never saw and its pose/trajectory feedback diverged.
const Z_MAX = 8, clampZ = (z) => (z > Z_MAX ? Z_MAX : z < -Z_MAX ? -Z_MAX : z);

export class DogNet {
  constructor(meta, buffer) {
    this.meta = meta; const T = {};
    for (const t of meta.tensors) T[t.name] = { data: half2float(buffer, t.offset, t.shape.reduce((a, b) => a * b, 1)), shape: t.shape };
    this.T = T; this.K = meta.experts; this.H = meta.hidden;
    this.gate = [['gate.0.weight', 'gate.0.bias'], ['gate.2.weight', 'gate.2.bias'], ['gate.4.weight', 'gate.4.bias']].map(([w, b]) => ({ w: T[w].data, b: T[b].data, o: T[w].shape[0], i: T[w].shape[1] }));
    this.layers = [0, 1, 2].map((l) => ({ W: T[`W.${l}`].data, B: T[`B.${l}`].data, i: T[`W.${l}`].shape[1], o: T[`W.${l}`].shape[2] }));
    this.xm = T.x_mean.data; this.xs = T.x_std.data; this.ym = T.y_mean.data; this.ys = T.y_std.data; this.gm = T.g_mean.data; this.gs = T.g_std.data;
    this.nx = this.xm.length; this.ny = this.ym.length; this.ng = this.gm.length;
    this.bufA = new Float32Array(Math.max(this.nx, this.H, this.ny)); this.bufB = new Float32Array(Math.max(this.nx, this.H, this.ny));
    this.alpha = new Float32Array(this.K);
    this.simd = simdKernel(this);
  }
  forward(g, x) {
    // gating network -> expert blend
    let h = new Float32Array(this.ng); for (let i = 0; i < this.ng; i++) h[i] = clampZ((g[i] - this.gm[i]) / this.gs[i]);
    this.gate.forEach((L, li) => {
      const o = new Float32Array(L.o);
      for (let r = 0; r < L.o; r++) { let s = L.b[r]; const off = r * L.i; for (let c = 0; c < L.i; c++) s += L.w[off + c] * h[c]; o[r] = li < 2 ? (s > 0 ? s : Math.exp(s) - 1) : s; }
      h = o;
    });
    let mx = -Infinity; for (const v of h) mx = Math.max(mx, v);
    let sum = 0; for (let k = 0; k < this.K; k++) { this.alpha[k] = Math.exp(h[k] - mx); sum += this.alpha[k]; }
    for (let k = 0; k < this.K; k++) this.alpha[k] /= sum;
    // motion network with expert-blended layers
    if (this.simd) {
      const S = this.simd, F = S.F;
      F.set(this.alpha, S.al >> 2);
      let ip = S.a, op = S.b;
      for (let i = 0; i < this.nx; i++) F[(ip >> 2) + i] = clampZ((x[i] - this.xm[i]) / this.xs[i]);
      S.layers.forEach((L, li) => {
        S.mix(L.w, L.b, S.al, ip, op, L.i, L.noP, this.K);
        if (li < 2) for (let o = op >> 2, e = o + L.o; o < e; o++) { const v = F[o]; F[o] = v > 0 ? v : Math.exp(v) - 1; }
        const t = ip; ip = op; op = t;
      });
      const y = new Float32Array(this.ny), base = ip >> 2;
      for (let i = 0; i < this.ny; i++) y[i] = F[base + i] * this.ys[i] + this.ym[i];
      return y;
    }
    let inp = this.bufA, out = this.bufB;
    for (let i = 0; i < this.nx; i++) inp[i] = clampZ((x[i] - this.xm[i]) / this.xs[i]);
    this.layers.forEach((L, li) => {
      const { W, B, i: ni, o: no } = L;
      out.fill(0, 0, no);
      for (let k = 0; k < this.K; k++) {
        const a = this.alpha[k], base = k * ni * no;
        for (let o = 0; o < no; o++) out[o] += a * B[k * no + o];
        for (let i = 0; i < ni; i++) {
          const xi = a * inp[i]; if (xi === 0) continue;
          const row = base + i * no;
          for (let o = 0; o < no; o++) out[o] += xi * W[row + o];
        }
      }
      if (li < 2) for (let o = 0; o < no; o++) { const v = out[o]; out[o] = v > 0 ? v : Math.exp(v) - 1; }
      const t = inp; inp = out; out = t;
    });
    const y = new Float32Array(this.ny);
    for (let i = 0; i < this.ny; i++) y[i] = inp[i] * this.ys[i] + this.ym[i];
    return y;
  }
}

// ---------------------------------------------------------------- controller + body
// How strongly the player's input steers the network: the path's spring half-life (s), how fast the path's heading
// follows the input (1/s), and the player path's weight in the network's nearest future sample. Measured
// (build/test/animal_sim.html), before (0.22 s, 7/s, 0.1) -> now: 90 deg turns dog 0.7-1.4 -> 0.48-0.62 s, horse
// 1.6-2.5 -> 0.83-0.93 s, elephant 1.5-2.4 -> 1.05-1.23 s; stops dog 1.17 -> 0.57 s, horse 1.8 -> 1.08 s (the elephant
// did not stop within 2 s). Stable under ?stress=1 for every species thanks to the input clamp (Z_MAX).
const BOUND_LAG = [4, 2]; // 60 Hz steps: front paws land ~67 ms apart, hind feet ~33 ms (real rabbits: staggered front, near-paired hind)
export const RESPONSE = { pathHalfLife: 0.08, yawRate: 20, nearWeight: 0.7 };
const DOG_SPEED = { walk: 0.9, run: 2.0, sprint: 3.6, crouch: 0.6 }; // m/s in the dog's own scale
const DOG_HIP = 0.469;

export class NeuralAnimal {
  constructor(THREE, scene, key, net, model = null) {
    this.THREE = THREE; this.key = key; this.sp = SPECIES[key]; this.net = net; this.model = model; const m = net.meta;
    this.names = m.points; this.parents = m.parents; this.rest = m.rest_lengths; this.P = this.names.length;
    this.ix = Object.fromEntries(this.names.map((n, i) => [n, i]));
    const sp = this.sp;
    this.scale = (model ? model.hips : sp.hip) / DOG_HIP; this.size = model ? model.hips * 1.05 : Math.max(sp.hip, sp.sh); this.timeScale = 1 / Math.sqrt(this.scale);
    this.style = model?.cfg.style || {}; this.timeScale *= this.style.cadence || 1; // per-species step rate on top of size
    // per-bone length factors: legs, body, neck/head, tail follow the species proportions
    const dogFront = 0.152 + 0.178 + 0.072, dogHind = 0.16 + 0.18 + 0.108;
    const legF = (sp.front[0] + sp.front[1] + sp.front[2]) / (this.scale * dogFront), legH = (sp.hind[0] + sp.hind[1] + sp.hind[2]) / (this.scale * dogHind);
    const bodyK = sp.len / (this.scale * 0.42), neckK = Math.max(0.35, sp.neck[0] / (this.scale * 0.225)), headK = Math.max(0.5, (sp.head[0] * 2 + sp.head[1]) / (this.scale * 0.31));
    const tailK = Math.max(0.1, sp.tail[0] / (this.scale * 0.31));
    this.boneK = this.names.map((n) => {
      if (/Hand|ForeArm|Arm$/.test(n) && !/Shoulder/.test(n)) return legF;
      if (/Leg|Foot/.test(n)) return legH;
      if (/Spine1|Shoulder|UpLeg/.test(n)) return bodyK;
      if (/Neck/.test(n)) return neckK;
      if (/Head/.test(n)) return headK;
      if (/Tail/.test(n)) return tailK;
      return 1;
    });
    if (model) { // a skinned model: bone lengths measured on it, so the network's paws land under the model's
      const L = model.legLengths(), reach = model.cfg.style?.legReach || 1, kF = reach * L.front / (0.402 * this.scale), kH = reach * L.hind / (0.448 * this.scale);
      this.boneK = this.names.map((n) => (/Arm|Hand/.test(n) ? kF : /Leg|Foot/.test(n) && !/UpLeg/.test(n) ? kH : 1));
      // longer (shorter) front legs than the dog's: pitch the body up (down) so all four paws stand on the ground
      const lp0 = this.localPose(m.start_pose), avg = (a, b, k) => (lp0[this.ix[a] * 3 + k] + lp0[this.ix[b] * 3 + k]) / 2;
      const yf = avg('LeftHandEnd', 'RightHandEnd', 1), zf = avg('LeftHandEnd', 'RightHandEnd', 2), yh = avg('LeftFootEnd', 'RightFootEnd', 1), zh = avg('LeftFootEnd', 'RightFootEnd', 2);
      this.pitch = Math.atan2(yh - yf, zf - zh);
      const sp0 = m.start_pose, Q = (n) => { const i = this.ix[n] * 3; return new THREE.Vector3(sp0[i], sp0[i + 1], sp0[i + 2]); };
      const x = Q('Spine1').sub(Q('Hips')).normalize(), l = Q('LeftUpLeg').sub(Q('RightUpLeg')), z = l.addScaledVector(x, -l.dot(x)).normalize();
      const Pn0 = new THREE.Matrix4().makeBasis(x, z.clone().cross(x), z).transpose();
      const cx = Q('Neck').sub(Q('Spine1')).normalize(), cl = Q('LeftShoulder').sub(Q('RightShoulder')), cz = cl.addScaledVector(cx, -cl.dot(cx)).normalize();
      const Cn0 = new THREE.Matrix4().makeBasis(cx, cz.clone().cross(cx), cz).transpose();
      this.netRest = { tailLocal: Q('Tail1End').sub(Q('Tail')).normalize().applyMatrix4(Pn0),
        neckLocal: Q('Head').sub(Q('Neck')).normalize().applyMatrix4(Cn0), headLocal: Q('HeadEnd').sub(Q('Head')).normalize().applyMatrix4(Cn0) };
      this.Wi = new Float32Array(this.P * 3);
    }
    this.root = new THREE.Group(); scene.add(this.root);
    if (model) this.root.add(model.group); else this.buildBody();
    this.reset(0, 0, 0);
    if (!model) this.dims = this.measureBody(this.world);
  }
  // body blob sizes and anchors from the network's own standing skeleton (yaw 0 faces +Z): the chest reaches just past
  // the shoulder joints and down to the elbows, the rump covers the hip joints; species ratios set width and length
  measureBody(w) {
    const T = this.THREE, sp = this.sp, P = (n) => { const i = this.ix[n] * 3; return new T.Vector3(w[i], w[i + 1], w[i + 2]); };
    const mid = (a, b) => P(a).add(P(b)).multiplyScalar(0.5);
    const hips = P('Hips'), withers = mid('LeftShoulder', 'RightShoulder'), shJ = mid('LeftArm', 'RightArm'), elbow = mid('LeftForeArm', 'RightForeArm');
    const hipJ = mid('LeftUpLeg', 'RightUpLeg'), stifle = mid('LeftLeg', 'RightLeg');
    const { sF, sU, sX } = this.spineFrame(hips, P('LeftShoulder'), P('RightShoulder'), P('LeftUpLeg'), P('RightUpLeg'));
    const deep = sp.heavy || sp.column ? 1.22 : 1, torso = Math.hypot(shJ.x - hipJ.x, shJ.z - hipJ.z), wd = withers.y - elbow.y;
    const top = withers.y + 0.26 * wd, cy = (top - (elbow.y - 0.06 * wd)) / 2 * deep, cx = cy * sp.rC[0] / sp.rC[1], cz = torso * sp.rC[2] / sp.len * 1.1;
    const chest = new T.Vector3(withers.x, top - cy, shJ.z + 0.25 * cz - cz);
    const hTop = hips.y + 0.2 * wd, ry = (hTop - (hipJ.y - 0.6 * (hipJ.y - stifle.y))) / 2 * deep, rx = ry * sp.rH[0] / sp.rH[1], rz = torso * sp.rH[2] / sp.len * 1.1;
    const rump = new T.Vector3(hips.x, hTop - ry, hipJ.z - 0.45 * rz + rz);
    const off = (p, o) => { const d = p.clone().sub(o); return [d.dot(sF), d.dot(sU), d.dot(sX)]; };
    return { cx, cy, cz, rx, ry, rz, chestOff: off(chest, withers), hipOff: off(rump, hips) };
  }
  // the spine's frame: forward from the pelvis to the shoulder blades, up, left
  spineFrame(hips, lS, rS, lH, rH) {
    const withers = lS.clone().add(rS).multiplyScalar(0.5), sF = withers.sub(hips).normalize();
    const across = lS.clone().sub(rS).add(lH.clone().sub(rH)).normalize();
    const sU = sF.clone().cross(across).normalize(); if (sU.y < 0) sU.multiplyScalar(-1);
    return { sF, sU, sX: sU.clone().cross(sF).normalize() };
  }
  // ---------------------------------------------------------------- state
  reset(x, z, yaw) {
    const P = this.P, m = this.net.meta;
    this.pos = [x, z]; this.yaw = yaw; this.rootVel = [0, 0]; this.lieS = 0; this.rearAt = -10;
    // start from a neutral standing pose: the mean pose the network was trained on
    const xm = this.net.xm, poseOff = m.offsets.length * (6 + m.labels.length);
    this.pose = new Float32Array(P * 3); this.vel = new Float32Array(P * 3);
    for (let i = 0; i < P * 3; i++) this.pose[i] = m.start_pose ? m.start_pose[i] : xm[poseOff + i];
    this.nL = m.labels.length;
    this.hist = []; for (let k = 0; k < 61; k++) this.hist.push({ x, z, yaw, vx: 0, vz: 0, lab: new Array(this.nL).fill(0) });
    this.future = m.future.map((o) => ({ x: x + Math.sin(yaw) * 0, z, dx: Math.sin(yaw), dz: Math.cos(yaw), vx: 0, vz: 0 }));
    this.contacts = [1, 1, 1, 1]; this.pawC = this.contacts; this.acc = 0; this.lie = 0; this.jumpAt = -10; this.clock = 0;
    this.poseHist = []; this.contactHist = []; // bounding gait: the right legs replay the left legs a few steps later
    this.ground = 0; this.speed = 0;
    this.locks = [0, 1, 2, 3].map(() => ({ on: false, p: null, w: 0, off: [0, 0, 0] }));
    this.world = this.toWorldPose(); this.prevWorld = this.world;
    this.tailPts = null; this.mode = 'stand'; this.action = null;
  }
  teleport(x, z, yaw) { this.reset(x / this.scale, z / this.scale, yaw); }
  worldXZ() { return [this.pos[0] * this.scale, this.pos[1] * this.scale]; }
  dispose() { this.root.removeFromParent(); }
  setVisible(v) { this.root.visible = v; }
  placeFeet() { /* the network plants the feet itself */ }
  trigger(name) {
    if (this.sp.column && (name === 'jump' || name === 'dodge')) name = 'block'; // elephants do not jump: they rear up instead
    if (name === 'jump' || name === 'jab' || name === 'cross' || name === 'hook' || name === 'uppercut' || name === 'front_kick' || name === 'roundhouse' || name === 'dodge') {
      if (this.clock - this.jumpAt > 1.0) { this.jumpAt = this.clock; this.jumpKind = name === 'dodge' ? 'side' : name === 'jump' ? 'up' : 'pounce'; }
    } else if (name === 'duck' || name === 'sit') { this.lie = this.lie > 0.5 ? 0 : 1; }
    else if (name === 'stand_up') { this.lie = 0; }
    else if (name === 'block') { if (this.lie > 0.5) this.lie = 0; else if (this.clock - this.rearAt > 1.8 && this.clock - this.jumpAt > 1.6) this.rearAt = this.clock + (this.speed > 1.1 * this.scale ? 0.45 : 0); } // from a run: pull up first, then rear; never straight out of a leap
  }
  // ---------------------------------------------------------------- per-frame update
  update(cmd, dt) {
    if (dt <= 0) return;
    if (cmd.action) this.trigger(cmd.action);
    this.cmd = cmd;
    this.acc += dt * this.timeScale;
    const step = 1 / 60; let n = 0;
    while (this.acc >= step && n < 4) { this.prevWorld = this.world; this.step(step); this.acc -= step; n++; }
    if (n === 4) this.acc = 0;
    const f = clamp(this.acc / step, 0, 1);
    this.render(this.prevWorld, this.world, f, dt);
  }
  desired() {
    const c = this.cmd || { move: [0, 0], gait: 'run' }, mag = Math.min(1, Math.hypot(c.move[0], c.move[1]));
    const rearing = this.clock - this.rearAt > -0.6 && this.clock - this.rearAt < 1.5; // animals stand still to rear up
    if (this.lie > 0.5 || mag < 0.05 || rearing) return { v: [0, 0], yaw: this.yaw, moving: false };
    const gait = this.style.maxGait === 'run' && c.gait === 'sprint' ? 'run' : c.gait; // e.g. elephants never gallop
    const sp = (DOG_SPEED[gait] ?? DOG_SPEED.run) * (this.style.speed || 1) * (gait === 'walk' || gait === 'crouch' ? 1 : mag > 0.6 ? 1 : 0.6);
    const yaw = Math.atan2(c.move[0], c.move[1]);
    return { v: [Math.sin(yaw) * sp, Math.cos(yaw) * sp], yaw, moving: true };
  }
  step(dt) {
    const net = this.net, m = net.meta, P = this.P, offs = m.offsets, fut = m.future;
    this.clock += dt;
    const des = this.desired();
    // ---- jump / pounce schedule: labels in the near future, plus a velocity burst for a pounce
    const sinceJump = this.clock - this.jumpAt, jumping = sinceJump >= 0 && sinceJump < 0.9, sinceRear = this.clock - this.rearAt;
    const jumpLabel = (tAhead) => { const s = sinceJump + tAhead; return s > 0.05 && s < 0.75 ? 1 : 0; };
    const rearLabel = (tAhead) => { const s = sinceRear + tAhead; return s > 0.05 && s < 1.6 ? 1 : 0; };
    // labels for a path sample tAhead seconds from now, in the order the network was trained with
    const labelsAt = (tAhead) => m.labels.map((n) => (n === 'lie' ? (tAhead <= 0 ? this.lieS : lerp(this.lieS, this.lie, Math.min(1, tAhead / 0.67))) : n === 'jump' ? jumpLabel(tAhead) : n === 'rear' ? rearLabel(tAhead) : 0));
    let dv = des.v.slice(), dyaw = des.yaw;
    if (jumping && this.jumpKind !== 'up') {
      const k = Math.sin(Math.PI * clamp(sinceJump / 0.8, 0, 1));
      if (this.jumpKind === 'pounce') { dv = [dv[0] + Math.sin(this.yaw) * 2.6 * k, dv[1] + Math.cos(this.yaw) * 2.6 * k]; }
      else { dv = [dv[0] + Math.cos(this.yaw) * 2.0 * k, dv[1] - Math.sin(this.yaw) * 2.0 * k]; }
    }
    this.lieS += (this.lie - this.lieS) * (1 - Math.exp(-dt * 2.5));
    // ---- desired path (critically damped spring from the current root velocity)
    const want = [], hl = RESPONSE.pathHalfLife; // player's path: critically damped spring toward the command
    { let px = this.pos[0], pz = this.pos[1], vx = this.rootVel[0], vz = this.rootVel[1], ax = 0, az = 0, y = this.yaw;
      const om = 2 / hl;
      for (let k = 1; k <= 60; k++) {
        const e = Math.exp(-om * dt);
        const jx = vx - dv[0], jz = vz - dv[1], j1x = ax + jx * om, j1z = az + jz * om;
        vx = e * (jx + j1x * dt) + dv[0]; vz = e * (jz + j1z * dt) + dv[1]; ax = e * (ax - j1x * om * dt); az = e * (az - j1z * om * dt);
        px += vx * dt; pz += vz * dt;
        const tgtYaw = Math.hypot(vx, vz) > 0.25 ? Math.atan2(vx, vz) : (des.moving ? dyaw : y);
        y = y + wrapA(tgtYaw - y) * (1 - Math.exp(-dt * RESPONSE.yawRate));
        if (k % 10 === 0) want.push({ x: px, z: pz, yaw: y, vx, vz });
      }
    }
    // blend the network's own future with the player's path (more player toward the end)
    const futW = fut.map((o, i) => {
      const p = this.future[i], w = want[i], b = Math.min(0.97, RESPONSE.nearWeight + (0.95 - RESPONSE.nearWeight) * Math.pow(o / 60, 0.6));
      const dx = lerp(p.dx, Math.sin(w.yaw), b), dz = lerp(p.dz, Math.cos(w.yaw), b), dn = Math.hypot(dx, dz) || 1;
      return { x: lerp(p.x, w.x, b), z: lerp(p.z, w.z, b), dx: dx / dn, dz: dz / dn, vx: lerp(p.vx, w.vx, b), vz: lerp(p.vz, w.vz, b) };
    });
    // ---- network input (root frame: facing -> +Z)
    const c = Math.cos(this.yaw), s = Math.sin(this.yaw);
    const toR = (x, z) => [c * x - s * z, s * x + c * z];
    const X = new Float32Array(net.nx); let q = 0; const TS = 6 + this.nL;
    for (const o of offs) {
      let sx, sz, fx, fz, vx, vz, lab;
      if (o < 0) { const h = this.hist[this.hist.length - 1 + o]; sx = h.x; sz = h.z; fx = Math.sin(h.yaw); fz = Math.cos(h.yaw); vx = h.vx; vz = h.vz; lab = h.lab; }
      else if (o === 0) { sx = this.pos[0]; sz = this.pos[1]; fx = s; fz = c; vx = this.rootVel[0]; vz = this.rootVel[1]; lab = labelsAt(0); }
      else { const f = futW[fut.indexOf(o)]; sx = f.x; sz = f.z; fx = f.dx; fz = f.dz; vx = f.vx; vz = f.vz; lab = labelsAt(o / 60); }
      const [px, pz] = toR(sx - this.pos[0], sz - this.pos[1]), [dx, dz] = toR(fx, fz), [rvx, rvz] = toR(vx, vz);
      X[q++] = px; X[q++] = pz; X[q++] = dx; X[q++] = dz; X[q++] = rvx; X[q++] = rvz;
      for (const v of lab) X[q++] = v;
    }
    X.set(this.pose, q); q += P * 3; X.set(this.vel, q);
    // gating input: paw velocities, root velocity, path velocity at +0.5 s and +1 s, labels now and in 0.5 s
    const G = new Float32Array(net.ng); let g = 0;
    for (const pn of m.paws) { const j = this.ix[pn]; G[g++] = this.vel[j * 3]; G[g++] = this.vel[j * 3 + 1]; G[g++] = this.vel[j * 3 + 2]; }
    const [rvx, rvz] = toR(this.rootVel[0], this.rootVel[1]); G[g++] = rvx; G[g++] = rvz;
    const tI = (o) => offs.indexOf(o) * TS;
    G[g++] = X[tI(30) + 4]; G[g++] = X[tI(30) + 5]; G[g++] = X[tI(60) + 4]; G[g++] = X[tI(60) + 5];
    for (let l = 0; l < this.nL; l++) G[g++] = X[tI(0) + 6 + l];
    for (let l = 0; l < this.nL; l++) G[g++] = X[tI(30) + 6 + l];
    const Y = net.forward(G, X);
    for (let i = 0; i < Y.length; i++) if (!Number.isFinite(Y[i])) { this.failures = (this.failures || 0) + 1; this.reset(this.pos[0], this.pos[1], this.yaw); return; }
    // ---- outputs
    let r = 0; const nP = new Float32Array(P * 3), nV = new Float32Array(P * 3);
    for (let i = 0; i < P * 3; i++) nP[i] = Y[r++];
    for (let i = 0; i < P * 3; i++) nV[i] = Y[r++];
    let vRx = Y[r++], vRz = Y[r++], yawRate = clamp(Y[r++], -6, 6);
    const vm = Math.hypot(vRx, vRz); if (vm > 7) { vRx *= 7 / vm; vRz *= 7 / vm; }
    // root motion (root frame of t) -> world; the yaw follows the predicted turn rate
    const wvx = c * vRx + s * vRz, wvz = -s * vRx + c * vRz;
    const oldPos = this.pos.slice(), oldYaw = this.yaw;
    this.pos[0] += wvx * dt; this.pos[1] += wvz * dt; this.yaw = wrapA(this.yaw + yawRate * dt);
    this.rootVel = [wvx, wvz];
    const c1 = Math.cos(this.yaw), s1 = Math.sin(this.yaw);
    const toW1 = (x, z) => [c1 * x + s1 * z, -s1 * x + c1 * z];
    this.future = fut.map((o) => {
      const px = Y[r++], pz = Y[r++], fx = Y[r++], fz = Y[r++], vx = Y[r++], vz = Y[r++];
      const [wx, wz] = toW1(px, pz), [dx, dz] = toW1(fx, fz), [wvx2, wvz2] = toW1(vx, vz), dn = Math.hypot(dx, dz) || 1;
      return { x: this.pos[0] + wx, z: this.pos[1] + wz, dx: dx / dn, dz: dz / dn, vx: wvx2, vz: wvz2 };
    });
    this.contacts = [Y[r++], Y[r++], Y[r++], Y[r++]];
    if (this.style.bound) { // hopping: the right legs land BOUND_LAG steps after the left ones (front pair more staggered)
      this.contactHist.push(this.contacts.slice()); if (this.contactHist.length > 8) this.contactHist.shift();
      const ago = (n) => this.contactHist[Math.max(0, this.contactHist.length - 1 - n)];
      this.pawC = [this.contacts[0], ago(BOUND_LAG[0])[0], this.contacts[2], ago(BOUND_LAG[1])[2]];
    } else this.pawC = this.contacts;
    // pose: average of the predicted pose and the previous pose advanced by the predicted velocity
    const prevW = this.toWorldPoseRaw(this.pose, oldPos, oldYaw);
    for (let j = 0; j < P; j++) {
      const wx = prevW[j * 3] + (c1 * nV[j * 3] + s1 * nV[j * 3 + 2]) * dt, wy = prevW[j * 3 + 1] + nV[j * 3 + 1] * dt, wz = prevW[j * 3 + 2] + (-s1 * nV[j * 3] + c1 * nV[j * 3 + 2]) * dt;
      const dx = wx - this.pos[0], dz = wz - this.pos[1], lx = c1 * dx - s1 * dz, lz = s1 * dx + c1 * dz;
      nP[j * 3] = lerp(lx, nP[j * 3], 0.5); nP[j * 3 + 1] = lerp(wy, nP[j * 3 + 1], 0.5); nP[j * 3 + 2] = lerp(lz, nP[j * 3 + 2], 0.5);
    }
    // keep the dog's bone lengths
    for (let j = 1; j < P; j++) {
      const p = this.parents[j], L = this.rest[j];
      let dx = nP[j * 3] - nP[p * 3], dy = nP[j * 3 + 1] - nP[p * 3 + 1], dz = nP[j * 3 + 2] - nP[p * 3 + 2]; const d = Math.hypot(dx, dy, dz);
      if (L < 1e-6) { nP[j * 3] = nP[p * 3]; nP[j * 3 + 1] = nP[p * 3 + 1]; nP[j * 3 + 2] = nP[p * 3 + 2]; continue; }
      if (d < 1e-6) continue;
      nP[j * 3] = nP[p * 3] + dx / d * L; nP[j * 3 + 1] = nP[p * 3 + 1] + dy / d * L; nP[j * 3 + 2] = nP[p * 3 + 2] + dz / d * L;
    }
    this.pose = nP; this.vel = nV;
    if (this.style.bound) { this.poseHist.push(nP.slice()); if (this.poseHist.length > 8) this.poseHist.shift(); }
    this.hist.push({ x: this.pos[0], z: this.pos[1], yaw: this.yaw, vx: wvx, vz: wvz, lab: labelsAt(0) });
    if (this.hist.length > 61) this.hist.shift();
    this.speed = Math.hypot(wvx, wvz) * this.timeScale * this.scale; // world m/s (dog units x size, Froude-slowed time)
    this.world = this.plantPaws(this.toWorldPose(), dt / this.timeScale);
  }
  toWorldPoseRaw(pose, pos, yaw) {
    const c = Math.cos(yaw), s = Math.sin(yaw), out = new Float32Array(this.P * 3);
    for (let j = 0; j < this.P; j++) { const x = pose[j * 3], z = pose[j * 3 + 2]; out[j * 3] = pos[0] + c * x + s * z; out[j * 3 + 1] = pose[j * 3 + 1]; out[j * 3 + 2] = pos[1] - s * x + c * z; }
    return out;
  }
  // species-proportioned world joints: scale about the root, re-length bones by body part, plant on the ground
  // root-frame pose re-lengthed bone by bone, pitched about the hips so a species' front and hind paws stand level
  localPose(pose) {
    const P = this.P, S = this.scale, lp = new Float32Array(P * 3);
    lp[0] = pose[0] * S; lp[1] = pose[1] * S; lp[2] = pose[2] * S;
    for (let j = 1; j < P; j++) {
      const p = this.parents[j], dx = pose[j * 3] - pose[p * 3], dy = pose[j * 3 + 1] - pose[p * 3 + 1], dz = pose[j * 3 + 2] - pose[p * 3 + 2], k = S * this.boneK[j];
      lp[j * 3] = lp[p * 3] + dx * k; lp[j * 3 + 1] = lp[p * 3 + 1] + dy * k; lp[j * 3 + 2] = lp[p * 3 + 2] + dz * k;
    }
    if (this.style.bound) { // hopping gait: each pair of legs moves together, the right leg BOUND_LAG steps behind the left
      [['Arm', ['ForeArm', 'Hand', 'HandEnd'], BOUND_LAG[0]], ['UpLeg', ['Leg', 'Foot', 'FootEnd'], BOUND_LAG[1]]].forEach(([root, chain, lag]) => {
        const H = this.poseHist || [], src = H.length ? H[Math.max(0, H.length - 1 - lag)] : pose, lr = this.ix['Left' + root], rr = this.ix['Right' + root] * 3;
        let prev = lr;
        for (const n of chain) {
          const l = this.ix['Left' + n], r = this.ix['Right' + n] * 3, pl = this.parents[l], k = S * this.boneK[l];
          // the delayed left segment, mirrored across the body (x), hung from the right joint above it
          const dx = (src[l * 3] - src[pl * 3]) * k, dy = (src[l * 3 + 1] - src[pl * 3 + 1]) * k, dz = (src[l * 3 + 2] - src[pl * 3 + 2]) * k;
          const rp = prev === lr ? rr : this.ix['Right' + chain[chain.indexOf(n) - 1]] * 3;
          lp[r] = lp[rp] - dx; lp[r + 1] = lp[rp + 1] + dy; lp[r + 2] = lp[rp + 2] + dz; prev = l;
        }
      });
    }
    if (this.pitch) {
      const c = Math.cos(this.pitch), s = Math.sin(this.pitch), py = lp[1], pz = lp[2];
      for (let j = 0; j < P; j++) { const y = lp[j * 3 + 1] - py, z = lp[j * 3 + 2] - pz; lp[j * 3 + 1] = py + y * c + z * s; lp[j * 3 + 2] = pz - y * s + z * c; }
    }
    return lp;
  }
  toWorldPose() {
    const P = this.P, c = Math.cos(this.yaw), s = Math.sin(this.yaw), S = this.scale, lp = this.localPose(this.pose);
    // ground: the lowest paw that the network says is in contact sits on the floor (smoothed)
    let low = Infinity; this.net.meta.paws.forEach((pn, i) => { const y = lp[this.ix[pn] * 3 + 1]; if (this.pawC[i] > 0.5) low = Math.min(low, y); });
    if (low === Infinity) { let mn = Infinity; for (const pn of this.net.meta.paws) mn = Math.min(mn, lp[this.ix[pn] * 3 + 1]); low = Math.min(mn, this.ground + 0.0); }
    const pawR = this.model ? this.model.pawR : 0.012 * S;
    this.ground += (low - pawR - this.ground) * 0.25;
    const out = new Float32Array(P * 3), wx = this.pos[0] * S, wz = this.pos[1] * S;
    for (let j = 0; j < P; j++) { const x = lp[j * 3], z = lp[j * 3 + 2]; out[j * 3] = wx + c * x + s * z; out[j * 3 + 1] = lp[j * 3 + 1] - this.ground; out[j * 3 + 2] = wz - s * x + c * z; }
    return out;
  }
  // paw planting: a paw the network marks as in contact stays where it landed, sliding only when it falls more than
  // maxLag behind the network's paw or the leg cannot reach it; the leg is re-solved with two-bone IK. On lift-off the
  // remaining offset fades out while the paw is already moving fast, so nothing pops.
  plantPaws(w, dt) {
    const chains = [['LeftArm', 'LeftForeArm', 'LeftHand', 'LeftHandEnd'], ['RightArm', 'RightForeArm', 'RightHand', 'RightHandEnd'],
      ['LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftFootEnd'], ['RightUpLeg', 'RightLeg', 'RightFoot', 'RightFootEnd']];
    const G = (j) => [w[j * 3], w[j * 3 + 1], w[j * 3 + 2]], Set3 = (j, v) => { w[j * 3] = v[0]; w[j * 3 + 1] = v[1]; w[j * 3 + 2] = v[2]; };
    const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]], add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]], len = (a) => Math.hypot(a[0], a[1], a[2]);
    const mul = (a, k) => [a[0] * k, a[1] * k, a[2] * k], maxLag = 0.1 * this.scale;
    chains.forEach((names, k) => {
      const [u, kn, an, pw] = names.map((n) => this.ix[n]), L = this.locks[k], c = this.pawC[k], p = G(pw);
      if (!L.on && c > 0.5) { L.on = true; L.p = p.slice(); }
      else if (L.on && c < 0.35) { L.on = false; L.off = sub(L.p, p); }
      if (L.on) { const dx = p[0] - L.p[0], dz = p[2] - L.p[2], d = Math.hypot(dx, dz); if (d > maxLag) { const f = 1 - maxLag / d; L.p[0] += dx * f; L.p[2] += dz * f; } L.p[1] = p[1]; }
      else L.p = add(p, L.off);
      const goal = L.on ? 1 : 0; L.w += (goal - L.w) * (1 - Math.exp(-dt / (L.on ? 0.02 : 0.07)));
      if (L.w < 1e-3 || !L.p) return;
      const A = G(u), Kp = G(kn), An0 = G(an), l1 = len(sub(Kp, A)), l2 = len(sub(An0, Kp)), reach = (l1 + l2) * 0.995;
      let delta = mul(sub(L.p, p), L.w);
      // keep the wrist/ankle within reach of the shoulder/hip: shorten the correction (and let the plant slide) if needed
      if (len(sub(add(An0, delta), A)) > reach) {
        let lo = 0, hi = 1; for (let it = 0; it < 14; it++) { const m = (lo + hi) / 2; if (len(sub(add(An0, mul(delta, m)), A)) > reach) hi = m; else lo = m; }
        delta = mul(delta, lo); if (L.on && L.w > 0.5) { L.p[0] = p[0] + delta[0] / L.w; L.p[2] = p[2] + delta[2] / L.w; }
      }
      const An = add(An0, delta);
      Set3(pw, add(p, delta)); Set3(an, An);
      // re-solve the knee/elbow: keep segment lengths, bend toward where it was
      const d = sub(An, A), dist = Math.min(Math.max(len(d), Math.abs(l1 - l2) + 1e-4), l1 + l2 - 1e-4);
      const dn = d.map((v) => v / (len(d) || 1)), a = (l1 * l1 - l2 * l2 + dist * dist) / (2 * dist), h = Math.sqrt(Math.max(0, l1 * l1 - a * a));
      const pole = sub(Kp, A), pd = pole[0] * dn[0] + pole[1] * dn[1] + pole[2] * dn[2], pp = sub(pole, dn.map((v) => v * pd)), pn = len(pp) || 1;
      Set3(kn, [A[0] + dn[0] * a + pp[0] / pn * h, A[1] + dn[1] * a + pp[1] / pn * h, A[2] + dn[2] * a + pp[2] / pn * h]);
    });
    return w;
  }
  get target() { const w = this.world, h = this.ix.Spine1; return new this.THREE.Vector3(w[h * 3], this.model ? this.model.hips * 0.9 : Math.max(this.sp.hip, this.sp.sh) * 0.9, w[h * 3 + 2]); }
  // ---------------------------------------------------------------- body
  buildBody() {
    const T = this.THREE, sp = this.sp;
    const mat = (c, r = 0.85, extra = {}) => new T.MeshStandardMaterial({ color: c, roughness: r, ...extra });
    this.mBody = mat(sp.col[0]); this.mBelly = mat(sp.col[1]); this.mDark = mat(sp.col[2], 0.5); this.mEye = mat('#0d0a08', 0.2);
    this.mFoot = sp.hooves ? mat(sp.hooves, 0.6) : sp.socks ? mat(sp.socks) : this.mBody;
    this.sphere = new T.SphereGeometry(1, 22, 16); this.capsule = new T.CapsuleGeometry(1, 1, 8, 14); this.cone = new T.ConeGeometry(1, 1, 14);
    const mesh = (geo, m) => { const o = new T.Mesh(geo, m); o.castShadow = true; o.receiveShadow = true; this.root.add(o); return o; };
    this.bodyParts = { hip: mesh(this.sphere, this.mBody), mid: mesh(this.capsule, this.mBody), chest: mesh(this.sphere, this.mBody), belly: mesh(this.capsule, this.mBelly), front: mesh(this.sphere, this.mBelly) };
    this.neck = mesh(this.capsule, this.mBody); this.skull = mesh(this.sphere, this.mBody); this.snout = mesh(this.capsule, sp.trunk ? this.mBody : this.mBelly);
    this.nose = mesh(this.sphere, this.mDark); this.eyes = [0, 1].map(() => mesh(this.sphere, this.mEye));
    this.ears = [0, 1].map(() => mesh(sp.ears === 'floppy' ? this.capsule : sp.ears === 'elephant' || sp.ears === 'round' ? this.sphere : this.cone, this.mBody));
    this.legs = ['Left', 'Right'].flatMap((side) => [['Arm', 'ForeArm', 'Hand', 'HandEnd'], ['UpLeg', 'Leg', 'Foot', 'FootEnd']].map((ch) => ({ pts: ch.map((n) => this.ix[side + n]), segs: [0, 1, 2].map((k) => mesh(this.capsule, k === 2 ? this.mFoot : this.mBody)), pad: mesh(this.sphere, sp.hooves || sp.socks ? this.mFoot : sp.column ? this.mBody : this.mDark), front: ch[0] === 'Arm' })));
    this.tailN = sp.tail[3] === 'stub' || sp.tail[3] === 'puff' ? 2 : 6;
    this.tail = Array.from({ length: this.tailN }, (_, k) => mesh(sp.tail[3] === 'puff' ? this.sphere : this.capsule, (sp.tail[4] && k === this.tailN - 1) ? mat(sp.tail[4]) : sp.tail[3] === 'hair' ? mat(sp.mane || sp.col[2], 0.95) : this.mBody));
    if (sp.mane) this.mane = [0, 1, 2, 3].map(() => mesh(this.sphere, mat(sp.mane, 0.95)));
    if (sp.maneHead) this.maneHead = mesh(this.sphere, mat(sp.maneHead, 0.95));
    if (sp.antlers) this.antlers = Array.from({ length: 8 }, () => mesh(this.capsule, mat(sp.antlers, 0.7)));
    if (sp.trunk) this.trunk = Array.from({ length: 6 }, () => mesh(this.capsule, this.mBody));
    if (sp.tusks) this.tusks = [0, 1].map(() => mesh(this.capsule, mat(sp.tusks, 0.4)));
  }
  render(A, B, f, dt) {
    if (this.model) { const W = this.Wi; for (let i = 0; i < W.length; i++) W[i] = lerp(A[i], B[i], f); if (!globalThis.SOMA_ANIMAL_REST) { this.model.apply(W, this.ix, this.netRest, this.pawC, dt); this.model.body?.update(this.pawC || this.contacts, this.speed, dt); } return; }
    const T = this.THREE, V = (x, y, z) => new T.Vector3(x, y, z), sp = this.sp;
    const J = (n) => { const i = this.ix[n] * 3; return V(lerp(A[i], B[i], f), lerp(A[i + 1], B[i + 1], f), lerp(A[i + 2], B[i + 2], f)); };
    const limb = (o, a, b, r, flat = 1) => { const d = b.clone().sub(a), len = Math.max(1e-4, d.length()); o.position.copy(a).add(b).multiplyScalar(0.5); o.quaternion.setFromUnitVectors(V(0, 1, 0), d.multiplyScalar(1 / len)); o.scale.set(r, (len + 1.4 * r) / 3, r * flat); };
    const blob = (o, p, rx, ry, rz, dir, up) => { o.position.copy(p); o.scale.set(rx, ry, rz); const z = dir.clone().normalize(), x = up.clone().cross(z).normalize(), y = z.clone().cross(x); o.quaternion.setFromRotationMatrix(new T.Matrix4().makeBasis(x, y, z)); };
    const hips = J('Hips'), head = J('Head'), headEnd = J('HeadEnd');
    const lS = J('LeftShoulder'), rS = J('RightShoulder'), lH = J('LeftUpLeg'), rH = J('RightUpLeg'), withers = lS.clone().add(rS).multiplyScalar(0.5);
    const { sF, sU, sX } = this.spineFrame(hips, lS, rS, lH, rH), D = this.dims;
    const at = (o, off) => o.clone().addScaledVector(sF, off[0]).addScaledVector(sU, off[1]).addScaledVector(sX, off[2]);
    const chestC = at(withers, D.chestOff), hipC = at(hips, D.hipOff);
    const fwd = chestC.clone().sub(hipC).normalize(), up = sU.clone().addScaledVector(fwd, -sU.dot(fwd)).normalize();
    const breath = 1 + 0.02 * Math.sin(this.clock * 2 * Math.PI * (0.3 + 0.6 * Math.min(1, this.speed / sp.speeds[2])));
    const B0 = this.bodyParts;
    blob(B0.hip, hipC, D.rx, D.ry, D.rz, fwd, up); blob(B0.chest, chestC, D.cx * breath, D.cy * breath, D.cz, fwd, up);
    limb(B0.mid, hipC, chestC, Math.min(D.ry, D.cy) * 0.95 * breath); B0.mid.scale.x = (D.rx + D.cx) / 2 * breath;
    const bellyA = hipC.clone().addScaledVector(up, -D.ry * 0.35), bellyB = chestC.clone().addScaledVector(up, -D.cy * 0.42); limb(B0.belly, bellyA, bellyB, (D.ry + D.cy) * 0.3);
    blob(B0.front, chestC.clone().addScaledVector(fwd, D.cz * 0.6).addScaledVector(up, -D.cy * 0.18), D.cx * 0.72, D.cy * 0.7, D.cz * 0.42, fwd, up);
    // neck & head
    const neckBase = J('Neck').lerp(chestC, 0.3);
    const hr = sp.head[0], hl = sp.head[1], hw = sp.head[2];
    const hDir = headEnd.clone().sub(head).normalize(), hUp = up.clone().sub(hDir.clone().multiplyScalar(up.dot(hDir))).normalize(), hSide = hDir.clone().cross(hUp).normalize();
    const skullP = head.clone().addScaledVector(hUp, hr * 0.15);
    limb(this.neck, neckBase, skullP, (D.cx * 0.5 + hr * 0.5) * 0.75);
    blob(this.skull, skullP, hr, hr * 0.92, hr * 1.1, hDir, hUp);
    const snA = skullP.clone().addScaledVector(hDir, hr * 0.5).addScaledVector(hUp, -hr * 0.2), snB = snA.clone().addScaledVector(hDir, hl * 0.85);
    limb(this.snout, snA, snB, hw);
    this.nose.position.copy(snB).addScaledVector(hDir, hw * 0.5); this.nose.scale.setScalar(hw * 0.45); this.nose.visible = !sp.trunk;
    this.eyes.forEach((o, k) => { o.position.copy(skullP).addScaledVector(hDir, hr * 0.62).addScaledVector(hSide, (k ? 1 : -1) * hr * 0.55).addScaledVector(hUp, hr * 0.22); o.scale.setScalar(hr * 0.13); });
    this.ears.forEach((o, k) => {
      const sgn = k ? 1 : -1, es = sp.earSize, base = skullP.clone().addScaledVector(hSide, sgn * hr * 0.55).addScaledVector(hUp, hr * 0.7);
      if (sp.ears === 'floppy') { const tip = base.clone().addScaledVector(hSide, sgn * es * 0.35).addScaledVector(hUp, -es * 0.9); limb(o, base, tip, es * 0.32, 0.4); }
      else if (sp.ears === 'elephant') { blob(o, skullP.clone().addScaledVector(hSide, sgn * hr).addScaledVector(hDir, -hr * 0.3), es * 0.6, es * 0.75, es * 0.15, hSide.clone().multiplyScalar(sgn), hUp); }
      else if (sp.ears === 'round') { blob(o, base, es, es * 0.85, es * 0.5, hDir, hUp); }
      else { const dir = hUp.clone().addScaledVector(hSide, sgn * 0.35).addScaledVector(hDir, sp.ears === 'long' ? -0.45 : -0.2).normalize(); o.position.copy(base).addScaledVector(dir, es * 0.5); o.scale.set(es * (sp.ears === 'oval' ? 0.32 : 0.28), es, es * (sp.ears === 'long' ? 0.15 : 0.18)); o.quaternion.setFromUnitVectors(V(0, 1, 0), dir); }
    });
    if (this.mane) this.mane.forEach((o, k) => { const p = neckBase.clone().lerp(skullP, (k + 0.5) / 4).addScaledVector(up, D.cx * 0.35); blob(o, p, D.cx * 0.18, D.cx * 0.25, sp.neck[0] * 0.15, skullP.clone().sub(neckBase), up); });
    if (this.maneHead) blob(this.maneHead, skullP.clone().addScaledVector(hDir, -hr * 0.45), hr * 1.75, hr * 1.7, hr * 1.25, hDir, hUp);
    if (this.antlers) this.antlers.forEach((o, k) => { const sgn = k < 4 ? 1 : -1, j = k % 4, base = skullP.clone().addScaledVector(hSide, sgn * hr * 0.35).addScaledVector(hUp, hr * 0.75); const beam = hUp.clone().addScaledVector(hSide, sgn * 0.45).addScaledVector(hDir, -0.35).normalize(); const a = base.clone().addScaledVector(beam, j === 0 ? 0 : 0.12 + 0.07 * j); const d = j === 0 ? beam : hUp.clone().addScaledVector(hSide, sgn * 0.25).addScaledVector(hDir, 0.45 - 0.2 * j).normalize(); limb(o, a, a.clone().addScaledVector(d, j === 0 ? 0.32 : 0.13), 0.012 + (j === 0 ? 0.006 : 0)); });
    if (this.trunk) { let p = snB.clone(), d = hDir.clone(); this.trunk.forEach((o, k) => { d = d.clone().lerp(V(Math.sin(this.clock * 1.1 + k) * 0.15, -1, 0.1), 0.45).normalize(); const q2 = p.clone().addScaledVector(d, 0.19); limb(o, p, q2, 0.11 * (1 - k * 0.11)); p = q2; }); }
    if (this.tusks) this.tusks.forEach((o, k) => { const sgn = k ? 1 : -1, a = snA.clone().addScaledVector(hSide, sgn * hw * 0.8).addScaledVector(hUp, -hw * 0.3); limb(o, a, a.clone().addScaledVector(hDir, 0.45).add(V(0, -0.12, 0)).addScaledVector(hSide, sgn * 0.05), 0.045); });
    // legs from the network's joints (the hidden shoulder/hip segment is inside the body)
    for (const L of this.legs) {
      const [a, b, c2, d] = L.pts.map((i) => J(this.names[i]));
      const thick = (L.front ? sp.front[0] + sp.front[1] : sp.hind[0] + sp.hind[1]) * 0.13 * (sp.heavy ? 1.3 : 1);
      limb(L.segs[0], a, b, thick * 1.05); limb(L.segs[1], b, c2, thick * 0.78); limb(L.segs[2], c2, d, thick * (sp.hooves ? 0.62 : 0.55));
      L.pad.position.copy(d).add(V(0, thick * 0.3, 0)); L.pad.scale.set(thick * 0.85, thick * 0.45, thick * 1.1); L.pad.quaternion.setFromUnitVectors(V(0, 0, 1), V(fwd.x, 0, fwd.z).normalize());
    }
    // tail: the network's tail bones, extended by a lagging spring chain to the species' length
    const t0 = J('Tail'), t1 = J('Tail1'), t2 = J('Tail1End'), segL = sp.tail[0] / this.tailN, style = sp.tail[3];
    const base = hipC.clone().addScaledVector(fwd, -D.rz * 0.85).addScaledVector(up, D.ry * 0.35);
    const dir0 = t2.clone().sub(t0).normalize();
    if (!this.tailPts) this.tailPts = Array.from({ length: this.tailN + 1 }, (_, k) => base.clone().addScaledVector(dir0, segL * k));
    const pts = this.tailPts; pts[0].copy(base);
    for (let k = 1; k <= this.tailN; k++) {
      const rest = base.clone().addScaledVector(dir0, segL * k);
      pts[k].lerp(rest, 1 - Math.exp(-dt * (style === 'hair' ? 8 : 14)));
      const d = pts[k].clone().sub(pts[k - 1]); d.setLength(segL); pts[k].copy(pts[k - 1]).add(d);
    }
    this.tail.forEach((o, k) => {
      const tr = sp.tail[2], taper = style === 'bushy' ? (0.6 + Math.sin(Math.PI * (k + 0.5) / this.tailN) * 0.9) : style === 'puff' ? 1.6 : style === 'tuft' && k === this.tailN - 1 ? 2.4 : 1 - 0.6 * k / this.tailN;
      if (style === 'puff') { o.position.copy(pts[k]).add(pts[k + 1]).multiplyScalar(0.5); o.scale.setScalar(tr * taper); }
      else limb(o, pts[k], pts[k + 1], tr * taper);
    });
    void t1;
  }
}
