// Body layer: muscle contraction, breathing and soft-tissue motion on the SOMA skin, in the vertex shader.
// Muscle regions are found on the bind-pose mesh from the skin weights and bone geometry (elbow/knee
// flexion sides come from the learned hinge axes); each region bulges along the skin normal by an
// amount driven per frame by joint angles, foot contact and effort, with activation/relaxation time
// constants. Breathing rate and depth follow exertion; soft tissue is a damped spring per region driven
// by the bone's acceleration. All procedural: the model and the mocap have no muscle or breath data.
import * as M from './math3d.js';

const smooth = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };
const win = (x, lo, hi, soft) => smooth(lo - soft, lo + soft, x) * (1 - smooth(hi - soft, hi + soft, x));

// muscle channels (index = attribute channel); amp in metres at full activation
const MUSCLES = [
  ['bicepsL', 0.012], ['bicepsR', 0.012], ['tricepsL', 0.008], ['tricepsR', 0.008],
  ['forearmL', 0.006], ['forearmR', 0.006], ['deltoidL', 0.008], ['deltoidR', 0.008],
  ['pecL', 0.007], ['pecR', 0.007], ['latL', 0.006], ['latR', 0.006],
  ['quadL', 0.011], ['quadR', 0.011], ['hamL', 0.009], ['hamR', 0.009],
  ['calfL', 0.011], ['calfR', 0.011], ['gluteL', 0.009], ['gluteR', 0.009],
  ['tibL', 0.004], ['tibR', 0.004], ['abs', 0.005], ['traps', 0.008],
  ['breathChest', 1], ['breathBelly', 1], ['_', 0], ['_', 0],
];
const JIGGLE = ['belly', 'glutes', 'chest', 'thighL', 'thighR', 'armL', 'armR', 'calves'];
const NMUS = MUSCLES.length, NV4 = NMUS / 4;

export class BodyFX {
  constructor(THREE, geo, sk, inverses, limits) {
    this.THREE = THREE; this.sk = sk;
    const ix = (n) => sk.names.indexOf(n);
    this.ix = ix;
    // bind pose: world transform of every joint
    const bind = inverses.map((m) => new THREE.Matrix4().copy(m).invert());
    const bp = bind.map((m) => new THREE.Vector3().setFromMatrixPosition(m));
    const bq = bind.map((m) => new THREE.Quaternion().setFromRotationMatrix(m));
    this.bindQ = bq.map((q) => [q.x, q.y, q.z, q.w]);
    const P = (n) => bp[ix(n)], Q = (n) => this.bindQ[ix(n)];
    const hinge = (n) => limits.items.find((it) => it.j === ix(n));
    // flexion direction of a hinge (where the child segment moves when the joint flexes), bind world space
    const flexDir = (joint, child, sign) => {
      const it = hinge(joint), axis = M.qrot(Q(joint), it.axes[0]), seg = M.vsub(P(child).toArray(), P(joint).toArray());
      return M.vnorm(M.vscale(M.vcross(axis, seg), sign));
    };
    const L = { elbow: flexDir('LeftForeArm', 'LeftHand', 1), knee: flexDir('LeftShin', 'LeftFoot', -1) };
    const R = { elbow: flexDir('RightForeArm', 'RightHand', 1), knee: flexDir('RightShin', 'RightFoot', -1) };
    this.axes = { LeftForeArm: hinge('LeftForeArm').axes[0], RightForeArm: hinge('RightForeArm').axes[0], LeftShin: hinge('LeftShin').axes[0], RightShin: hinge('RightShin').axes[0] };
    const pos = geo.attributes.position, nrm = geo.attributes.normal, si = geo.attributes.skinIndex, sw = geo.attributes.skinWeight, V = pos.count;
    const mus = new Uint8Array(V * NMUS), jig = new Uint8Array(V * 8);
    const yArm = P('LeftArm').y, yNeck = P('Neck1').y, yChest = P('Chest').y, ySp2 = P('Spine2').y, ySp1 = P('Spine1').y, yLeg = P('LeftLeg').y, yHip = P('Hips').y;
    const p = new THREE.Vector3(), n = new THREE.Vector3();
    // weight of a vertex on a bone segment a->b: position along it (t), and how much its radial direction faces `dir`
    const seg = (a, b, t0, t1, dir, sharp) => {
      const A = P(a), B = P(b), ab = new THREE.Vector3().subVectors(B, A), len = ab.length(); ab.normalize();
      const ap = new THREE.Vector3().subVectors(p, A), t = ap.dot(ab) / len;
      const radial = ap.addScaledVector(ab, -ap.dot(ab));
      const r = radial.length(); if (r < 1e-5) return 0;
      const facing = Math.max(0, Math.min(1, radial.dot(new THREE.Vector3(...dir).normalize()) / r));
      return win(t, t0, t1, 0.12) * Math.pow(facing, sharp);
    };
    for (let v = 0; v < V; v++) {
      p.fromBufferAttribute(pos, v); n.fromBufferAttribute(nrm, v);
      const w = {}; for (let k = 0; k < 4; k++) { const j = si.getComponent(v, k); w[j] = (w[j] || 0) + sw.getComponent(v, k); }
      const ws = (...names) => names.reduce((a, nm) => a + (w[ix(nm)] || 0), 0);
      const side = p.x >= 0 ? 'L' : 'R', S = side === 'L' ? L : R, sx = side === 'L' ? 1 : -1;
      const ch = {};
      const arm = ws(`${side === 'L' ? 'Left' : 'Right'}Arm`), fore = ws(`${side === 'L' ? 'Left' : 'Right'}ForeArm`);
      const thigh = ws(`${side === 'L' ? 'Left' : 'Right'}Leg`), shin = ws(`${side === 'L' ? 'Left' : 'Right'}Shin`);
      const pre = side === 'L' ? 'Left' : 'Right';
      if (arm > 0.2) {
        ch['biceps' + side] = arm * seg(`${pre}Arm`, `${pre}ForeArm`, 0.3, 0.85, S.elbow, 1.6);
        ch['triceps' + side] = arm * seg(`${pre}Arm`, `${pre}ForeArm`, 0.25, 0.85, M.vscale(S.elbow, -1), 1.6);
        ch['deltoid' + side] = arm * seg(`${pre}Arm`, `${pre}ForeArm`, -0.1, 0.3, [sx, 0.6, 0], 1.0);
        jig[v * 8 + (side === 'L' ? 5 : 6)] = Math.round(255 * arm * seg(`${pre}Arm`, `${pre}ForeArm`, 0.3, 0.8, M.vscale(S.elbow, -1), 1.2) * 0.8);
      }
      if (fore > 0.2) ch['forearm' + side] = fore * seg(`${pre}ForeArm`, `${pre}Hand`, 0.05, 0.55, S.elbow, 1.2);
      if (thigh > 0.2) {
        ch['quad' + side] = thigh * seg(`${pre}Leg`, `${pre}Shin`, 0.15, 0.85, M.vscale(S.knee, -1), 1.4);
        ch['ham' + side] = thigh * seg(`${pre}Leg`, `${pre}Shin`, 0.2, 0.85, S.knee, 1.4);
        jig[v * 8 + (side === 'L' ? 3 : 4)] = Math.round(255 * thigh * seg(`${pre}Leg`, `${pre}Shin`, 0.05, 0.6, [sx * 0.3, 0, -0.95], 0.6) * 0.7);
      }
      if (shin > 0.2) {
        ch['calf' + side] = shin * seg(`${pre}Shin`, `${pre}Foot`, 0.05, 0.55, S.knee, 1.4);
        ch['tib' + side] = shin * seg(`${pre}Shin`, `${pre}Foot`, 0.1, 0.7, M.vscale(S.knee, -1), 2.0) * 0.8;
        jig[v * 8 + 7] = Math.round(255 * shin * seg(`${pre}Shin`, `${pre}Foot`, 0.05, 0.45, S.knee, 1.4) * 0.6);
      }
      const torso = ws('Chest', 'Spine2', 'Spine1', `${pre}Shoulder`), lower = ws('Hips', 'Spine1'), gl = ws('Hips', `${pre}Leg`);
      if (torso > 0.15) {
        ch['pec' + side] = torso * win(sx * p.x, 0.015, 0.15, 0.03) * win(p.y, yArm - 0.15, yArm + 0.01, 0.03) * smooth(0.2, 0.7, n.z);
        ch['lat' + side] = torso * win(p.y, ySp2 - 0.04, yArm - 0.06, 0.04) * smooth(0.3, 0.8, sx * n.x) * smooth(-0.6, 0.2, -n.z);
        ch.traps = Math.max(ch.traps || 0, ws('Chest', 'Neck1', 'LeftShoulder', 'RightShoulder') * win(p.y, yArm - 0.02, yNeck + 0.06, 0.03) * win(Math.abs(p.x), -0.1, 0.16, 0.04) * smooth(-0.1, 0.6, -n.z + 0.6 * n.y));
        ch.breathChest = torso * win(p.y, ySp2 - 0.02, yArm - 0.02, 0.04) * Math.max(smooth(0.1, 0.6, n.z), 0.6 * smooth(0.4, 0.9, Math.abs(n.x)), 0.35 * smooth(0.3, 0.8, -n.z));
        jig[v * 8 + 2] = Math.round(255 * torso * ch['pec' + side] * 0.6);
      }
      if (lower > 0.15) {
        ch.abs = Math.max(ch.abs || 0, lower * win(p.y, yLeg - 0.02, ySp2, 0.04) * win(Math.abs(p.x), -0.1, 0.11, 0.03) * smooth(0.4, 0.85, n.z));
        ch.breathBelly = lower * win(p.y, yLeg - 0.03, ySp1 + 0.06, 0.04) * smooth(0.2, 0.8, n.z);
        jig[v * 8] = Math.round(255 * lower * win(p.y, yLeg - 0.05, ySp1 + 0.03, 0.05) * smooth(0.1, 0.8, n.z) * 0.9);
      }
      if (gl > 0.15) {
        const g = gl * win(p.y, yLeg - 0.16, yHip + 0.03, 0.04) * win(sx * p.x, 0.005, 0.17, 0.03) * smooth(0.25, 0.8, -n.z);
        ch['glute' + side] = g; jig[v * 8 + 1] = Math.round(255 * g * 0.9);
      }
      MUSCLES.forEach(([name], c) => { if (ch[name]) mus[v * NMUS + c] = Math.round(255 * Math.min(1, ch[name])); });
    }
    // per-vertex surface gradient of its dominant muscle's weight (normal tilt => bulges catch the light)
    const idx = geo.index.array, nb = Array.from({ length: V }, () => new Set());
    for (let t = 0; t < idx.length; t += 3) { const a = idx[t], b = idx[t + 1], c = idx[t + 2]; nb[a].add(b).add(c); nb[b].add(a).add(c); nb[c].add(a).add(b); }
    const grad = new Float32Array(V * 4), P3 = (v) => [pos.getX(v), pos.getY(v), pos.getZ(v)];
    for (let v = 0; v < V; v++) {
      let best = -1, bw = 0.05;
      for (let c = 0; c < 24; c++) { const wv = mus[v * NMUS + c] / 255; if (wv > bw) { bw = wv; best = c; } }
      if (best < 0) { grad[v * 4 + 3] = 26; continue; } // channel 26 is always zero
      const pv = P3(v), g = [0, 0, 0]; let cnt = 0;
      for (const u of nb[v]) {
        const pu = P3(u), d = M.vsub(pu, pv), l2 = M.vdot(d, d); if (l2 < 1e-12) continue;
        const dw = (mus[u * NMUS + best] - mus[v * NMUS + best]) / 255;
        g[0] += dw * d[0] / l2; g[1] += dw * d[1] / l2; g[2] += dw * d[2] / l2; cnt++;
      }
      const nv = [nrm.getX(v), nrm.getY(v), nrm.getZ(v)], gn = M.vscale(g, cnt ? 1 / cnt : 0), tg = M.vsub(gn, M.vscale(nv, M.vdot(gn, nv)));
      const mag = Math.min(60, M.vlen(tg)), dir = M.vlen(tg) > 1e-9 ? M.vscale(tg, mag / M.vlen(tg)) : [0, 0, 0];
      grad.set([dir[0], dir[1], dir[2], best], v * 4);
    }
    geo.setAttribute('mgrad', new THREE.BufferAttribute(grad, 4));
    for (let a = 0; a < NV4; a++) {
      const arr = new Uint8Array(V * 4);
      for (let v = 0; v < V; v++) for (let c = 0; c < 4; c++) arr[v * 4 + c] = mus[v * NMUS + a * 4 + c];
      geo.setAttribute(`mus${a}`, new THREE.BufferAttribute(arr, 4, true));
    }
    for (let a = 0; a < 2; a++) {
      const arr = new Uint8Array(V * 4);
      for (let v = 0; v < V; v++) for (let c = 0; c < 4; c++) arr[v * 4 + c] = jig[v * 8 + a * 4 + c];
      geo.setAttribute(`jig${a}`, new THREE.BufferAttribute(arr, 4, true));
    }
    this.uMus = Array.from({ length: NV4 }, () => new THREE.Vector4());
    this.uJig = Array.from({ length: 8 }, () => new THREE.Vector3());
    this.uAct = new Float32Array(NMUS);
    this.act = new Float64Array(NMUS);
    this.jig = JIGGLE.map(() => ({ x: [0, 0, 0], v: [0, 0, 0], prev: null, prevV: null }));
    this.jigBone = ['Spine1', 'Hips', 'Chest', 'LeftLeg', 'RightLeg', 'LeftArm', 'RightArm', 'Hips'].map(ix);
    this.exertion = 0; this.effort = 0; this.breath = 0; this.prevElbow = [0, 0];
    this.jaw = ix('Jaw'); this.jawAxis = null;
    // mouth-opening axis: rotation of the jaw that moves the chin down
    const jp = P('Jaw'), chinY = [];
    for (let v = 0; v < V; v++) { const j0 = si.getComponent(v, 0); if (j0 === this.jaw && sw.getComponent(v, 0) > 0.8) chinY.push(v); }
    if (chinY.length) {
      let low = chinY[0]; for (const v of chinY) if (pos.getY(v) < pos.getY(low)) low = v;
      const c = new THREE.Vector3().fromBufferAttribute(pos, low).sub(jp).toArray();
      const ax = [1, 0, 0], test = M.qrot(M.qAxisAngle(ax, 0.2), c);
      const worldAxis = test[1] < c[1] ? ax : [-1, 0, 0];
      this.jawAxis = M.qrotInv(Q('Jaw'), worldAxis); // local axis
    }
    this.chest = ix('Chest'); this.spine2 = ix('Spine2');
    // anatomical lateral axis (world +X in the bind pose) in each spine joint's own frame
    this.spineAxis = Object.fromEntries([this.spine2, this.chest].map((j) => [j, M.qrotInv(this.bindQ[j], [1, 0, 0])]));
  }
  install(material) {
    const self = this;
    material.onBeforeCompile = (sh) => {
      for (let a = 0; a < NV4; a++) sh.uniforms[`uMus${a}`] = { value: self.uMus[a] };
      sh.uniforms.uJig = { value: self.uJig };
      sh.uniforms.uAct = { value: self.uAct };
      const decl = Array.from({ length: NV4 }, (_, a) => `attribute vec4 mus${a};\nuniform vec4 uMus${a};`).join('\n') + `\nattribute vec4 jig0;\nattribute vec4 jig1;\nattribute vec4 mgrad;\nuniform vec3 uJig[8];\nuniform float uAct[${NMUS}];\n`;
      const disp = Array.from({ length: NV4 }, (_, a) => `dot(mus${a}, uMus${a})`).join(' + ');
      sh.vertexShader = decl + sh.vertexShader.replace('#include <beginnormal_vertex>', `#include <beginnormal_vertex>
        objectNormal = normalize(objectNormal - mgrad.xyz * uAct[int(mgrad.w + 0.5)]);`).replace('#include <begin_vertex>', `#include <begin_vertex>
        transformed += normal * (${disp});
        transformed += jig0.x * uJig[0] + jig0.y * uJig[1] + jig0.z * uJig[2] + jig0.w * uJig[3] + jig1.x * uJig[4] + jig1.y * uJig[5] + jig1.z * uJig[6] + jig1.w * uJig[7];`);
    };
    material.needsUpdate = true;
  }
  off() { for (const u of this.uMus) u.set(0, 0, 0, 0); for (const u of this.uJig) u.set(0, 0, 0); this.uAct.fill(0); }
  // before FK: breathing posture (chest rise) and the jaw when breathing hard
  posePass(local, dt) {
    const out = new Float64Array(local), b = 0.5 - 0.5 * Math.cos(this.breath), E = this.exertion;
    const lift = (0.006 + 0.012 * E) * b; // radians of thoracic extension on the inhale
    for (const j of [this.spine2, this.chest]) M.setQ(out, j, M.qnorm(M.qmul(M.Q(out, j), M.qAxisAngle(this.spineAxis[j], -lift))));
    if (this.jawAxis && E > 0.45) M.setQ(out, this.jaw, M.qnorm(M.qmul(M.Q(out, this.jaw), M.qAxisAngle(this.jawAxis, (E - 0.45) * 0.22 * (0.4 + 0.6 * b)))));
    return out;
  }
  // after FK: drive activations from the displayed pose
  update(local, world, contacts, speed, maxSpeed, dt) {
    if (dt <= 0) return;
    const ix = this.ix, WP = (n) => M.P(world, ix(n)), sk = this.sk;
    const target = new Float64Array(NMUS);
    const effortT = Math.min(1, speed / Math.max(1, maxSpeed));
    this.effort += (effortT - this.effort) * (1 - Math.exp(-dt / 0.3));
    const E0 = this.exertion; this.exertion += (this.effort - E0) * (1 - Math.exp(-dt / (this.effort > E0 ? 6 : 18)));
    const eff = this.effort;
    const hingeAngle = (n) => M.vdot(M.qToScaledAxis(M.qabs(M.Q(local, ix(n)))), this.axes[n]);
    const up = M.vnorm(M.vsub(WP('Neck1'), WP('Hips')));
    ['Left', 'Right'].forEach((side, s) => {
      const S = s === 0 ? 'L' : 'R', c = contacts ? Math.min(1, contacts[s * 2] + contacts[s * 2 + 1]) : 1;
      const elbow = Math.min(1, Math.max(0, hingeAngle(`${side}ForeArm`) / 2.4));
      const knee = Math.min(1, Math.max(0, -hingeAngle(`${side}Shin`) / 2.3));
      const ev = (elbow - this.prevElbow[s]) / dt; this.prevElbow[s] = elbow;
      const upper = M.vnorm(M.vsub(WP(`${side}ForeArm`), WP(`${side}Arm`)));
      const elev = Math.min(1, Math.max(0, (Math.acos(Math.max(-1, Math.min(1, -M.vdot(upper, up)))) - 0.35) / 1.4));
      const thighV = M.vnorm(M.vsub(WP(`${side}Shin`), WP(`${side}Leg`)));
      const fwd = M.vnorm(M.vcross(M.vsub(WP('RightLeg'), WP('LeftLeg')), up));
      const hipFlex = M.vdot(thighV, fwd), armFwd = M.vdot(upper, fwd);
      const foot = M.vnorm(M.vsub(WP(`${side}ToeBase`), WP(`${side}Foot`))), shinV = M.vnorm(M.vsub(WP(`${side}Foot`), WP(`${side}Shin`)));
      const plantar = Math.max(0, Math.min(1, (M.vdot(foot, shinV) + 0.05) / 0.5));
      const t = (k, v) => { target[MUSCLES.findIndex(([n]) => n === k + S)] = Math.min(1, Math.max(0, v)); };
      t('biceps', 0.12 + 0.75 * elbow + 0.15 * eff);
      t('triceps', 0.1 + 0.5 * (1 - elbow) * eff + 0.25 * Math.max(0, -ev) * 0.2);
      t('forearm', 0.15 + 0.35 * eff + 0.2 * elbow);
      t('deltoid', 0.15 + 0.7 * elev + 0.35 * eff * Math.abs(armFwd));
      t('pec', 0.1 + 0.5 * Math.max(0, armFwd) * (0.4 + eff) + 0.2 * elev);
      t('lat', 0.1 + 0.6 * Math.max(0, -armFwd) * (0.4 + eff));
      t('quad', 0.12 + c * (0.25 + 0.75 * knee) * (0.35 + 0.65 * eff));
      t('ham', 0.1 + (1 - c) * knee * (0.3 + 0.7 * eff) + 0.3 * Math.max(0, -hipFlex) * c * eff);
      t('calf', 0.12 + c * (0.3 + 0.7 * plantar) * (0.35 + 0.65 * eff));
      t('glute', 0.1 + c * Math.max(0, -hipFlex + 0.15) * (0.5 + eff));
      t('tib', 0.1 + (1 - c) * (1 - plantar) * 0.8 * (0.3 + eff));
    });
    // breathing: rate and depth follow exertion, recovery is slow; abs tighten on the exhale when working
    this.breath += dt * 2 * Math.PI * (0.22 + 0.55 * this.exertion);
    const b = 0.5 - 0.5 * Math.cos(this.breath), bl = 0.5 - 0.5 * Math.cos(this.breath - 0.6);
    target[MUSCLES.findIndex(([n]) => n === 'abs')] = 0.15 + 0.45 * eff + 0.3 * this.exertion * (1 - b);
    target[MUSCLES.findIndex(([n]) => n === 'traps')] = 0.15 + 0.4 * eff + 0.15 * this.exertion * b;
    for (let i = 0; i < NMUS; i++) {
      const [name, amp] = MUSCLES[i];
      if (name.startsWith('breath') || name === '_') continue;
      const a = this.act[i], tau = target[i] > a ? 0.06 : 0.16;
      this.act[i] = a + (target[i] - a) * (1 - Math.exp(-dt / tau));
    }
    const chest = (0.0025 + 0.006 * this.exertion) * b, belly = (0.002 + 0.0035 * this.exertion) * bl;
    if (this.preview) { this.act.fill(1); }
    for (let a = 0; a < NV4; a++) {
      const v = [0, 1, 2, 3].map((c) => { const i = a * 4 + c, [name, amp] = MUSCLES[i]; return name === 'breathChest' ? chest : name === 'breathBelly' ? belly : this.act[i] * amp; });
      this.uMus[a].set(v[0], v[1], v[2], v[3]);
      for (let c = 0; c < 4; c++) this.uAct[a * 4 + c] = a * 4 + c < 24 ? v[c] : 0;
    }
    // soft tissue: damped spring per region, driven by the bone's acceleration, expressed in bind space
    const k = [5.5, 6, 6.5, 6, 6, 7, 7, 7].map((f) => (2 * Math.PI * f) ** 2), zeta = 0.3;
    this.jig.forEach((r, i) => {
      const j = this.jigBone[i], pnow = M.P(world, j);
      if (r.prev && r.prevV) {
        const vel = M.vscale(M.vsub(pnow, r.prev), 1 / dt), acc = M.vscale(M.vsub(vel, r.prevV), 1 / dt);
        const w = Math.sqrt(k[i]), n = Math.max(1, Math.ceil(dt * 480)), h = dt / n; // sub-steps keep the stiff spring stable at low fps
        for (let d = 0; d < 3; d++) {
          const drive = Math.max(-40, Math.min(40, acc[d])) * (d === 1 ? 0.45 : 0.2);
          for (let it = 0; it < n; it++) { r.v[d] += (-k[i] * r.x[d] - 2 * zeta * w * r.v[d] - drive) * h; r.x[d] += r.v[d] * h; }
          if (Math.abs(r.x[d]) > 0.005) { r.x[d] = Math.sign(r.x[d]) * 0.005; r.v[d] *= 0.5; }
        }
        r.prevV = vel;
      } else if (r.prev) r.prevV = M.vscale(M.vsub(pnow, r.prev), 1 / dt);
      r.prev = pnow;
      const local3 = M.qrot(this.bindQ[j], M.qrotInv(M.Q(world, j), r.x));
      this.uJig[i].set(local3[0], local3[1], local3[2]);
    });
  }
}
