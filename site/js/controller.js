// Character controller: port of runtime/somart/controller.py + postprocess.py + anatomy.FingerAnimator.
// Generation is asynchronous (Web Worker); the playhead never waits for the network.
import * as M from './math3d.js';
import { applyLimits, headingFromHips, validFrames, MIN_TOKENS, MAX_TOKENS } from './codec.js';

const FPS = 30, DT = 1 / 30, LN2 = Math.log(2);

function springVelocity(v, a, goal, halflife, dt) {
  const y = (4 * LN2 / Math.max(halflife, 1e-5)) / 2, e = Math.exp(-y * dt), vo = [], ao = [];
  for (let k = 0; k < 3; k++) {
    const j0 = v[k] - goal[k], j1 = a[k] + j0 * y;
    vo.push(e * (j0 + j1 * dt) + goal[k]); ao.push(e * (a[k] - j1 * y * dt));
  }
  return [vo, ao];
}
function damperDecay(x, v, halflife, dt) {
  const y = (4 * LN2 / Math.max(halflife, 1e-5)) / 2, e = Math.exp(-y * dt);
  const j1 = v + x * y;
  return [e * (x + j1 * dt), e * (v - j1 * y * dt)];
}

// ------------------------------------------------------------------ clip library (target windows)
export class Library {
  constructor(sk, meta, buffer) {
    this.sk = sk; this.meta = meta;
    const W = meta.count, J = meta.joints, root = new Float32Array(buffer, 0, W * 4 * 3);
    const qi = new Int16Array(buffer, W * 4 * 3 * 4, W * 4 * J * 4);
    this.windows = [];
    for (let w = 0; w < W; w++) {
      const frames = [];
      for (let f = 0; f < 4; f++) {
        const pose = new Float64Array(sk.reference);
        for (let j = 0; j < J; j++) {
          const o = ((w * 4 + f) * J + j) * 4;
          M.setQ(pose, j, M.qnorm([qi[o] / 32767, qi[o + 1] / 32767, qi[o + 2] / 32767, qi[o + 3] / 32767]));
        }
        const r = (w * 4 + f) * 3;
        M.setP(pose, 0, [root[r], root[r + 1], root[r + 2]]);
        frames.push(pose);
      }
      this.windows.push(frames);
    }
    this.loco = meta.windows.map((m, i) => (m.kind === 'loco' ? i : -1)).filter((i) => i >= 0);
  }
  match(vLocal, yawRate, crouch, variation, rand) {
    let best = Infinity;
    const cost = this.loco.map((i) => {
      const m = this.meta.windows[i];
      let c = (m.v[0] - vLocal[0]) ** 2 + (m.v[1] - vLocal[1]) ** 2 + 0.15 * (m.yr - yawRate) ** 2;
      if ((m.h < 0.75) !== crouch) c += 1e3;
      best = Math.min(best, c);
      return c;
    });
    if (variation > 0) {
      const cand = this.loco.filter((_, k) => cost[k] <= best + variation * 0.25 + 1e-9);
      return cand[Math.floor(rand() * cand.length)];
    }
    return this.loco[cost.indexOf(best)];
  }
}
export function place(frames, pos, yaw) {
  const tq = M.qYaw(yaw), tp = [pos[0], 0, pos[2]];
  return frames.map((f) => {
    const out = new Float64Array(f);
    const [p, q] = M.tfMul(tp, tq, M.P(f, 0), M.Q(f, 0));
    M.setP(out, 0, p); M.setQ(out, 0, q);
    return out;
  });
}

// ------------------------------------------------------------------ post-processing
class Inertializer {
  constructor(J, halflife) { this.J = J; this.halflife = halflife; this.ro = new Float64Array(J * 3); this.rv = new Float64Array(J * 3); this.po = [0, 0, 0]; this.pv = [0, 0, 0]; }
  transition(src, srcPrev, dst, dstPrev, dt) {
    for (let j = 0; j < this.J; j++) {
      const a = M.qToScaledAxis(M.qmul(M.Q(src, j), M.qconj(M.Q(dst, j))));
      const b = M.qToScaledAxis(M.qmul(M.Q(srcPrev, j), M.qconj(M.Q(dstPrev, j))));
      for (let k = 0; k < 3; k++) { this.ro[j * 3 + k] = a[k]; this.rv[j * 3 + k] = (a[k] - b[k]) / dt; }
    }
    const s = M.P(src, 0), sp = M.P(srcPrev, 0), d = M.P(dst, 0), dp = M.P(dstPrev, 0);
    for (let k = 0; k < 3; k++) { this.po[k] = s[k] - d[k]; this.pv[k] = Math.max(-5, Math.min(5, ((s[k] - sp[k]) - (d[k] - dp[k])) / dt)); }
  }
  get active() { let m = 0; for (const x of this.ro) m = Math.max(m, Math.abs(x)); return m > 1e-4 || Math.abs(this.po[0]) + Math.abs(this.po[1]) + Math.abs(this.po[2]) > 1e-4; }
  apply(pose, dt) {
    for (let i = 0; i < this.ro.length; i++) [this.ro[i], this.rv[i]] = damperDecay(this.ro[i], this.rv[i], this.halflife, dt);
    for (let k = 0; k < 3; k++) [this.po[k], this.pv[k]] = damperDecay(this.po[k], this.pv[k], this.halflife, dt);
    const out = new Float64Array(pose);
    for (let j = 0; j < this.J; j++) M.setQ(out, j, M.qmul(M.qFromScaledAxis([this.ro[j * 3], this.ro[j * 3 + 1], this.ro[j * 3 + 2]]), M.Q(pose, j)));
    const p = M.P(pose, 0); M.setP(out, 0, [p[0] + this.po[0], p[1] + this.po[1], p[2] + this.po[2]]);
    return out;
  }
}

function twoBoneIK(local, world, sk, hip, knee, ankle, target, weight) {
  const a = M.P(world, hip), b = M.P(world, knee), c = M.P(world, ankle);
  const t = [c[0] + (target[0] - c[0]) * weight, c[1] + (target[1] - c[1]) * weight, c[2] + (target[2] - c[2]) * weight];
  const eps = 1e-5, lab = M.vlen(M.vsub(b, a)), lcb = M.vlen(M.vsub(b, c));
  if (lab < eps || lcb < eps) return local;
  const lat = Math.min(Math.max(M.vlen(M.vsub(t, a)), eps), lab + lcb - eps);
  const ang = (u, v) => Math.acos(Math.max(-1, Math.min(1, M.vdot(M.vnorm(u), M.vnorm(v)))));
  const ac0 = ang(M.vsub(c, a), M.vsub(b, a)), ba0 = ang(M.vsub(a, b), M.vsub(c, b));
  const ac1 = Math.acos(Math.max(-1, Math.min(1, (lcb * lcb - lab * lab - lat * lat) / (-2 * lab * lat))));
  const ba1 = Math.acos(Math.max(-1, Math.min(1, (lat * lat - lab * lab - lcb * lcb) / (-2 * lab * lcb))));
  const agr = M.Q(world, hip), bgr = M.Q(world, knee);
  let axis0 = M.vcross(M.vsub(c, a), M.vsub(b, a));
  if (M.vlen(axis0) < 1e-9) axis0 = M.qrot(agr, [1, 0, 0]);
  axis0 = M.vnorm(axis0);
  const r0 = M.qAxisAngle(M.qrotInv(agr, axis0), ac1 - ac0), r1 = M.qAxisAngle(M.qrotInv(bgr, axis0), ba1 - ba0);
  let hipQ = M.qnorm(M.qmul(M.Q(local, hip), r0));
  const kneeQ = M.qnorm(M.qmul(M.Q(local, knee), r1));
  const hp = sk.parents[hip], pw = hp >= 0 ? [M.P(world, hp), M.Q(world, hp)] : [[0, 0, 0], [0, 0, 0, 1]];
  const chain = (hq) => { const hw = M.tfMul(pw[0], pw[1], M.P(local, hip), hq); const kw = M.tfMul(hw[0], hw[1], M.P(local, knee), kneeQ); return [hw, kw, M.tfMul(kw[0], kw[1], M.P(local, ankle), [0, 0, 0, 1])[0]]; };
  let [hw, kw, c1] = chain(hipQ);
  const axis1 = M.vcross(M.vsub(c1, a), M.vsub(t, a));
  if (M.vlen(axis1) > 1e-9) {
    hipQ = M.qnorm(M.qmul(hipQ, M.qAxisAngle(M.qrotInv(hw[1], M.vnorm(axis1)), ang(M.vsub(c1, a), M.vsub(t, a)))));
    [hw, kw] = chain(hipQ);
  }
  const out = new Float64Array(local);
  M.setQ(out, hip, hipQ); M.setQ(out, knee, kneeQ);
  if (sk.parents[ankle] === knee) M.setQ(out, ankle, M.qnorm(M.qmul(M.qconj(kw[1]), M.Q(world, ankle))));
  return out;
}

class FootLocker {
  constructor(sk, s) {
    this.sk = sk; this.s = s;
    this.chains = [['LeftLeg', 'LeftShin', 'LeftFoot', 0], ['RightLeg', 'RightShin', 'RightFoot', 2]].map(([a, b, c, ci]) => [sk.names.indexOf(a), sk.names.indexOf(b), sk.names.indexOf(c), ci]);
    this.locked = [false, false]; this.lockPos = [[0, 0, 0], [0, 0, 0]]; this.weight = [0, 0];
    this.order = [];
    for (const [hip, knee, ankle] of this.chains) {
      const path = []; let j = hip;
      while (j >= 0) { path.push(j); j = sk.parents[j]; }
      for (const x of [...path.reverse(), knee, ankle]) if (!this.order.includes(x)) this.order.push(x);
    }
  }
  apply(local, contacts, dt) {
    const s = this.s(), world = new Float64Array(local.length);
    for (const j of this.order) {
      const p = this.sk.parents[j];
      if (p < 0) { for (let k = 0; k < 7; k++) world[j * 7 + k] = local[j * 7 + k]; continue; }
      const [wp, wq] = M.tfMul(M.P(world, p), M.Q(world, p), M.P(local, j), M.Q(local, j));
      M.setP(world, j, wp); M.setQ(world, j, wq);
    }
    let out = local;
    this.chains.forEach(([hip, knee, ankle, ci], i) => {
      const c = contacts[ci], pos = M.P(world, ankle);
      if (!this.locked[i] && c >= s.on) { this.locked[i] = true; this.lockPos[i] = pos.slice(); }
      else if (this.locked[i] && (c <= s.off || M.vlen(M.vsub(this.lockPos[i], pos)) > s.max)) this.locked[i] = false;
      const rate = dt / Math.max(this.locked[i] ? s.blendIn : s.blendOut, 1e-4);
      this.weight[i] = Math.min(1, Math.max(0, this.weight[i] + (this.locked[i] ? rate : -rate)));
      if (this.weight[i] > 1e-3) out = twoBoneIK(out, world, this.sk, hip, knee, ankle, this.lockPos[i], this.weight[i]);
    });
    return out;
  }
}

const FINGERS = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky'];
class FingerAnimator {
  constructor(sk, relaxed, s, seed = 1) {
    this.sk = sk; this.s = s; this.joints = [];
    const world = M.fk(relaxed, sk.parents), ix = (n) => sk.names.indexOf(n);
    ['Left', 'Right'].forEach((side, hand) => {
      const h = ix(`${side}Hand`), m1 = ix(`${side}HandMiddle1`), mt = ix(`${side}HandMiddleEnd`);
      const baseDir = M.vnorm(M.vsub(M.P(world, m1), M.P(world, h))), tip = M.vsub(M.P(world, mt), M.P(world, m1));
      const palm = M.vnorm(M.vsub(tip, M.vscale(baseDir, M.vdot(tip, baseDir))));
      FINGERS.forEach((name, f) => {
        const chain = [1, 2, 3, 4].map((k) => ix(`${side}Hand${name}${k}`)).filter((c) => c >= 0);
        const weights = (name === 'Thumb' ? [0.35, 0.55, 0.5] : [0.15, 1.0, 1.0, 0.75]).slice(0, chain.length);
        chain.forEach((c, k) => {
          const kid = sk.parents.indexOf(c);
          if (kid < 0) return;
          const bdir = M.vnorm(M.vsub(M.P(world, kid), M.P(world, c)));
          const n = name === 'Thumb' ? M.vnorm(M.vadd(palm, M.vscale(baseDir, 0.6))) : palm;
          this.joints.push({ c, axis: M.qrotInv(M.Q(world, c), M.vnorm(M.vcross(bdir, n))), w: weights[k], f, hand });
        });
      });
    });
    this.chains = ['LeftHand', 'RightHand'].map((n) => { const ch = []; let j = ix(n); while (j >= 0) { ch.push(j); j = sk.parents[j]; } return ch.reverse(); });
    const r = M.rng(seed);
    this.phase = Array.from({ length: 10 }, () => [r() * 6.283, r() * 6.283]);
    this.freq = Array.from({ length: 10 }, () => [0.15 + 0.3 * r(), 0.15 + 0.3 * r()]);
    this.x = new Float64Array(10); this.v = new Float64Array(10); this.prev = null; this.t = 0;
  }
  apply(local, speed, dt) {
    const s = this.s(); this.t += dt;
    let target;
    if (speed < 1.6) target = s.curlIdle + (s.curlWalk - s.curlIdle) * Math.min(1, speed / 1.6);
    else { const u = Math.min(1, (speed - 1.6) / 2.8); target = s.curlWalk + (s.curlRun - s.curlWalk) * u * u * (3 - 2 * u); }
    const hq = this.chains.map((ch) => ch.slice(1).reduce((q, j) => M.qmul(q, M.Q(local, j)), M.Q(local, ch[0])));
    if (this.prev && dt > 0) {
      for (let h = 0; h < 2; h++) {
        const drive = M.vlen(M.qToScaledAxis(M.qmul(hq[h], M.qconj(this.prev[h])))) / dt;
        [1, 0.8, 1, 1.1, 1.2].forEach((g, f) => {
          const i = h * 5 + f, acc = -120 * this.x[i] - 2 * Math.sqrt(120) * this.v[i] + s.follow * drive * g;
          this.v[i] += acc * dt; this.x[i] += this.v[i] * dt;
        });
      }
    }
    this.prev = hq;
    const out = new Float64Array(local);
    for (const fj of this.joints) {
      const i = fj.hand * 5 + fj.f, ph = this.phase[i], fr = this.freq[i];
      const noise = s.variation * (0.6 * Math.sin(6.2832 * fr[0] * this.t + ph[0]) + 0.4 * Math.sin(6.2832 * fr[1] * this.t + ph[1]));
      const curl = (target + this.x[i] + noise) * (fj.f === 0 ? 0.7 : 1);
      M.setQ(out, fj.c, M.qnorm(M.qmul(M.Q(local, fj.c), M.qAxisAngle(fj.axis, curl * Math.PI / 180 * fj.w))));
    }
    return out;
  }
}

// Head and chest turn toward the commanded direction before the body does (people look where they are
// about to go). Gives an immediate visual answer to input while the network is still generating.
class LookAt {
  constructor(sk) {
    const ix = (n) => sk.names.indexOf(n);
    this.sk = sk; this.chain = [['Chest', 0.15], ['Neck1', 0.25], ['Neck2', 0.25], ['Head', 0.35]].map(([n, f]) => [ix(n), f]);
    this.iR = ix('RightLeg'); this.iL = ix('LeftLeg'); this.x = 0; this.v = 0; this.heading = null;
  }
  apply(local, goal, dt, maxDeg) {
    const w = M.fk(local, this.sk.parents), h = headingFromHips(M.P(w, this.iR), M.P(w, this.iL));
    this.heading = this.heading === null ? h : this.heading + M.wrap(h - this.heading) * (1 - Math.exp(-dt / 0.18));
    const moving = Math.hypot(goal[0], goal[2]) > 0.5, lim = maxDeg * Math.PI / 180;
    const want = moving ? Math.max(-lim, Math.min(lim, M.wrap(Math.atan2(goal[0], goal[2]) - this.heading))) : 0;
    [this.x, this.v] = damperDecay(this.x - want, this.v, 0.09, dt); this.x += want;
    if (Math.abs(this.x) < 1e-3) return local;
    const out = new Float64Array(local);
    let pw = M.Q(w, this.sk.parents[this.chain[0][0]]);
    for (const [j, f] of this.chain) {
      const lq = M.qmul(M.qconj(pw), M.qmul(M.qYaw(this.x * f), M.qmul(pw, M.Q(out, j))));
      M.setQ(out, j, M.qnorm(lq)); pw = M.qmul(pw, lq);
    }
    return out;
  }
}

// ------------------------------------------------------------------ controller
// Animation-authoritative: every target is predicted from the root state of the frame the new segment
// starts from (never from a free-running gameplay position), so the model is never asked to catch up.
// Slow devices: the buffer is extended by looping the last gait cycle instead of freezing, segments get
// longer as latency grows, and steady motion only replans when the buffer runs low, so the worker is
// usually free the moment the input changes. Direction changes are answered at once by rotating the
// buffered future (steering warp); the warp is replayed onto in-flight results so they stay continuous.
const yawRot = (v, a) => M.qrot(M.qYaw(a), v);
function rotateFrame(f, pivot, ang) {
  const out = new Float64Array(f), p = M.P(f, 0), d = yawRot([p[0] - pivot[0], 0, p[2] - pivot[2]], ang);
  M.setP(out, 0, [pivot[0] + d[0], p[1], pivot[2] + d[2]]);
  M.setQ(out, 0, M.qnorm(M.qmul(M.qYaw(ang), M.Q(f, 0))));
  return out;
}
// a warp rotates frames after startAbs about pivot by sign * min(total, rate * elapsed): a ramp, then rigid
const warpAngle = (w, abs) => (abs <= w.startAbs ? 0 : w.sign * Math.min(w.total, w.rate * (abs - w.startAbs) / FPS));
const LEG_ARM = ['LeftLeg', 'LeftShin', 'LeftFoot', 'RightLeg', 'RightShin', 'RightFoot', 'LeftArm', 'LeftForeArm', 'RightArm', 'RightForeArm', 'Spine1', 'Spine2'];

export class Controller {
  constructor({ sk, lib, limits, settings, generate, actions }) {
    this.sk = sk; this.lib = lib; this.limits = limits; this.S = settings; this.generate = generate; this.actions = actions;
    this.rand = M.rng(7);
    this.simPos = [0, 0, 0]; this.simVel = [0, 0, 0]; this.simAcc = [0, 0, 0]; this.simYaw = 0;
    const idle = place(lib.windows[actions.idle.window], this.simPos, 0).map((f) => applyLimits(limits, f, true));
    this.buffer = [...idle, ...Array.from({ length: 8 }, () => new Float64Array(idle[3]))];
    this.contacts = this.buffer.map(() => [1, 1, 1, 1]);
    this.time = 3 * DT; this.pending = false; this.framesSincePlan = 1e6; this.lastGoal = [0, 0, 0];
    this.queue = []; this.commitEnd = -1; this.latency = 0.3; this.latHist = [];
    this.base = 0; // frames dropped from the front of the buffer so far (absolute index = base + local index)
    this.warps = []; this.warpSeq = 0; this.planYaw = null; this.warpSincePlan = 0; this.sinceChange = 1e6;
    this.stats = { generations: 0, late: 0, lastLatency: 0, lastTokens: 0, failures: 0, error: '', extended: 0, steering: 0 };
    this.inert = new Inertializer(sk.names.length, 0.07);
    this.feet = new FootLocker(sk, () => ({ on: 0.55, off: 0.35, max: 0.12, blendIn: 0.06, blendOut: 0.1 }));
    this.fingers = new FingerAnimator(sk, this.buffer[0], () => ({ curlIdle: 6, curlWalk: 12, curlRun: this.S().fingerCurlRun,
      follow: this.S().fingerFollow, variation: 3 }));
    this.look = new LookAt(sk);
    this.cycleJoints = LEG_ARM.map((n) => sk.names.indexOf(n)).filter((j) => j > 0);
    this.iR = sk.names.indexOf('RightLeg'); this.iL = sk.names.indexOf('LeftLeg');
    this.display = null; this.prevDisplay = null;
    this.events = []; // e.g. {type:'sit', pos, yaw, seat}
  }
  get frame() { return this.time * FPS; }
  goalVelocity(cmd) {
    const s = this.S(), speed = { walk: s.walkSpeed, run: s.runSpeed, sprint: s.sprintSpeed, crouch: s.crouchSpeed }[cmd.gait] ?? s.runSpeed;
    const mag = Math.min(1, Math.hypot(cmd.move[0], cmd.move[1]));
    if (mag < 1e-3) return [0, 0, 0];
    const n = Math.hypot(cmd.move[0], cmd.move[1]);
    return [cmd.move[0] / n * speed * mag, 0, cmd.move[1] / n * speed * mag];
  }
  // spring-damper velocity with acceleration caps (gameplay feel); integrates position
  step(state, goal, dt) {
    const s = this.S(), hl = s.responsiveness * s.inertia;
    if (M.vdot(goal, state.v) < 0) goal = M.vadd(goal, M.vscale(state.v, s.momentum));
    let [vn, an] = springVelocity(state.v, state.a, goal, hl, dt);
    let dv = M.vsub(vn, state.v);
    const cap = (M.vlen(vn) > M.vlen(state.v) ? s.acceleration : s.deceleration) * dt;
    if (M.vlen(dv) > cap) { dv = M.vscale(dv, cap / M.vlen(dv)); an = M.vscale(dv, 1 / Math.max(dt, 1e-6)); }
    state.v = M.vadd(state.v, dv); state.a = an; state.p = M.vadd(state.p, M.vscale(state.v, dt));
  }
  simulate(goal, dt) { // smoothed command velocity (HUD, fingers, replan trigger)
    const st = { p: this.simPos, v: this.simVel, a: this.simAcc };
    this.step(st, goal, dt);
    this.simPos = st.p; this.simVel = st.v; this.simAcc = st.a;
    if (Math.hypot(this.simVel[0], this.simVel[2]) > 0.3) this.simYaw = Math.atan2(this.simVel[0], this.simVel[2]);
  }
  rootState(k) { // position, horizontal velocity and heading of the buffer at local frame k
    const b = this.buffer, k0 = Math.max(0, k - 2), p = M.P(b[k], 0), q = M.P(b[k0], 0), n = Math.max(1, k - k0);
    const w = M.fk(b[k], this.sk.parents);
    return { p: [p[0], 0, p[2]], v: [(p[0] - q[0]) * FPS / n, 0, (p[2] - q[2]) * FPS / n], yaw: headingFromHips(M.P(w, this.iR), M.P(w, this.iL)) };
  }
  predictFrom(k, goal, horizon) {
    const r = this.rootState(k), st = { p: r.p, v: r.v, a: [0, 0, 0] };
    for (let i = 0; i < Math.max(1, Math.round(horizon * FPS)); i++) this.step(st, goal, DT);
    const sp = Math.hypot(st.v[0], st.v[2]);
    const yaw = sp > 0.3 ? Math.atan2(st.v[0], st.v[2]) : r.yaw;
    return [st.p, st.v, yaw, r];
  }
  groundYaw(pose) {
    const w = M.fk(pose, this.sk.parents);
    const h = M.P(w, 0);
    return [[h[0], 0, h[2]], headingFromHips(M.P(w, this.iR), M.P(w, this.iL))];
  }
  trigger(name) { if (this.actions[name]) { this.queue.push(name); this.framesSincePlan = 1e6; } }
  noteLatency(lat) { // EMA, but never below the median of the last 8
    this.latHist.push(lat); if (this.latHist.length > 8) this.latHist.shift();
    const srt = this.latHist.slice().sort((a, b) => b - a);
    this.latency = Math.max(0.7 * this.latency + 0.3 * lat, srt[Math.floor(srt.length / 2)]); // late results are inertialized, not frozen
  }

  plan(cmd, goal) {
    const s = this.S(), now = this.frame;
    const lead = Math.ceil(this.latency * FPS) + s.replanLead;
    let k = Math.floor(now) + Math.max(1, lead);
    k = Math.min(Math.max(k, 3), this.buffer.length - 1);
    if (!this.queue.length) this.presteer(goal, k);
    const context = this.buffer.slice(k - 3, k + 1);
    let allowed = []; for (let t = s.minTokens; t <= s.maxTokens; t++) allowed.push(t);
    // the segment must outlast the measured latency; steady input gets long segments (worker stays free)
    const needT = Math.ceil((this.latency * FPS * 1.3 + s.replanLead + 6) / 4);
    const steady = this.sinceChange > 0.6 && !this.queue.length;
    const Th = steady ? MAX_TOKENS : Math.round((s.predictionHorizon * FPS - (k - now) + 7) / 4);
    const T = Math.min(MAX_TOKENS, Math.max(s.minTokens, MIN_TOKENS, needT, Th));
    const horizon = (validFrames(T) - 1) / FPS;
    let target, action = this.queue.shift() || null;
    const a = action ? this.actions[action] : null;
    if (a) {
      const win = this.lib.windows[a.window];
      if (a.anchor === 'clip') {
        const [pc, yc] = this.groundYaw(this.buffer[k]);
        const d = [a.rootk[0] - a.root0[0], a.rootk[1] - a.root0[1]], y0 = a.root0[2];
        const lx = Math.cos(-y0) * d[0] + Math.sin(-y0) * d[1], lz = -Math.sin(-y0) * d[0] + Math.cos(-y0) * d[1];
        const fp = [pc[0] + Math.cos(yc) * lx + Math.sin(yc) * lz, 0, pc[2] - Math.sin(yc) * lx + Math.cos(yc) * lz];
        const fyaw = M.wrap(yc + (a.rootk[2] - y0));
        target = place(win, fp, fyaw);
      } else {
        const [fp, , fyaw] = this.predictFrom(k, goal, horizon);
        target = place(win, fp, fyaw);
        if (action === 'sit') this.events.push({ type: 'sit', pos: fp, yaw: fyaw, seat: Math.max(0.3, M.P(target[0], 0)[1] - 0.12) });
      }
      allowed = allowed.filter((t) => t >= Math.min(needT, MAX_TOKENS));
      if (!allowed.length) allowed = [MAX_TOKENS];
    } else {
      const [fp, fv, fyaw] = this.predictFrom(k, goal, horizon);
      const fwd = [Math.sin(fyaw), Math.cos(fyaw)], right = [fwd[1], -fwd[0]];
      const vl = [fv[0] * right[0] + fv[2] * right[1], fv[0] * fwd[0] + fv[2] * fwd[1]];
      const w = this.lib.match(vl, 0, cmd.gait === 'crouch', s.variation, this.rand);
      target = place(this.lib.windows[w], fp, fyaw);
      allowed = s.durationMode === 'horizon' ? [T] : allowed.filter((t) => t >= Math.min(needT, MAX_TOKENS));
    }
    const info = { kAbs: this.base + k, t0: performance.now(), action, warpSeq: this.warpSeq };
    this.pending = true;
    this.framesSincePlan = 0; this.lastGoal = goal.slice();
    this.planYaw = Math.hypot(goal[0], goal[2]) > 0.5 && !a ? Math.atan2(goal[0], goal[2]) : null; this.warpSincePlan = 0;
    this.generate({ context, target, allowed, anatomical: s.anatomical, velocityMatching: s.velocityMatching })
      .then((res) => this.splice(res, info))
      .catch((e) => { this.stats.failures++; this.stats.error = String(e); })
      .finally(() => { this.pending = false; });
  }

  splice(res, info) {
    const lat = (performance.now() - info.t0) / 1000;
    this.noteLatency(lat);
    Object.assign(this.stats, { generations: this.stats.generations + 1, lastLatency: lat, lastTokens: res.tokens });
    const k = info.kAbs - this.base, now = this.frame, before = this.display;
    if (k < 3 || k > this.buffer.length) return; // stale (cannot happen with one request in flight)
    let frames = res.frames;
    for (const w of this.warps) if (w.seq > info.warpSeq) frames = frames.map((f, i) => { const a = warpAngle(w, info.kAbs + i); return a ? rotateFrame(f, w.pivot, a) : f; });
    this.buffer = [...this.buffer.slice(0, k), ...frames];
    this.contacts = [...this.contacts.slice(0, k), ...res.contacts];
    const drop = Math.max(0, Math.floor(now) - 8);
    if (drop > 0) { this.buffer = this.buffer.slice(drop); this.contacts = this.contacts.slice(drop); this.time -= drop * DT; this.commitEnd -= drop; this.base += drop; }
    this.warps = []; // only one request is ever in flight, and it has just been spliced
    if (info.action) this.commitEnd = (k - drop) + res.frames.length - 1 - (this.latency * FPS + this.S().replanLead + 2);
    if (this.frame > k - drop && before) {
      this.stats.late++;
      this.inert.transition(before, this.prevDisplay || before, this.sample(this.time), this.sample(this.time - DT), DT);
    }
  }
  // append a copy of the last gait cycle (re-rooted, seam blended) so playback never stalls on a late result
  extend() {
    const b = this.buffer, e = b.length - 1, J = this.sk.names.length;
    let best = -1, bestCost = Infinity;
    for (let s = Math.max(1, e - 44); s <= e - 14; s++) {
      let c = 0;
      for (const j of this.cycleJoints) c += M.vlen(M.qToScaledAxis(M.qmul(M.Q(b[e], j), M.qconj(M.Q(b[s], j)))));
      const ve = M.vsub(M.P(b[e], 0), M.P(b[e - 1], 0)), vs = M.vsub(M.P(b[s], 0), M.P(b[s - 1], 0));
      c += 4 * Math.abs(M.vlen(ve) - M.vlen(vs)) * FPS;
      if (c < bestCost) { bestCost = c; best = s; }
    }
    if (best < 0) { b.push(new Float64Array(b[e])); this.contacts.push(this.contacts[e].slice()); return; }
    const ge = this.groundYaw(b[e]), gs = this.groundYaw(b[best]), dyaw = M.wrap(ge[1] - gs[1]);
    const seam = [], pe = M.P(b[e], 0), ps = M.P(b[best], 0);
    const rq0 = M.qmul(M.qYaw(dyaw), M.Q(b[best], 0));
    seam.push(M.qToScaledAxis(M.qmul(M.Q(b[e], 0), M.qconj(rq0))));
    for (let j = 1; j < J; j++) seam.push(M.qToScaledAxis(M.qmul(M.Q(b[e], j), M.qconj(M.Q(b[best], j)))));
    const n = e - best, N = Math.min(10, n);
    for (let i = 1; i <= n; i++) {
      const src = b[best + i], out = new Float64Array(src), w = i < N ? (1 - i / N) ** 2 : 0;
      const p = M.P(src, 0), d = yawRot([p[0] - ps[0], 0, p[2] - ps[2]], dyaw);
      M.setP(out, 0, [pe[0] + d[0], p[1] + w * (pe[1] - ps[1]), pe[2] + d[2]]);
      M.setQ(out, 0, M.qnorm(M.qmul(M.qFromScaledAxis(M.vscale(seam[0], w)), M.qmul(M.qYaw(dyaw), M.Q(src, 0)))));
      if (w > 0) for (let j = 1; j < J; j++) M.setQ(out, j, M.qnorm(M.qmul(M.qFromScaledAxis(M.vscale(seam[j], w)), M.Q(src, j))));
      b.push(out); this.contacts.push(this.contacts[best + i].slice());
    }
    this.stats.extended++;
  }
  // rotate the buffered future (frames after the playhead) toward the commanded direction
  warp(w, record) {
    const i0 = Math.floor(this.frame);
    for (let i = i0 + 1; i < this.buffer.length; i++) { const a = warpAngle(w, this.base + i); if (a) this.buffer[i] = rotateFrame(this.buffer[i], w.pivot, a); }
    if (record) { this.warps.push({ ...w, seq: ++this.warpSeq }); if (this.warps.length > 400) this.warps.shift(); }
    this.stats.steering += w.total;
  }
  movingAt(i) {
    if (i < 2 || i >= this.buffer.length) return null;
    const a = M.P(this.buffer[i], 0), c = M.P(this.buffer[i - 2], 0), v = [(a[0] - c[0]) * FPS / 2, (a[2] - c[2]) * FPS / 2];
    return Math.hypot(v[0], v[1]) >= 0.8 ? Math.atan2(v[0], v[1]) : null;
  }
  // input changed while a request is in flight: keep turning the future; the warp is replayed on its result
  steer(goal, dt) {
    const s = this.S(), rate = (s.steering ?? 200) * Math.PI / 180;
    if (rate <= 0 || this.planYaw === null || Math.hypot(goal[0], goal[2]) < 0.5 || this.movingAt(Math.floor(this.frame)) === null) return;
    const err = M.wrap(Math.atan2(goal[0], goal[2]) - (this.planYaw + this.warpSincePlan));
    if (Math.abs(err) < 0.05) return;
    const maxW = 90 * Math.PI / 180, step = Math.max(-rate * dt, Math.min(rate * dt, err));
    const d = Math.max(-maxW, Math.min(maxW, this.warpSincePlan + step)) - this.warpSincePlan;
    if (Math.abs(d) < 1e-6) return;
    this.warp({ pivot: M.P(this.sample(this.time), 0), total: Math.abs(d), rate: Infinity, sign: Math.sign(d), startAbs: this.base + Math.floor(this.frame) + 0.5 }, true);
    this.warpSincePlan += d;
  }
  // at plan time: start turning now, so the motion until the new segment begins already heads the right way
  presteer(goal, k) {
    const s = this.S(), rate = (s.steering ?? 200) * Math.PI / 180, hk = this.movingAt(k);
    if (rate <= 0 || hk === null || Math.hypot(goal[0], goal[2]) < 0.5 || this.movingAt(Math.floor(this.frame)) === null) return;
    const err = M.wrap(Math.atan2(goal[0], goal[2]) - hk);
    if (Math.abs(err) < 0.05 || Math.abs(err) > 135 * Math.PI / 180) return; // reversals are left to the model
    const total = Math.min(Math.abs(err), rate * (k - this.frame) / FPS, 90 * Math.PI / 180);
    this.warp({ pivot: M.P(this.sample(this.time), 0), total, rate, sign: Math.sign(err), startAbs: this.base + this.frame }, false);
  }
  sample(t) {
    const f = Math.min(Math.max(t * FPS, 0), this.buffer.length - 1), i0 = Math.floor(f), i1 = Math.min(i0 + 1, this.buffer.length - 1);
    return M.blendPose(this.buffer[i0], this.buffer[i1], f - i0);
  }
  sampleContacts(t) {
    const f = Math.min(Math.max(t * FPS, 0), this.contacts.length - 1), i0 = Math.floor(f), i1 = Math.min(i0 + 1, this.contacts.length - 1), u = f - i0;
    return this.contacts[i0].map((c, n) => c * (1 - u) + this.contacts[i1][n] * u);
  }

  update(cmd, dt) {
    const s = this.S();
    if (cmd.action) this.trigger(cmd.action);
    const goal = this.goalVelocity(cmd);
    this.simulate(goal, dt);
    const changed = M.vlen(M.vsub(goal, this.lastGoal)) > s.inputReplanThreshold;
    this.sinceChange = changed ? 0 : this.sinceChange + dt;
    const remaining = this.buffer.length - 1 - this.frame;
    const committed = this.frame < this.commitEnd;
    const low = remaining < this.latency * FPS * 1.15 + s.replanLead + 4;
    const fresh = this.sinceChange < 0.6 && this.framesSincePlan >= s.replanInterval; // keep refining right after input changes
    const need = low || (!committed && ((changed && this.framesSincePlan >= 2) || fresh || this.queue.length));
    if (need && !this.pending) this.plan(cmd, goal);
    if (!committed) this.steer(goal, dt);
    this.inert.halflife = Math.max(s.transitionDuration / 2.5, 1e-3);
    while (this.buffer.length - 1 - (this.frame + dt * s.playbackRate * FPS) < 2) this.extend();
    this.time = Math.min(this.time + dt * s.playbackRate, (this.buffer.length - 1) * DT);
    this.framesSincePlan += dt * FPS; // game frames, independent of the render rate
    let pose = this.sample(this.time);
    this.simPos = M.P(pose, 0);
    if (this.inert.active) pose = this.inert.apply(pose, dt);
    if (s.footLocking) pose = this.feet.apply(pose, this.sampleContacts(this.time), dt);
    if (s.anatomical) pose = applyLimits(this.limits, pose);
    if (s.fingers) pose = this.fingers.apply(pose, Math.hypot(this.simVel[0], this.simVel[2]), dt);
    if (s.lookAt ?? true) pose = this.look.apply(pose, goal, dt, 70);
    if (s.smoothing > 0 && this.display) {
      const alpha = 1 - Math.pow(s.smoothing, dt * 30), out = new Float64Array(pose);
      for (let j = 1; j < this.sk.names.length; j++) M.setQ(out, j, M.qSlerp(M.Q(this.display, j), M.Q(pose, j), alpha));
      pose = out;
    }
    this.prevDisplay = this.display; this.display = pose;
    return pose;
  }
}
