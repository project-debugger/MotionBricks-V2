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

// ------------------------------------------------------------------ controller
export class Controller {
  constructor({ sk, lib, limits, settings, generate, actions }) {
    this.sk = sk; this.lib = lib; this.limits = limits; this.S = settings; this.generate = generate; this.actions = actions;
    this.rand = M.rng(7);
    this.simPos = [0, 0, 0]; this.simVel = [0, 0, 0]; this.simAcc = [0, 0, 0]; this.simYaw = 0;
    const idle = place(lib.windows[actions.idle.window], this.simPos, 0).map((f) => applyLimits(limits, f, true));
    this.buffer = [...idle, ...Array(8).fill(idle[3])];
    this.contacts = this.buffer.map(() => [1, 1, 1, 1]);
    this.time = 3 * DT; this.pending = false; this.framesSincePlan = 1e6; this.lastGoal = [0, 0, 0];
    this.queue = []; this.commitEnd = -1; this.latency = 0.3;
    this.stats = { generations: 0, late: 0, lastLatency: 0, lastTokens: 0, failures: 0, error: '' };
    this.inert = new Inertializer(sk.names.length, 0.07);
    this.feet = new FootLocker(sk, () => ({ on: 0.55, off: 0.35, max: 0.12, blendIn: 0.06, blendOut: 0.1 }));
    this.fingers = new FingerAnimator(sk, this.buffer[0], () => ({ curlIdle: 6, curlWalk: 12, curlRun: this.S().fingerCurlRun,
      follow: this.S().fingerFollow, variation: 3 }));
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
  simulate(goal, dt) {
    const s = this.S(), hl = s.responsiveness * s.inertia;
    if (M.vdot(goal, this.simVel) < 0) goal = M.vadd(goal, M.vscale(this.simVel, s.momentum));
    let [vn, an] = springVelocity(this.simVel, this.simAcc, goal, hl, dt);
    let dv = M.vsub(vn, this.simVel);
    const cap = (M.vlen(vn) > M.vlen(this.simVel) ? s.acceleration : s.deceleration) * dt;
    if (M.vlen(dv) > cap) { dv = M.vscale(dv, cap / M.vlen(dv)); an = M.vscale(dv, 1 / Math.max(dt, 1e-6)); }
    this.simVel = M.vadd(this.simVel, dv); this.simAcc = an;
    this.simPos = M.vadd(this.simPos, M.vscale(this.simVel, dt));
    if (Math.hypot(this.simVel[0], this.simVel[2]) > 0.3) {
      const target = Math.atan2(this.simVel[0], this.simVel[2]), err = M.wrap(target - this.simYaw);
      const y = (4 * LN2 / (s.facingResponsiveness * s.inertia)) / 2, lim = s.turnRate * Math.PI / 180 * dt;
      this.simYaw = M.wrap(this.simYaw + Math.max(-lim, Math.min(lim, err * (1 - Math.exp(-y * dt)))));
    }
  }
  predict(horizon, goal) {
    const s = this.S(), hl = s.responsiveness * s.inertia;
    let p = this.simPos.slice(), v = this.simVel.slice(), a = this.simAcc.slice();
    for (let i = 0; i < Math.max(1, Math.round(horizon * FPS)); i++) { [v, a] = springVelocity(v, a, goal, hl, DT); p = M.vadd(p, M.vscale(v, DT)); }
    const yaw = Math.hypot(goal[0], goal[2]) > 0.3 ? Math.atan2(goal[0], goal[2]) : this.simYaw;
    return [p, v, yaw];
  }
  groundYaw(pose) {
    const w = M.fk(pose, this.sk.parents), r = this.sk.names.indexOf('RightLeg'), l = this.sk.names.indexOf('LeftLeg');
    const h = M.P(w, 0);
    return [[h[0], 0, h[2]], headingFromHips(M.P(w, r), M.P(w, l))];
  }
  trigger(name) { if (this.actions[name]) { this.queue.push(name); this.framesSincePlan = 1e6; } }

  plan(cmd, goal) {
    const s = this.S(), now = this.frame;
    const lead = Math.ceil(this.latency * FPS) + s.replanLead;
    let k = Math.floor(now) + Math.max(1, lead);
    k = Math.min(Math.max(k, 3), this.buffer.length - 1);
    const context = this.buffer.slice(k - 3, k + 1);
    let allowed = []; for (let t = s.minTokens; t <= s.maxTokens; t++) allowed.push(t);
    // segment must outlast the measured latency (slow phones) and cover the horizon
    const needT = Math.ceil((this.latency * FPS * 1.3 + s.replanLead + 6) / 4);
    let horizon = s.predictionHorizon;
    const Th = Math.round((horizon * FPS - (k - now) + 7) / 4);
    const T = Math.min(MAX_TOKENS, Math.max(s.minTokens, MIN_TOKENS, needT, Th));
    horizon = ((k - now) + validFrames(T) - 1) / FPS;
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
        this.simPos = fp.slice(); this.simVel = [0, 0, 0]; this.simAcc = [0, 0, 0]; this.simYaw = fyaw;
        target = place(win, fp, fyaw);
      } else {
        const [fp] = this.predict(horizon, goal);
        target = place(win, fp, this.simYaw);
        if (action === 'sit') this.events.push({ type: 'sit', pos: fp, yaw: this.simYaw, seat: Math.max(0.3, M.P(target[0], 0)[1] - 0.12) });
      }
      allowed = allowed.filter((t) => t >= Math.min(needT, MAX_TOKENS));
      if (!allowed.length) allowed = [MAX_TOKENS];
    } else {
      const [fp, fv, fyaw] = this.predict(horizon, goal);
      const fwd = [Math.sin(fyaw), Math.cos(fyaw)], right = [fwd[1], -fwd[0]];
      const vl = [fv[0] * right[0] + fv[2] * right[1], fv[0] * fwd[0] + fv[2] * fwd[1]];
      const w = this.lib.match(vl, 0, cmd.gait === 'crouch', s.variation, this.rand);
      target = place(this.lib.windows[w], fp, fyaw);
      allowed = s.durationMode === 'horizon' ? [T] : allowed.filter((t) => t >= Math.min(needT, MAX_TOKENS));
    }
    const info = { k, t0: performance.now(), action };
    this.pending = true;
    this.framesSincePlan = 0; this.lastGoal = goal.slice();
    this.generate({ context, target, allowed, anatomical: s.anatomical, velocityMatching: s.velocityMatching })
      .then((res) => this.splice(res, info))
      .catch((e) => { this.stats.failures++; this.stats.error = String(e); })
      .finally(() => { this.pending = false; });
  }

  splice(res, info) {
    const lat = (performance.now() - info.t0) / 1000;
    this.latency = 0.7 * this.latency + 0.3 * lat;
    Object.assign(this.stats, { generations: this.stats.generations + 1, lastLatency: lat, lastTokens: res.tokens });
    const k = info.k, now = this.frame, before = this.display;
    this.buffer = [...this.buffer.slice(0, k), ...res.frames];
    this.contacts = [...this.contacts.slice(0, k), ...res.contacts];
    const drop = Math.max(0, Math.floor(now) - 8);
    if (drop > 0) { this.buffer = this.buffer.slice(drop); this.contacts = this.contacts.slice(drop); this.time -= drop * DT; this.commitEnd -= drop; }
    if (info.action) this.commitEnd = (k - drop) + res.frames.length - 1 - (this.latency * FPS + this.S().replanLead + 2);
    if (this.frame > k - drop && before) {
      this.stats.late++;
      this.inert.transition(before, this.prevDisplay || before, this.sample(this.time), this.sample(this.time - DT), DT);
    }
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
    const remaining = this.buffer.length - 1 - this.frame;
    const changed = M.vlen(M.vsub(goal, this.lastGoal)) > s.inputReplanThreshold;
    const committed = this.frame < this.commitEnd;
    const need = remaining < this.latency * FPS + s.replanLead + 2 ||
      (!committed && (this.framesSincePlan >= s.replanInterval || (changed && this.framesSincePlan >= 2) || this.queue.length));
    if (need && !this.pending) this.plan(cmd, goal);
    this.inert.halflife = Math.max(s.transitionDuration / 2.5, 1e-3);
    this.time = Math.min(this.time + dt * s.playbackRate, (this.buffer.length - 1) * DT);
    this.framesSincePlan += dt * FPS; // game frames, independent of the render rate
    let pose = this.sample(this.time);
    if (this.inert.active) pose = this.inert.apply(pose, dt);
    if (s.footLocking) pose = this.feet.apply(pose, this.sampleContacts(this.time), dt);
    if (s.anatomical) pose = applyLimits(this.limits, pose);
    if (s.fingers) pose = this.fingers.apply(pose, Math.hypot(this.simVel[0], this.simVel[2]), dt);
    if (s.smoothing > 0 && this.display) {
      const alpha = 1 - Math.pow(s.smoothing, dt * 30), out = new Float64Array(pose);
      for (let j = 1; j < this.sk.names.length; j++) M.setQ(out, j, M.qSlerp(M.Q(this.display, j), M.Q(pose, j), alpha));
      pose = out;
    }
    this.prevDisplay = this.display; this.display = pose;
    return pose;
  }
}
