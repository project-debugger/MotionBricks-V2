// Continuous mocap clips (30 fps, plus left/right mirrored copies) used to pick target windows.
// Every frame is a candidate window start, labelled with root velocity, yaw rate, hip height and the
// gait phase (from the left-right foot separation), so targets can be chosen to continue the stride
// the character is actually in instead of snapping to a pose from another part of the cycle.
import * as M from './math3d.js';
import { applyLimits, headingFromHips } from './codec.js';

const FPS = 30;

export function legIndices(sk) {
  const ix = (n) => sk.names.indexOf(n);
  return { lleg: ix('LeftLeg'), lshin: ix('LeftShin'), lfoot: ix('LeftFoot'), rleg: ix('RightLeg'), rshin: ix('RightShin'), rfoot: ix('RightFoot') };
}
// hips ground point, heading and both feet (world) from a local pose, with a 3-bone FK per leg
export function legState(pose, L) {
  const p0 = M.P(pose, 0), q0 = M.Q(pose, 0);
  const chain = (a, b, c) => {
    let [p, q] = M.tfMul(p0, q0, M.P(pose, a), M.Q(pose, a)); const pa = p;
    [p, q] = M.tfMul(p, q, M.P(pose, b), M.Q(pose, b));
    [p] = M.tfMul(p, q, M.P(pose, c), M.Q(pose, c));
    return [pa, p];
  };
  const [la, lf] = chain(L.lleg, L.lshin, L.lfoot), [ra, rf] = chain(L.rleg, L.rshin, L.rfoot);
  const yaw = headingFromHips(ra, la);
  return { g: [p0[0], p0[2]], h: p0[1], yaw, lf, rf };
}
// forward separation of the feet (left minus right) in the heading frame
export const footSep = (s) => (s.lf[0] - s.rf[0]) * Math.sin(s.yaw) + (s.lf[2] - s.rf[2]) * Math.cos(s.yaw);
// fractional indices where the series crosses from negative to non-negative
export function upCrossings(d) {
  const out = [];
  for (let i = 1; i < d.length; i++) if (d[i - 1] < 0 && d[i] >= 0) out.push(i - 1 + d[i - 1] / (d[i - 1] - d[i]));
  return out;
}

export class ClipLibrary {
  constructor(sk, meta, buffer, limits) {
    this.sk = sk; this.limits = limits; this.meta = meta;
    const J = sk.names.length, jm = meta.joints.map((n) => sk.names.indexOf(n));
    this.base = applyLimits(limits, new Float64Array(sk.reference), true);
    this.L = legIndices(sk);
    this.clips = meta.clips.map((c) => {
      const F = c.frames, root = new Float32Array(buffer, c.offset, F * 3), qi = new Int16Array(buffer, c.offset + F * 12, F * jm.length * 4);
      return { ...c, F, root, qi, jm, J };
    });
    this.cand = []; // candidate window starts
    for (let ci = 0; ci < this.clips.length; ci++) {
      const c = this.clips[ci], st = [];
      for (let f = 0; f < c.F; f++) st.push(legState(this.frame(ci, f), this.L));
      const d = st.map(footSep), ups = upCrossings(d);
      for (let f = 1; f + 4 < c.F; f++) {
        const a = st[f - 1], b = st[f + 1], yaw = st[f].yaw;
        const vw = [(b.g[0] - a.g[0]) * FPS / 2, (b.g[1] - a.g[1]) * FPS / 2];
        const fwd = [Math.sin(yaw), Math.cos(yaw)], right = [fwd[1], -fwd[0]];
        const v = [vw[0] * right[0] + vw[1] * right[1], vw[0] * fwd[0] + vw[1] * fwd[1]];
        const yr = M.wrap(b.yaw - a.yaw) * FPS / 2;
        let phase = null, period = null;
        for (let u = 1; u < ups.length; u++) if (ups[u - 1] <= f && f < ups[u]) { period = ups[u] - ups[u - 1]; phase = (f - ups[u - 1]) / period; }
        if (period !== null && (period < 12 || period > 60)) { phase = null; period = null; }
        this.cand.push({ ci, f, v, yr, h: st[f].h, speed: Math.hypot(v[0], v[1]), phase, period, cat: c.category, crouch: st[f].h < 0.75, g: st[f].g, yaw });
      }
    }
    this.cache = new Map();
  }
  frame(ci, f) { // local pose of clip frame f (relaxed pose for the joints the network does not predict)
    const c = this.clips[ci], out = new Float64Array(this.base), n = c.jm.length;
    out[0] = c.root[f * 3]; out[1] = c.root[f * 3 + 1]; out[2] = c.root[f * 3 + 2];
    for (let k = 0; k < n; k++) {
      const o = (f * n + k) * 4, j = c.jm[k];
      M.setQ(out, j, M.qnorm([c.qi[o] / 32767, c.qi[o + 1] / 32767, c.qi[o + 2] / 32767, c.qi[o + 3] / 32767]));
    }
    return out;
  }
  // canonical 4-frame window (frame f's root ground point at the origin, facing +Z)
  window(i) {
    if (this.cache.has(i)) return this.cache.get(i);
    const e = this.cand[i], inv = M.qYaw(-e.yaw), frames = [];
    for (let k = 0; k < 4; k++) {
      const p = this.frame(e.ci, e.f + k), r = M.P(p, 0);
      const d = M.qrot(inv, [r[0] - e.g[0], 0, r[2] - e.g[1]]);
      M.setP(p, 0, [d[0], r[1], d[2]]); M.setQ(p, 0, M.qnorm(M.qmul(inv, M.Q(p, 0))));
      frames.push(applyLimits(this.limits, p, true));
    }
    if (this.cache.size > 400) this.cache.delete(this.cache.keys().next().value);
    this.cache.set(i, frames);
    return frames;
  }
  // median stride period (frames) of moving candidates near a speed
  periodFor(speed, crouch) {
    const ps = this.cand.filter((e) => e.period !== null && e.crouch === crouch && Math.abs(e.speed - speed) < 0.5).map((e) => e.period).sort((a, b) => a - b);
    return ps.length ? ps[ps.length >> 1] : null;
  }
  match({ v, yr = 0, crouch = false, phase = null, variation = 0, rand = Math.random, moving = true }) {
    let best = Infinity;
    const cost = this.cand.map((e) => {
      let c = (e.v[0] - v[0]) ** 2 + (e.v[1] - v[1]) ** 2 + 0.4 * (e.yr - yr) ** 2;
      if (e.crouch !== crouch) c += 1e3;
      if (moving && phase !== null) {
        if (e.phase === null) c += 0.5;
        else { const dp = Math.abs(((e.phase - phase) % 1 + 1.5) % 1 - 0.5); c += 6 * dp * dp; }
      }
      if (c < best) best = c;
      return c;
    });
    if (variation > 0) {
      const band = best + variation * 0.25 + 1e-9, pool = [];
      cost.forEach((c, i) => { if (c <= band) pool.push(i); });
      return pool[Math.floor(rand() * pool.length)];
    }
    return cost.indexOf(best);
  }
}
