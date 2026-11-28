// Transition bank: starts, stops, turns, gait changes and actions generated ahead of time by the same
// SOMA network and pipeline (build/tools/gen_bank.html). On an input change the controller jumps to the
// best-matching bank frame immediately (motion matching on feet, root velocity and the desired path),
// so the response begins on the next frame; live SOMA generation then continues from inside it.
import * as M from './math3d.js';
import { applyLimits } from './codec.js';
import { legState } from './library.js';

const FPS = 30, H = [10, 20, 30];
// feature groups: path (6), facing along the path (6), feet positions (6), feet velocities (6), root velocity (2)
const GROUPS = [[0, 6, 1.0], [6, 12, 1.4], [12, 18, 0.8], [18, 24, 1.2], [24, 26, 2.0]];
export const NF = 26;
const GAIT_MISMATCH = 2, SAME_FAMILY = 0.8;
export const family = (g) => (g === 'run' || g === 'sprint' ? 'run' : g === 'crouch' || g === 'crouch_stop' ? 'crouch' : g);

// R(-yaw) for the yaw convention facing = (sin yaw, cos yaw): maps the facing direction to +Z
const local = (g, yaw, p) => { const c = Math.cos(yaw), s = Math.sin(yaw), dx = p[0] - g[0], dz = p[1] - g[1]; return [c * dx - s * dz, s * dx + c * dz]; };
const local3 = (g, yaw, p) => { const [x, z] = local(g, yaw, [p[0], p[2]]); return [x, p[1], z]; };
const vlocal = (yaw, v) => { const c = Math.cos(yaw), s = Math.sin(yaw); return [c * v[0] - s * v[1], s * v[0] + c * v[1]]; };

// raw (unnormalised) features of state `cur` (legState), its neighbours and three future points
export function rawFeatures(prev, cur, next, futures, futureYaws) {
  const g = cur.g, yaw = cur.yaw, out = new Float64Array(NF);
  // local frame: rotate the world by -yaw so the character faces +Z (local() applies that rotation)
  const rot = yaw;
  for (let i = 0; i < 3; i++) {
    const [x, z] = local(g, rot, futures[i]); out[i * 2] = x; out[i * 2 + 1] = z;
    const d = M.wrap(futureYaws[i] - yaw); out[6 + i * 2] = Math.sin(d); out[6 + i * 2 + 1] = Math.cos(d);
  }
  const lf = local3(g, rot, cur.lf), rf = local3(g, rot, cur.rf);
  out.set(lf, 12); out.set(rf, 15);
  const half = FPS / 2;
  const lv = [(next.lf[0] - prev.lf[0]) * half, (next.lf[1] - prev.lf[1]) * half, (next.lf[2] - prev.lf[2]) * half];
  const rv = [(next.rf[0] - prev.rf[0]) * half, (next.rf[1] - prev.rf[1]) * half, (next.rf[2] - prev.rf[2]) * half];
  const [lx, lz] = vlocal(rot, [lv[0], lv[2]]), [rx, rz] = vlocal(rot, [rv[0], rv[2]]);
  out[18] = lx; out[19] = lv[1]; out[20] = lz; out[21] = rx; out[22] = rv[1]; out[23] = rz;
  const [vx, vz] = vlocal(rot, [(next.g[0] - prev.g[0]) * half, (next.g[1] - prev.g[1]) * half]);
  out[24] = vx; out[25] = vz;
  return out;
}

export class TransitionBank {
  constructor(sk, meta, buffer, limits, legs) {
    this.sk = sk; this.limits = limits; this.legs = legs;
    this.base = applyLimits(limits, new Float64Array(sk.reference), true);
    const jm = meta.joints.map((n) => sk.names.indexOf(n));
    this.entries = meta.entries.map((e) => {
      const F = e.frames, nq = F * jm.length * 4;
      return { ...e, F, jm, root: new Float32Array(buffer, e.offset, F * 3), qi: new Int16Array(buffer, e.offset + F * 12, nq),
        ct: new Uint8Array(buffer, e.offset + F * 12 + nq * 2, F * 4) };
    });
    this.actions = new Set(this.entries.filter((e) => e.action).map((e) => e.action));
    this.cand = []; const raw = [];
    this.entries.forEach((e, ei) => {
      const st = []; for (let f = 0; f < e.F; f++) st.push(legState(this.pose(ei, f), legs));
      e.states = st;
      for (let f = 1; f + H[2] < e.F; f++) {
        raw.push(rawFeatures(st[f - 1], st[f], st[f + 1], H.map((h) => st[f + h].g), H.map((h) => st[f + h].yaw)));
        const r = raw[raw.length - 1];
        this.cand.push({ ei, f, crouch: st[f].h < 0.75 || (e.gait || '').startsWith('crouch'), speed: Math.hypot(r[24], r[25]) });
      }
    });
    // per-dimension scale; each group then gets a total weight independent of its size
    const n = raw.length, mean = new Float64Array(NF), sd = new Float64Array(NF);
    for (const r of raw) for (let d = 0; d < NF; d++) mean[d] += r[d] / n;
    for (const r of raw) for (let d = 0; d < NF; d++) sd[d] += (r[d] - mean[d]) ** 2 / n;
    this.w = new Float64Array(NF);
    for (const [a, b, wt] of GROUPS) for (let d = a; d < b; d++) this.w[d] = wt / (Math.sqrt(sd[d]) + 1e-3) / Math.sqrt(b - a);
    this.feat = new Float32Array(n * NF);
    raw.forEach((r, i) => { for (let d = 0; d < NF; d++) this.feat[i * NF + d] = r[d] * this.w[d]; });
  }
  pose(ei, f) {
    const e = this.entries[ei], out = new Float64Array(this.base), n = e.jm.length;
    out[0] = e.root[f * 3]; out[1] = e.root[f * 3 + 1]; out[2] = e.root[f * 3 + 2];
    for (let k = 0; k < n; k++) {
      const o = (f * n + k) * 4;
      M.setQ(out, e.jm[k], M.qnorm([e.qi[o] / 32767, e.qi[o + 1] / 32767, e.qi[o + 2] / 32767, e.qi[o + 3] / 32767]));
    }
    return out;
  }
  contacts(ei, f) { const c = this.entries[ei].ct; return [c[f * 4] / 255, c[f * 4 + 1] / 255, c[f * 4 + 2] / 255, c[f * 4 + 3] / 255]; }
  weigh(raw) { const q = new Float64Array(NF); for (let d = 0; d < NF; d++) q[d] = raw[d] * this.w[d]; return q; }
  cost(q, i) { let c = 0; const o = i * NF; for (let d = 0; d < NF; d++) { const x = this.feat[o + d] - q[d]; c += x * x; } return c; }
  // best candidate: action entries only for that action; otherwise locomotion entries of the right posture
  // best candidates, cheapest first (top = how many to return); match() is the single best
  match(q, opts = {}) { const t = this.matchTop(q, opts, 1); return t.length ? t[0] : null; }
  matchTop(q, { crouch = false, action = null, pathOnly = false, speed = null, stopping = null, gait = null } = {}, top = 8) {
    let best = -1, bestCost = Infinity; const keep = [];
    // only enter where the bank is moving about as fast as the character is now (no gait-mismatch blends)
    const tol = speed === null ? Infinity : Math.max(0.8, 0.3 * speed);
    for (let pass = 0; pass < 2 && best < 0; pass++) for (let i = 0; i < this.cand.length; i++) {
      const cnd = this.cand[i], e = this.entries[cnd.ei];
      if (action ? e.action !== action : (e.action || cnd.crouch !== crouch)) continue;
      if (pass === 0 && Math.abs(cnd.speed - speed) > tol) continue;
      const isStop = e.gait === 'stop' || e.gait === 'crouch_stop';
      if (!action && stopping !== null && isStop !== stopping) continue; // intent first: stop entries only for stops
      if (action && cnd.f > 6) continue; // enter an action near its start
      // another gait costs a little within a family (run/sprint) and a lot across (a walking turn is no jog start)
      const gp = !gait || action || stopping || e.gait === gait ? 0 : family(e.gait) === family(gait) ? SAME_FAMILY : GAIT_MISMATCH;
      const c = (pathOnly ? this.costFeet(q, i) : this.cost(q, i)) + gp;
      if (c < bestCost) { bestCost = c; best = i; }
      if (keep.length < top || c < keep[keep.length - 1].cost) {
        keep.push({ ci: i, cost: c }); keep.sort((a, b) => a.cost - b.cost); if (keep.length > top) keep.pop();
      }
    }
    return keep;
  }
  costPath(q, i) { let c = 0; const o = i * NF; for (let d = 0; d < 12; d++) { const x = this.feat[o + d] - q[d]; c += x * x; } return c; }
  costFeet(q, i) { let c = 0; const o = i * NF; for (let d = 12; d < NF; d++) { const x = this.feat[o + d] - q[d]; c += x * x; } return c; }
}
