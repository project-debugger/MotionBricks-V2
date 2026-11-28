// SOMA model I/O, a port of runtime/somart/codec.py (itself a port of NVIDIA's Warp kernels)
// plus the anatomical limits of runtime/somart/anatomy.py.
import * as M from './math3d.js';

export const JOINT_NAMES = ['Hips', 'Spine1', 'Spine2', 'Chest', 'Neck1', 'Neck2', 'Head', 'Jaw', 'LeftEye', 'RightEye',
  'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'LeftHandThumbEnd', 'LeftHandMiddleEnd', 'RightShoulder',
  'RightArm', 'RightForeArm', 'RightHand', 'RightHandThumbEnd', 'RightHandMiddleEnd', 'LeftLeg', 'LeftShin',
  'LeftFoot', 'LeftToeBase', 'RightLeg', 'RightShin', 'RightFoot', 'RightToeBase'];
export const FPS = 30, PPT = 4, MIN_TOKENS = 6, MAX_TOKENS = 16, MAX_VALID = 58, OUT_DIM = 366, ROT_OFF = 92;
export const validFrames = (t) => t * 4 - 6;

export function bind(sk) {
  const idx = (n) => sk.names.indexOf(n);
  const m2p = JOINT_NAMES.map(idx);
  if (m2p.includes(-1)) throw new Error('skeleton is missing SOMA model joints');
  const J = sk.names.length, p2m = new Int32Array(J).fill(-1);
  m2p.forEach((pj, mj) => { const par = sk.parents[pj]; if (par === -1 || p2m[par] >= 0) p2m[pj] = mj; });
  const ppm = new Int32Array(J).fill(-1);
  for (let j = 0; j < J; j++) if (sk.parents[j] >= 0) ppm[j] = p2m[sk.parents[j]];
  return { sk, m2p, p2m, ppm, right: idx('RightLeg'), left: idx('LeftLeg') };
}

export function headingFromHips(r, l) {
  let ax = r[0] - l[0], az = r[2] - l[2];
  const n = Math.hypot(ax, az);
  if (n <= 1e-6) return 0;
  ax /= n; az /= n;
  return Math.atan2(az, -ax);
}

// guides: array of 8 local poses (4 context + 4 target, padded). Returns feeds + alignment.
export function encode(b, guides, nTargets, allowed, heading = [1, 0]) {
  const sk = b.sk, worlds = guides.map((g) => M.fk(g, sk.parents));
  const w3 = worlds[3], hips3 = M.P(w3, 0);
  const cur = headingFromHips(M.P(w3, b.right), M.P(w3, b.left));
  const desired = Math.atan2(heading[0], heading[1]);
  const ap = [hips3[0], 0, hips3[2]], aq = M.qYaw(cur - desired), aqi = M.qconj(aq);
  const g = new Float32Array(8 * 5), lr = new Float32Array(8 * 4), lp = new Float32Array(8 * 267);
  const pos = [], rot = [], head = [];
  for (let f = 0; f < 8; f++) {
    const w = worlds[f], pf = [], rf = [];
    for (let m = 0; m < 30; m++) {
      const j = b.m2p[m];
      pf.push(M.qrot(aqi, M.vsub(M.P(w, j), ap)));
      rf.push(M.qmul(aqi, M.Q(w, j)));
    }
    pos.push(pf); rot.push(rf);
    // heading joints are model joints 22 (LeftLeg) / 26 (RightLeg)
    head.push(headingFromHips(M.qrot(aqi, M.vsub(M.P(w, b.right), ap)), M.qrot(aqi, M.vsub(M.P(w, b.left), ap))));
  }
  for (let f = 0; f < 8; f++) {
    const nx = f === 3 ? 3 : Math.min(f + 1, 7), h = pos[f][0], hn = pos[nx][0];
    g.set([h[0], h[1], h[2], Math.cos(head[f]), Math.sin(head[f])], f * 5);
    lr.set([M.wrap(head[nx] - head[f]) * FPS, (hn[0] - h[0]) * FPS, (hn[2] - h[2]) * FPS, h[1]], f * 4);
    let o = f * 267;
    for (let m = 1; m < 30; m++) { const p = pos[f][m]; lp[o++] = p[0] - h[0]; lp[o++] = p[1]; lp[o++] = p[2] - h[2]; }
    for (let m = 0; m < 30; m++) { const r = M.rot6d(rot[f][m]); for (let k = 0; k < 6; k++) lp[o++] = r[k]; }
  }
  const given = new Int32Array(8), lrm = new Int32Array(8), al = new Int32Array(11);
  for (let i = 0; i < 8; i++) { given[i] = i < 4 + nTargets ? 1 : 0; lrm[i] = given[i]; }
  lrm[3] = 0; lrm[4 + nTargets - 1] = 0;
  for (let i = 0; i < 11; i++) al[i] = allowed.includes(6 + i) ? 1 : 0;
  return { feeds: { global_root_values: g, local_root_values: lr, local_poses: lp, has_global_root_values: given,
                    has_local_root_values: lrm, has_local_poses: given.slice(), allowed_pred_num_tokens: al },
           align: [ap, aq] };
}

export function decode(b, pred, tokens, align) {
  const sk = b.sk, J = sk.names.length, raw = Math.min(tokens * 4, 64), count = Math.max(0, raw - 6);
  const frames = [], contacts = [];
  for (let f = 0; f < count; f++) {
    const o = (3 + f) * OUT_DIM, gq = [];
    for (let m = 0; m < 30; m++) gq.push(M.qFromRot6d(pred.subarray(o + ROT_OFF + 6 * m, o + ROT_OFF + 6 * m + 6)));
    const out = new Float64Array(sk.reference);
    const [rp, rq] = M.tfMul(align[0], align[1], [pred[o], pred[o + 1], pred[o + 2]], gq[0]);
    M.setP(out, 0, rp); M.setQ(out, 0, rq);
    for (let j = 1; j < J; j++) {
      if (b.p2m[j] < 0) continue;
      M.setQ(out, j, M.qnorm(M.qmul(M.qconj(gq[b.ppm[j]]), gq[b.p2m[j]])));
    }
    frames.push(out);
    contacts.push([0, 1, 2, 3].map((k) => Math.min(1, Math.max(0, pred[o + 362 + k]))));
  }
  return { frames, contacts, count };
}

function rootDelta(gen, guide) {
  const q = M.qnorm(M.qmul(M.qconj(M.qnorm(M.Q(gen, 0))), M.qnorm(M.Q(guide, 0))));
  return [M.vsub(M.P(guide, 0), M.P(gen, 0)), q];
}

export function endpointCorrection(frames, ctxLast, tgtFirst) {
  const c = frames.length;
  if (c < 2) return frames;
  const J = frames[0].length / 7, first = frames[0], last = frames[c - 1];
  const [sdp, sdq] = rootDelta(first, ctxLast), [edp, edq] = rootDelta(last, tgtFirst);
  const sdel = [], edel = [];
  for (let j = 1; j < J; j++) {
    const [ip, iq] = M.tfInv(M.P(first, j), M.Q(first, j));
    sdel.push(M.tfMul(ip, iq, M.P(ctxLast, j), M.Q(ctxLast, j)));
    const [lp, lq] = M.tfInv(M.P(last, j), M.Q(last, j));
    edel.push(M.tfMul(lp, lq, M.P(tgtFirst, j), M.Q(tgtFirst, j)));
  }
  return frames.map((fr, f) => {
    const we = f / (c - 1), ws = 1 - we, out = new Float64Array(fr.length);
    const rq = M.qnorm(M.qmul(M.qmul(M.Q(fr, 0), M.qFraction(sdq, ws)), M.qFraction(edq, we)));
    const p = M.P(fr, 0);
    M.setP(out, 0, [p[0] + sdp[0] * ws + edp[0] * we, p[1] + sdp[1] * ws + edp[1] * we, p[2] + sdp[2] * ws + edp[2] * we]);
    M.setQ(out, 0, rq);
    for (let j = 1; j < J; j++) {
      const [sp, sq] = sdel[j - 1], [ep, eq] = edel[j - 1];
      let [bp, bq] = M.tfMul(M.P(fr, j), M.Q(fr, j), M.vscale(sp, ws), M.qFraction(sq, ws));
      [bp, bq] = M.tfMul(bp, bq, M.vscale(ep, we), M.qFraction(eq, we));
      M.setP(out, j, bp); M.setQ(out, j, bq);
    }
    return out;
  });
}

const rotvel = (a, b, j) => M.qToScaledAxis(M.qmul(M.Q(b, j), M.qconj(M.Q(a, j))));

export function velocityMatching(frames, ctxPrev, ctxLast, tgtFirst, tgtSecond, tau = 4) {
  const c = frames.length;
  if (c < 4) return frames;
  const J = frames[0].length / 7, s0 = frames[0], s1 = frames[1], e1 = frames[c - 2], e0 = frames[c - 1];
  const dvp = M.vsub(M.vsub(M.P(ctxLast, 0), M.P(ctxPrev, 0)), M.vsub(M.P(s1, 0), M.P(s0, 0)));
  const dvr = [], der = [];
  for (let j = 0; j < J; j++) dvr.push(M.vsub(rotvel(ctxPrev, ctxLast, j), rotvel(s0, s1, j)));
  let dep = [0, 0, 0];
  if (tgtSecond) {
    dep = M.vsub(M.vsub(M.P(tgtSecond, 0), M.P(tgtFirst, 0)), M.vsub(M.P(e0, 0), M.P(e1, 0)));
    for (let j = 0; j < J; j++) der.push(M.vsub(rotvel(tgtFirst, tgtSecond, j), rotvel(e1, e0, j)));
  }
  return frames.map((fr, t) => {
    const w = t * Math.exp(-t / tau), s = c - 1 - t, wsv = tgtSecond ? s * Math.exp(-s / tau) : 0;
    const out = new Float64Array(fr);
    const p = M.P(fr, 0);
    M.setP(out, 0, [p[0] + w * dvp[0] - wsv * dep[0], p[1] + w * dvp[1] - wsv * dep[1], p[2] + w * dvp[2] - wsv * dep[2]]);
    for (let j = 0; j < J; j++) {
      const a = dvr[j], e = tgtSecond ? der[j] : [0, 0, 0];
      const off = [w * a[0] - wsv * e[0], w * a[1] - wsv * e[1], w * a[2] - wsv * e[2]];
      M.setQ(out, j, M.qnorm(M.qmul(M.qFromScaledAxis(off), M.Q(fr, j))));
    }
    return out;
  });
}

// ---- anatomical limits (anatomy.BoundLimits)
export function bindLimits(limits, sk) {
  const items = [];
  for (const l of limits.limits) {
    const j = sk.names.indexOf(l.joint);
    if (j < 0) continue;
    const children = [];
    for (let c = 0; c < sk.parents.length; c++) if (sk.parents[c] === j) children.push(c);
    items.push({ j, children, rank: l.rank, axes: l.axes, lo: l.lo, hi: l.hi, rest: l.rest });
  }
  const relaxed = Object.entries(limits.relaxed).map(([n, q]) => [sk.names.indexOf(n), q]).filter(([j]) => j >= 0);
  return { items, relaxed };
}

export function applyLimits(bl, pose, relax = false) {
  const out = new Float64Array(pose);
  for (const it of bl.items) {
    const qOld = M.qabs(M.Q(out, it.j));
    let qNew;
    if (it.rank === 0) qNew = it.rest.slice();
    else {
      const r = M.qToScaledAxis(qOld), v = [0, 0, 0];
      for (let k = 0; k < it.rank; k++) {
        const ax = it.axes[k], c = Math.min(it.hi[k], Math.max(it.lo[k], M.vdot(r, ax)));
        v[0] += c * ax[0]; v[1] += c * ax[1]; v[2] += c * ax[2];
      }
      qNew = M.qFromScaledAxis(v);
    }
    const delta = M.qmul(M.qconj(qNew), qOld);
    M.setQ(out, it.j, qNew);
    for (const c of it.children) M.setQ(out, c, M.qnorm(M.qmul(delta, M.Q(out, c))));
  }
  if (relax) for (const [j, q] of bl.relaxed) M.setQ(out, j, q);
  return out;
}

// joints the network does not predict (fingers, eyes, ...) take the relaxed pose; nothing it predicts is changed
export function relaxPose(bl, pose) {
  const out = new Float64Array(pose);
  for (const [j, q] of bl.relaxed) M.setQ(out, j, q);
  return out;
}

// Full generation post-process (engine.generate_batch): decode -> relax -> endpoint -> velocity -> limits
// opts.constrain === false (AI-only mode): relax the unpredicted joints only, no joint clamping
export function postProcess(b, bl, pred, tokens, align, guides, nTargets, opts) {
  let { frames, contacts } = decode(b, pred, tokens, align);
  if (bl && opts.constrain === false) { frames = frames.map((f) => relaxPose(bl, f)); bl = null; }
  if (bl) frames = frames.map((f) => applyLimits(bl, f, true));
  if (opts.correctEndpoints !== false) frames = endpointCorrection(frames, guides[3], guides[4]);
  if (opts.velocityMatching) frames = velocityMatching(frames, guides[2], guides[3], guides[4], nTargets >= 2 ? guides[5] : null, opts.tau || 4);
  if (bl) frames = frames.map((f) => applyLimits(bl, f, false));
  return { frames, contacts };
}
