// Rigid-transform math, a line-by-line port of runtime/somart/math3d.py.
// Quaternions are [x, y, z, w]; a pose is a Float64Array of J*7: (px, py, pz, qx, qy, qz, qw) per joint.

export const EPS = 1e-8;

export function qmul(a, b) {
  const [ax, ay, az, aw] = a, [bx, by, bz, bw] = b;
  return [aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
          aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz];
}
export const qconj = (q) => [-q[0], -q[1], -q[2], q[3]];
export function qnorm(q) {
  const n = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
  return [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
}
export const qabs = (q) => (q[3] < 0 ? [-q[0], -q[1], -q[2], -q[3]] : q);
export function qrot(q, v) {
  const [qx, qy, qz, qw] = q;
  const tx = 2 * (qy * v[2] - qz * v[1]), ty = 2 * (qz * v[0] - qx * v[2]), tz = 2 * (qx * v[1] - qy * v[0]);
  return [v[0] + qw * tx + (qy * tz - qz * ty), v[1] + qw * ty + (qz * tx - qx * tz), v[2] + qw * tz + (qx * ty - qy * tx)];
}
export const qrotInv = (q, v) => qrot(qconj(q), v);
export function qAxisAngle(axis, angle) {
  const n = Math.hypot(axis[0], axis[1], axis[2]) || 1, h = 0.5 * angle, s = Math.sin(h) / n;
  return [axis[0] * s, axis[1] * s, axis[2] * s, Math.cos(h)];
}
export const qYaw = (a) => [0, Math.sin(0.5 * a), 0, Math.cos(0.5 * a)];
export function qToScaledAxis(q) {
  q = qabs(q);
  const s = Math.hypot(q[0], q[1], q[2]);
  if (s < 1e-9) return [2 * q[0], 2 * q[1], 2 * q[2]];
  const a = 2 * Math.atan2(s, q[3]) / s;
  return [q[0] * a, q[1] * a, q[2] * a];
}
export function qFromScaledAxis(r) {
  const angle = Math.hypot(r[0], r[1], r[2]), half = 0.5 * angle;
  const k = angle > 1e-9 ? Math.sin(half) / angle : 0.5;
  return qnorm([r[0] * k, r[1] * k, r[2] * k, Math.cos(half)]);
}
// slerp(identity, q, w) via the log map (math3d.quat_fraction)
export function qFraction(q, w) {
  q = qabs(q);
  const s = Math.hypot(q[0], q[1], q[2]);
  const half = 0.5 * 2 * Math.atan2(s, q[3]) * w, k = Math.sin(half) / Math.max(s, 1e-12);
  return [q[0] * k, q[1] * k, q[2] * k, Math.cos(half)];
}
export function qSlerp(a, b, t) {
  let d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  if (d < 0) { b = [-b[0], -b[1], -b[2], -b[3]]; d = -d; }
  if (d > 0.9995) return qnorm([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t, a[3] + (b[3] - a[3]) * t]);
  const th = Math.acos(Math.min(1, d)), s = Math.sin(th), wa = Math.sin((1 - t) * th) / s, wb = Math.sin(t * th) / s;
  return [wa * a[0] + wb * b[0], wa * a[1] + wb * b[1], wa * a[2] + wb * b[2], wa * a[3] + wb * b[3]];
}
export function qFromMatrixCols(x, y, z) { // columns x, y, z (Shepperd)
  const m00 = x[0], m10 = x[1], m20 = x[2], m01 = y[0], m11 = y[1], m21 = y[2], m02 = z[0], m12 = z[1], m22 = z[2];
  const tr = m00 + m11 + m22;
  let q;
  if (tr > 0) { const s = Math.sqrt(Math.max(tr + 1, EPS)) * 2; q = [(m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, 0.25 * s]; }
  else if (m00 > m11 && m00 > m22) { const s = Math.sqrt(Math.max(1 + m00 - m11 - m22, EPS)) * 2; q = [0.25 * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s]; }
  else if (m11 > m22) { const s = Math.sqrt(Math.max(1 + m11 - m00 - m22, EPS)) * 2; q = [(m01 + m10) / s, 0.25 * s, (m12 + m21) / s, (m02 - m20) / s]; }
  else { const s = Math.sqrt(Math.max(1 + m22 - m00 - m11, EPS)) * 2; q = [(m02 + m20) / s, (m12 + m21) / s, 0.25 * s, (m10 - m01) / s]; }
  return qnorm(q);
}
export const vsub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const vadd = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const vscale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
export const vdot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const vcross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
export const vlen = (a) => Math.hypot(a[0], a[1], a[2]);
export function vnorm(a) { const n = vlen(a) || 1; return [a[0] / n, a[1] / n, a[2] / n]; }
export const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));

// ---- pose helpers (Float64Array J*7)
export const P = (pose, j) => [pose[j * 7], pose[j * 7 + 1], pose[j * 7 + 2]];
export const Q = (pose, j) => [pose[j * 7 + 3], pose[j * 7 + 4], pose[j * 7 + 5], pose[j * 7 + 6]];
export function setP(pose, j, p) { pose[j * 7] = p[0]; pose[j * 7 + 1] = p[1]; pose[j * 7 + 2] = p[2]; }
export function setQ(pose, j, q) { pose[j * 7 + 3] = q[0]; pose[j * 7 + 4] = q[1]; pose[j * 7 + 5] = q[2]; pose[j * 7 + 6] = q[3]; }
export function tfMul(pa, qa, pb, qb) { const r = qrot(qa, pb); return [[pa[0] + r[0], pa[1] + r[1], pa[2] + r[2]], qmul(qa, qb)]; }
export function tfInv(p, q) { const qi = qconj(q); const r = qrot(qi, p); return [[-r[0], -r[1], -r[2]], qi]; }

export function fk(local, parents) {
  const J = parents.length, world = new Float64Array(J * 7);
  for (let j = 0; j < J; j++) {
    const p = parents[j];
    if (p < 0) { for (let k = 0; k < 7; k++) world[k + j * 7] = local[k + j * 7]; continue; }
    const [pp, pq] = tfMul(P(world, p), Q(world, p), P(local, j), Q(local, j));
    setP(world, j, pp); setQ(world, j, pq);
  }
  return world;
}
export function rot6d(q) { const x = qrot(q, [1, 0, 0]), y = qrot(q, [0, 1, 0]); return [x[0], x[1], x[2], y[0], y[1], y[2]]; }
export function qFromRot6d(r) {
  const x = vnorm([r[0], r[1], r[2]]), z = vnorm(vcross(x, [r[3], r[4], r[5]])), y = vcross(z, x);
  return qFromMatrixCols(x, y, z);
}
export function blendPose(a, b, t) { // lerp positions, slerp rotations
  const J = a.length / 7, out = new Float64Array(a.length);
  for (let j = 0; j < J; j++) {
    for (let k = 0; k < 3; k++) out[j * 7 + k] = a[j * 7 + k] + (b[j * 7 + k] - a[j * 7 + k]) * t;
    setQ(out, j, qSlerp(Q(a, j), Q(b, j), t));
  }
  return out;
}
// deterministic PRNG (mulberry32)
export function rng(seed) {
  let s = seed >>> 0;
  return () => { s += 0x6D2B79F5; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
