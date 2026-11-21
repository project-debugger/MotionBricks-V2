// Lean into turns: on circular paths turning either way, the body must bank toward the centre.  node site/build/test/lean_test.mjs
import { Lean } from '../../js/controller.js';
import * as M from '../../js/math3d.js';
for (const dirSign of [1, -1]) {
  const L = new Lean(), R = 5, v = 4, dt = 1 / 60; let out;
  for (let k = 0; k < 120; k++) {
    const th = v * k * dt / R, pos = [dirSign * (R - R * Math.cos(th)), 1, R * Math.sin(th)]; // from the origin heading +z, curving toward x = dirSign * R
    const local = new Float64Array(7); M.setP(local, 0, pos); M.setQ(local, 0, [0, 0, 0, 1]);
    out = L.apply(local, 1, dt); var center = [dirSign * R, 0, 0], p = pos;
  }
  const up = M.qrot(M.Q(out, 0), [0, 1, 0]), toC = M.vnorm([center[0] - p[0], 0, center[2] - p[2]]);
  console.log(`turn ${dirSign > 0 ? 'A' : 'B'}: lean ${(L.a * 57.3).toFixed(1)} deg, up . toCenter = ${M.vdot(up, toC).toFixed(3)} (positive = leaning into the turn)`);
}
