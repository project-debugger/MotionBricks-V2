// Fighting moves: hand-built key poses (guard, jab, cross, hook, uppercut, front kick, roundhouse, block)
// made by IK on the mocap idle pose. The SOMA network generates the motion into each key pose and
// back to the stance (same pipeline as every other action; the bank holds them for instant response).
import * as M from './math3d.js';
import { applyLimits } from './codec.js';
import { twoBoneIK } from './controller.js';

export const FIGHT_MOVES = ['jab', 'cross', 'hook', 'uppercut', 'front_kick', 'roundhouse', 'block'];

export function buildFightPoses(sk, base, limits) {
  const ix = (n) => sk.names.indexOf(n), par = sk.parents;
  const fk = (p) => M.fk(p, par);
  const W = (w, n) => M.P(w, ix(n));
  const w0 = fk(base);
  const left = M.vnorm([W(w0, 'LeftArm')[0] - W(w0, 'RightArm')[0], 0, W(w0, 'LeftArm')[2] - W(w0, 'RightArm')[2]]);
  const fwd = M.vnorm(M.vcross(left, [0, 1, 0])), fwdS = M.vdot(fwd, [0, 0, 1]) < 0 ? M.vscale(fwd, -1) : fwd; // facing +Z
  const F = fwdS, Lf = left, U = [0, 1, 0];
  const add = (...vs) => vs.reduce((a, b) => M.vadd(a, b));
  const twist = (p, angle, joints = [['Spine1', 0.25], ['Spine2', 0.35], ['Chest', 0.4]]) => {
    let out = new Float64Array(p);
    for (const [n, f] of joints) {
      const w = fk(out), j = ix(n), ax = M.qrotInv(M.Q(w, j), U);
      M.setQ(out, j, M.qnorm(M.qmul(M.Q(out, j), M.qAxisAngle(ax, angle * f))));
    }
    return out;
  };
  const lean = (p, angle) => { // forward (+) / back (-) lean of the upper body
    let out = new Float64Array(p);
    for (const [n, f] of [['Spine1', 0.3], ['Spine2', 0.3], ['Chest', 0.4]]) {
      const w = fk(out), j = ix(n), ax = M.qrotInv(M.Q(w, j), Lf);
      M.setQ(out, j, M.qnorm(M.qmul(M.Q(out, j), M.qAxisAngle(ax, -angle * f))));
    }
    return out;
  };
  const reach = (p, side, target) => twoBoneIK(p, fk(p), sk, ix(`${side}Arm`), ix(`${side}ForeArm`), ix(`${side}Hand`), target, 1);
  const leg = (p, side, target) => twoBoneIK(p, fk(p), sk, ix(`${side}Leg`), ix(`${side}Shin`), ix(`${side}Foot`), target, 1);
  const lowerHips = (p, dy, shift = [0, 0, 0], feet = {}) => {
    const w = fk(p), lf = W(w, 'LeftFoot'), rf = W(w, 'RightFoot');
    let out = new Float64Array(p); const r = M.P(out, 0);
    M.setP(out, 0, [r[0] + shift[0], r[1] - dy, r[2] + shift[2]]);
    out = leg(out, 'Left', feet.L || lf); out = leg(out, 'Right', feet.R || rf);
    return out;
  };
  // orthodox stance: left foot forward, right foot back, knees soft, hands up
  const stanceFeet = () => ({ L: add(W(w0, 'LeftFoot'), M.vscale(F, 0.13), M.vscale(Lf, 0.03)), R: add(W(w0, 'RightFoot'), M.vscale(F, -0.13), M.vscale(Lf, -0.04)) });
  const guardHands = (p) => {
    const w = fk(p), head = W(w, 'Head');
    p = reach(p, 'Left', add(head, M.vscale(F, 0.2), M.vscale(Lf, 0.07), M.vscale(U, -0.12)));
    return reach(p, 'Right', add(head, M.vscale(F, 0.14), M.vscale(Lf, -0.08), M.vscale(U, -0.1)));
  };
  let guard = lowerHips(base, 0.06, [0, 0, 0], stanceFeet());
  guard = twist(guard, 0.2);
  guard = guardHands(guard);
  const shoulder = (p, side) => W(fk(p), `${side}Arm`);
  const poses = {};
  // jab: lead (left) hand straight out at shoulder height
  {
    let p = twist(guard, -0.12);
    const s = shoulder(p, 'Left');
    p = reach(p, 'Left', add(s, M.vscale(F, 0.58), M.vscale(Lf, -0.1), M.vscale(U, 0.03)));
    poses.jab = p;
  }
  // cross: rear (right) hand straight, hips and shoulders turn through
  {
    let p = twist(guard, -0.75);
    const s = shoulder(p, 'Right');
    p = reach(p, 'Right', add(s, M.vscale(F, 0.6), M.vscale(Lf, 0.12), M.vscale(U, 0.03)));
    poses.cross = lean(p, 0.06);
  }
  // left hook: elbow up, forearm horizontal, torso rotated
  {
    let p = twist(guard, -0.6);
    const s = shoulder(p, 'Left');
    p = reach(p, 'Left', add(s, M.vscale(F, 0.32), M.vscale(Lf, -0.3), M.vscale(U, 0.0)));
    poses.hook = p;
  }
  // right uppercut: dip, then the rear hand drives up in front of the chin
  {
    let p = lowerHips(guard, 0.04, [0, 0, 0], {});
    p = twist(p, -0.45);
    const head = W(fk(p), 'Head');
    p = reach(p, 'Right', add(head, M.vscale(F, 0.3), M.vscale(Lf, 0.02), M.vscale(U, -0.05)));
    poses.uppercut = lean(p, 0.05);
  }
  // front kick (right leg): foot driven forward at hip height, body leans back over the left leg
  {
    let p = lean(guard, -0.18);
    const hip = W(fk(p), 'RightLeg');
    p = leg(p, 'Right', add([hip[0], 0.72, hip[2]], M.vscale(F, 0.62)));
    poses.front_kick = p;
  }
  // roundhouse (right leg): hips turned, leg swung across at chest height
  {
    let p = twist(guard, -0.9, [['Spine1', 0.4], ['Spine2', 0.3], ['Chest', 0.3]]);
    p = lean(p, -0.1);
    const hip = W(fk(p), 'RightLeg');
    p = leg(p, 'Right', add([hip[0], 0.98, hip[2]], M.vscale(F, 0.5), M.vscale(Lf, 0.22)));
    poses.roundhouse = p;
  }
  // block: both forearms up in front of the face
  {
    const head = W(fk(guard), 'Head');
    let p = reach(guard, 'Left', add(head, M.vscale(F, 0.17), M.vscale(Lf, 0.06), M.vscale(U, 0.02)));
    p = reach(p, 'Right', add(head, M.vscale(F, 0.17), M.vscale(Lf, -0.06), M.vscale(U, 0.02)));
    poses.block = p;
  }
  const out = {};
  for (const [k, p] of Object.entries({ guard, ...poses })) out[k] = applyLimits(limits, p, false);
  return out;
}

// add the fight key poses to the action table (4-frame windows of the same pose), usable live and by the bank
export function registerFightActions(sk, lib, actions, limits) {
  const base = lib.windows[actions.idle.window][3];
  const poses = buildFightPoses(sk, base, limits);
  for (const [name, p] of Object.entries(poses)) {
    const g = M.P(p, 0), win = [0, 1, 2, 3].map(() => { const f = new Float64Array(p); M.setP(f, 0, [0, g[1], 0]); return f; });
    actions[name] = { window: lib.windows.length, anchor: '', kind: 'oneshot', fight: true, root0: [0, 0, 0], rootk: [0, 0, 0] };
    lib.windows.push(win);
  }
  return poses;
}
