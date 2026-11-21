// Procedural animals. The SOMA network is human-only, so animals are animated by a procedural gait system:
// walk / trot / gallop (or hop) foot patterns with planted feet, two-bone leg IK with species-specific
// joint directions, a bending spine, a stabilised head that looks where it is going, a spring-driven tail,
// ears, breathing, and actions (jump, attack, rear-up/buck, sit, lie down, stalk, sidestep).
// Bodies are built from smooth primitives at runtime (no external assets).

const TAU = Math.PI * 2;
const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
const lerp = (a, b, t) => a + (b - a) * t;
const smoothstep = (t) => { t = clamp(t, 0, 1); return t * t * (3 - 2 * t); };
const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));
const frac = (x) => x - Math.floor(x);

// leg = [upper, lower, foot] lengths; r = torso radii [x, y, z-half] at hips and chest
export const SPECIES = {
  dog: { name: 'Dog', hip: 0.55, sh: 0.57, len: 0.55, w: 0.11, rH: [0.13, 0.15, 0.17], rC: [0.14, 0.18, 0.17], front: [0.22, 0.2, 0.1], hind: [0.25, 0.22, 0.14],
    neck: [0.2, 50], head: [0.105, 0.13, 0.055], ears: 'floppy', earSize: 0.09, tail: [0.36, 35, 0.025, 'thin'], col: ['#b98a4a', '#e9cf9c', '#2b2420'], speeds: [1.3, 2.8, 7], freq: 1.7, attack: 'bite', kick: 'pounce', wag: 1 },
  wolf: { name: 'Wolf', hip: 0.72, sh: 0.76, len: 0.72, w: 0.12, rH: [0.14, 0.17, 0.2], rC: [0.16, 0.22, 0.2], front: [0.3, 0.27, 0.13], hind: [0.33, 0.3, 0.18],
    neck: [0.26, 38], head: [0.12, 0.17, 0.06], ears: 'pointy', earSize: 0.09, tail: [0.45, -25, 0.05, 'bushy'], col: ['#7f7f80', '#d8d3c9', '#222'], speeds: [1.4, 3.2, 9], freq: 1.5, attack: 'bite', kick: 'pounce', wag: 0.4 },
  fox: { name: 'Fox', hip: 0.34, sh: 0.35, len: 0.4, w: 0.08, rH: [0.08, 0.1, 0.12], rC: [0.09, 0.11, 0.12], front: [0.14, 0.13, 0.07], hind: [0.16, 0.14, 0.09],
    neck: [0.14, 40], head: [0.075, 0.11, 0.04], ears: 'pointy', earSize: 0.085, tail: [0.42, -10, 0.055, 'bushy', '#f2efe8'], col: ['#c8622a', '#f4ede2', '#1d1a18'], speeds: [1.0, 2.5, 8], freq: 2.0, attack: 'bite', kick: 'pounce', wag: 0.5, socks: '#2a211c' },
  cat: { name: 'Cat', hip: 0.26, sh: 0.25, len: 0.32, w: 0.06, rH: [0.07, 0.085, 0.1], rC: [0.07, 0.09, 0.1], front: [0.11, 0.1, 0.05], hind: [0.12, 0.11, 0.08],
    neck: [0.09, 45], head: [0.06, 0.04, 0.035], ears: 'pointy', earSize: 0.045, tail: [0.3, 55, 0.018, 'thin'], col: ['#8a8178', '#d9d2c7', '#ff9fb0'], speeds: [0.7, 1.8, 6.5], freq: 2.2, attack: 'swipe', kick: 'pounce', wag: 0.25, stripes: '#5b544c' },
  horse: { name: 'Horse', hip: 1.45, sh: 1.48, len: 1.05, w: 0.2, rH: [0.28, 0.33, 0.3], rC: [0.27, 0.36, 0.3], front: [0.42, 0.42, 0.42], hind: [0.48, 0.46, 0.46],
    neck: [0.72, 55], head: [0.13, 0.42, 0.1], ears: 'pointy', earSize: 0.11, tail: [0.75, -55, 0.06, 'hair'], col: ['#6e4428', '#7a4d2e', '#1c1410'], speeds: [1.7, 4.0, 12], freq: 0.95, hp: 1.0, attack: 'rear', kick: 'buck', wag: 0.3, mane: '#241812', hooves: '#1d1915' },
  deer: { name: 'Deer', hip: 0.95, sh: 0.92, len: 0.72, w: 0.13, rH: [0.15, 0.2, 0.2], rC: [0.15, 0.22, 0.2], front: [0.3, 0.3, 0.27], hind: [0.33, 0.32, 0.32],
    neck: [0.38, 60], head: [0.09, 0.2, 0.06], ears: 'oval', earSize: 0.12, tail: [0.12, 40, 0.035, 'tuft', '#f5f1ea'], col: ['#9c6a3d', '#ecdcc4', '#1b1512'], speeds: [1.4, 3.5, 11], freq: 1.25, hp: 0.7, attack: 'rear', kick: 'buck', wag: 0.6, antlers: '#d9c7a4', hooves: '#1d1915' },
  bear: { name: 'Bear', hip: 0.88, sh: 0.95, len: 0.85, w: 0.2, rH: [0.3, 0.33, 0.32], rC: [0.33, 0.38, 0.32], front: [0.34, 0.31, 0.12], hind: [0.36, 0.33, 0.13],
    neck: [0.22, 25], head: [0.17, 0.15, 0.09], ears: 'round', earSize: 0.06, tail: [0.1, 10, 0.04, 'stub'], col: ['#4a3323', '#5b4130', '#141010'], speeds: [1.1, 2.6, 9], freq: 1.1, attack: 'swipe', kick: 'rear', wag: 0, heavy: 1 },
  elephant: { name: 'Elephant', hip: 1.55, sh: 1.7, len: 1.35, w: 0.36, rH: [0.52, 0.62, 0.62], rC: [0.56, 0.7, 0.6], front: [0.72, 0.68, 0.12], hind: [0.75, 0.68, 0.12],
    neck: [0.2, 15], head: [0.42, 0.25, 0.3], ears: 'elephant', earSize: 0.55, tail: [0.6, -70, 0.03, 'tuft', '#3c3a38'], col: ['#7d7f82', '#8a8c8f', '#2a2a2b'], speeds: [1.2, 2.5, 5], freq: 0.75, attack: 'charge', kick: 'stomp', wag: 0.3, trunk: 1, tusks: '#efe6d2', column: 1 },
  rabbit: { name: 'Rabbit', hip: 0.15, sh: 0.12, len: 0.18, w: 0.05, rH: [0.07, 0.08, 0.08], rC: [0.055, 0.065, 0.07], front: [0.06, 0.06, 0.03], hind: [0.09, 0.08, 0.1],
    neck: [0.04, 50], head: [0.05, 0.035, 0.03], ears: 'long', earSize: 0.11, tail: [0.04, 40, 0.025, 'puff', '#f8f5f0'], col: ['#9b8166', '#eadfcf', '#3a2a24'], speeds: [0.5, 1.6, 4.5], freq: 2.6, attack: 'thump', kick: 'pounce', wag: 0, hop: 1 },
  lion: { name: 'Lion', hip: 0.92, sh: 0.98, len: 0.95, w: 0.17, rH: [0.2, 0.24, 0.26], rC: [0.22, 0.29, 0.26], front: [0.38, 0.34, 0.13], hind: [0.42, 0.37, 0.18],
    neck: [0.26, 30], head: [0.17, 0.15, 0.08], ears: 'round', earSize: 0.06, tail: [0.75, -40, 0.03, 'tuft', '#3b2617'], col: ['#c39557', '#e3c79a', '#1c1410'], speeds: [1.3, 3.2, 13], freq: 1.15, attack: 'swipe', kick: 'pounce', wag: 0.3, maneHead: '#6b4423' },
};
export const SPECIES_KEYS = Object.keys(SPECIES);

// gait foot phase offsets [LF, RF, LH, RH] and duty factors
const GAITS = {
  walk: { off: [0.25, 0.75, 0.0, 0.5], duty: 0.64, lift: 0.18 },
  trot: { off: [0.0, 0.5, 0.5, 0.0], duty: 0.42, lift: 0.26 },
  gallop: { off: [0.58, 0.46, 0.0, 0.1], duty: 0.3, lift: 0.34 },
  hop: { off: [0.48, 0.52, 0.0, 0.02], duty: 0.28, lift: 0.5 },
};

export class Animal {
  constructor(THREE, scene, key) {
    this.THREE = THREE; this.key = key; this.sp = SPECIES[key];
    const sp = this.sp, V = (x, y, z) => new THREE.Vector3(x, y, z);
    this.V = V;
    this.root = new THREE.Group(); scene.add(this.root);
    this.pos = V(0, 0, 0); this.vel = V(0, 0, 0); this.yaw = 0; this.yawRate = 0;
    this.phase = 0; this.gaitMix = { walk: 1, trot: 0, gallop: 0, hop: 0 }; this.speed = 0;
    this.time = 0; this.breath = 0; this.exert = 0; this.action = null; this.mode = 'stand'; this.stalk = false;
    this.lookYaw = 0; this.lookPitch = 0; this.lookT = 2; this.lookGoal = [0, 0]; this.earT = 1; this.earFlick = [0, 0];
    const size = Math.max(sp.hip, sp.sh); this.size = size;
    // materials
    const mat = (c, r = 0.85, extra = {}) => new THREE.MeshStandardMaterial({ color: c, roughness: r, ...extra });
    this.mBody = mat(sp.col[0]); this.mBelly = mat(sp.col[1]); this.mDark = mat(sp.col[2], 0.5);
    this.mEye = mat('#0d0a08', 0.2, { metalness: 0.1 }); this.mFoot = sp.hooves ? mat(sp.hooves, 0.6) : sp.socks ? mat(sp.socks) : this.mBody;
    this.parts = [];
    const mesh = (geo, m, cast = true) => { const o = new THREE.Mesh(geo, m); o.castShadow = cast; o.receiveShadow = true; this.root.add(o); this.parts.push(o); return o; };
    this.sphere = new THREE.SphereGeometry(1, 20, 14);
    this.capsule = new THREE.CapsuleGeometry(1, 1, 6, 12); // unit radius, unit length along +Y
    this.cone = new THREE.ConeGeometry(1, 1, 12);
    // torso blobs: hips, mid, chest, belly
    this.tHip = mesh(this.sphere, this.mBody); this.tMid = mesh(this.sphere, this.mBody); this.tChest = mesh(this.sphere, this.mBody);
    this.tBelly = mesh(this.sphere, this.mBelly); this.tChestFront = mesh(this.sphere, this.mBelly);
    if (sp.stripes) { this.stripes = [0, 1, 2].map(() => mesh(this.sphere, mat(sp.stripes))); }
    // neck & head
    this.neckParts = [0, 1].map(() => mesh(this.capsule, this.mBody));
    this.skull = mesh(this.sphere, this.mBody); this.snout = mesh(this.sphere, sp.trunk ? this.mBody : this.mBelly); this.jaw = mesh(this.sphere, this.mBelly);
    this.nose = mesh(this.sphere, this.mDark); this.eyes = [0, 1].map(() => mesh(this.sphere, this.mEye, false));
    this.ears = [0, 1].map(() => mesh(sp.ears === 'floppy' ? this.capsule : sp.ears === 'pointy' || sp.ears === 'long' || sp.ears === 'oval' ? this.cone : this.sphere, this.mBody));
    if (sp.mane) this.maneParts = [0, 1, 2, 3].map(() => mesh(this.sphere, mat(sp.mane, 0.95)));
    if (sp.maneHead) this.maneHead = mesh(this.sphere, mat(sp.maneHead, 0.95));
    if (sp.antlers) this.antlers = Array.from({ length: 8 }, () => mesh(this.capsule, mat(sp.antlers, 0.7)));
    if (sp.trunk) this.trunk = Array.from({ length: 6 }, () => mesh(this.capsule, this.mBody));
    if (sp.tusks) this.tusks = [0, 1].map(() => mesh(this.capsule, mat(sp.tusks, 0.4)));
    // legs: 3 segments each + foot pad
    this.legs = ['LF', 'RF', 'LH', 'RH'].map((n, i) => {
      const front = i < 2, side = i % 2 === 0 ? 1 : -1, L = front ? sp.front : sp.hind;
      const thick = (sp.column ? 0.13 : 0.13) * (L[0] + L[1]) * (front ? 1 : 1.1) * (sp.heavy ? 1.3 : 1);
      return { name: n, front, side, L, thick, segs: [0, 1, 2].map((k) => mesh(this.capsule, k === 2 ? this.mFoot : this.mBody)), pad: mesh(this.sphere, sp.hooves ? this.mFoot : sp.socks ? this.mFoot : sp.column ? this.mBody : this.mDark),
        foot: V(0, 0, 0), plant: V(0, 0, 0), lift: V(0, 0, 0), stance: true, idleStep: 0, idleFrom: V(), idleTo: V() };
    });
    // tail chain
    const [tl, , tr, style] = sp.tail;
    this.tailN = style === 'stub' || style === 'puff' ? 2 : 6;
    this.tailParts = Array.from({ length: this.tailN }, (_, k) => mesh(style === 'puff' ? this.sphere : this.capsule, (sp.tail[4] && k === this.tailN - 1) ? mat(sp.tail[4]) : style === 'hair' ? mat(sp.mane || sp.col[2], 0.95) : this.mBody));
    this.tailPts = Array.from({ length: this.tailN + 1 }, () => V()); this.tailPrev = null;
    this.reset();
  }
  reset() {
    this.placeFeet(true);
  }
  setVisible(v) { this.root.visible = v; }
  teleport(x, z, yaw) { this.pos.set(x, 0, z); this.yaw = yaw; this.speed = 0; this.placeFeet(); this.tailPrev = null; this.mode = 'stand'; this.action = null; }
  worldXZ() { return [this.pos.x, this.pos.z]; }
  dispose() { this.root.removeFromParent(); }
  // neutral foot position in the root frame (x side, z along body)
  neutral(leg) {
    const sp = this.sp, zH = -sp.len / 2, zC = sp.len / 2;
    return this.V(leg.side * sp.w * (leg.front ? 1.0 : 1.05), 0, leg.front ? zC + 0.02 * this.size : zH - 0.02 * this.size);
  }
  toWorld(v) { const c = Math.cos(this.yaw), s = Math.sin(this.yaw); return this.V(this.pos.x + v.x * c + v.z * s, v.y, this.pos.z - v.x * s + v.z * c); }
  toLocal(v) { const c = Math.cos(this.yaw), s = Math.sin(this.yaw), dx = v.x - this.pos.x, dz = v.z - this.pos.z; return this.V(dx * c - dz * s, v.y, dx * s + dz * c); }
  placeFeet() { for (const l of this.legs) { l.foot.copy(this.toWorld(this.neutral(l))); l.plant.copy(l.foot); l.stance = true; } }
  trigger(name) {
    const sp = this.sp, map = { jab: sp.attack, cross: sp.attack, hook: sp.attack, uppercut: sp.attack, front_kick: sp.kick, roundhouse: sp.kick, block: 'defend', dodge: 'sidestep', jump: 'jump' };
    if (name === 'sit') { this.mode = 'sit'; return; }
    if (name === 'stand_up') { this.mode = 'stand'; return; }
    if (name === 'duck') { this.mode = this.mode === 'lie' ? 'stand' : 'lie'; return; }
    const a = map[name];
    if (!a) return; // e.g. the human-only sit_idle
    if (this.action && this.action.t < this.action.dur * 0.7) return;
    const dur = { jump: 0.95, bite: 0.55, swipe: 0.7, pounce: 0.9, rear: 1.4, buck: 0.9, defend: 0.9, sidestep: 0.55, charge: 1.1, stomp: 1.0, thump: 0.6 }[a] || 0.8;
    this.action = { name: a, t: 0, dur, side: Math.random() < 0.5 ? 1 : -1, y: 0, vy: 0 };
    if (this.mode !== 'stand') this.mode = 'stand';
  }
  update(cmd, dt) {
    if (dt <= 0) return;
    const sp = this.sp, [vw, vt, vg] = sp.speeds;
    this.time += dt;
    if (cmd.action) this.trigger(cmd.action);
    this.stalk = cmd.gait === 'crouch';
    // ---------------- locomotion: spring velocity toward the command, turn toward the motion
    const mag = Math.min(1, Math.hypot(cmd.move[0], cmd.move[1]));
    let want = 0;
    if (mag > 0.05 && this.mode === 'stand') want = (cmd.gait === 'sprint' ? vg : cmd.gait === 'walk' ? vw : this.stalk ? vw * 0.6 : vt) * mag;
    const act = this.action;
    if (act && ['rear', 'defend', 'stomp', 'buck', 'swipe', 'thump'].includes(act.name)) want *= 0.2;
    const dir = mag > 0.05 ? Math.atan2(cmd.move[0], cmd.move[1]) : this.yaw;
    const turnErr = wrap(dir - this.yaw);
    // big animals turn slower; turning in place while stopped
    const turnRate = (sp.hop ? 6 : 4.2) / Math.sqrt(Math.max(0.3, this.size)) * (want > 0.1 ? 1 : 0.6);
    const yr = clamp(turnErr * 6, -turnRate, turnRate) * (mag > 0.05 ? 1 : 0);
    this.yawRate += (yr - this.yawRate) * (1 - Math.exp(-dt * 8));
    this.yaw = wrap(this.yaw + this.yawRate * dt);
    const align = Math.max(0, Math.cos(turnErr)); // slow down while facing away from the goal
    const target = want * (0.35 + 0.65 * align);
    const acc = target > this.speed ? 2.5 + vg * 0.6 : 4 + vg * 0.5;
    this.speed += clamp(target - this.speed, -acc * dt, acc * dt);
    let vx = Math.sin(this.yaw) * this.speed, vz = Math.cos(this.yaw) * this.speed;
    if (act && act.name === 'sidestep') { const k = Math.sin(Math.PI * act.t / act.dur) * 3.2 * Math.sqrt(this.size); vx += Math.cos(this.yaw) * k * act.side; vz -= Math.sin(this.yaw) * k * act.side; }
    if (act && (act.name === 'pounce' || act.name === 'charge' || act.name === 'bite')) {
      const u = act.t / act.dur, k = (act.name === 'charge' ? 4 : act.name === 'pounce' ? 3 : 1.4) * Math.sqrt(this.size) * Math.max(0, Math.sin(Math.PI * clamp((u - 0.2) / 0.5, 0, 1)));
      vx += Math.sin(this.yaw) * k; vz += Math.cos(this.yaw) * k;
    }
    this.vel.set(vx, 0, vz);
    this.pos.x += vx * dt; this.pos.z += vz * dt;
    const vmag = Math.hypot(vx, vz);
    // ---------------- gait: blend walk -> trot -> gallop (or hop) by speed
    const g = sp.hop ? { walk: vmag < vw * 0.9 ? 1 : 0, trot: 0, gallop: 0, hop: vmag < vw * 0.9 ? 0 : 1 }
      : { walk: 1 - smoothstep((vmag - vw * 1.1) / (vt * 0.5)), gallop: smoothstep((vmag - vt * 1.15) / (vt * 0.5)), trot: 0, hop: 0 };
    if (!sp.hop) g.trot = Math.max(0, 1 - g.walk - g.gallop);
    if (this.stalk) { g.walk = 1; g.trot = g.gallop = 0; }
    for (const k of Object.keys(this.gaitMix)) this.gaitMix[k] += (g[k] - this.gaitMix[k]) * (1 - Math.exp(-dt * 5));
    const mix = this.gaitMix, tot = Object.values(mix).reduce((a, b) => a + b, 0) || 1;
    const duty = Object.entries(mix).reduce((a, [k, w]) => a + GAITS[k].duty * w, 0) / tot;
    const lift = Object.entries(mix).reduce((a, [k, w]) => a + GAITS[k].lift * w, 0) / tot;
    const offs = [0, 1, 2, 3].map((i) => { let x = 0, y = 0; for (const [k, w] of Object.entries(mix)) { const a = TAU * GAITS[k].off[i]; x += Math.cos(a) * w; y += Math.sin(a) * w; } return frac(Math.atan2(y, x) / TAU); });
    // stride frequency grows with speed (stride length grows too)
    const moving = vmag > 0.05 || Math.abs(this.yawRate) > 0.4;
    const legLen = (sp.hind[0] + sp.hind[1] + sp.front[0] + sp.front[1]) / 2, maxStride = legLen * (1.1 + 1.3 * this.gaitMix.gallop + 0.8 * this.gaitMix.hop);
    const freq = Math.max(sp.freq * (0.75 + 0.55 * Math.pow(vmag / vw, 0.45)) * (this.stalk ? 0.75 : 1), vmag / maxStride);
    if (moving) this.phase = frac(this.phase + freq * dt);
    const stride = vmag / Math.max(freq, 0.1);
    // ---------------- feet
    const airborne = act && act.name === 'jump' && act.y > 0.02;
    this.legs.forEach((l, i) => {
      const n = this.neutral(l), ph = frac(this.phase + offs[i]);
      if (!moving && !airborne) {
        // standing: feet stay planted; a foot that ends up far from where it belongs takes a quick step
        const home = this.toWorld(n), d = Math.hypot(l.foot.x - home.x, l.foot.z - home.z);
        if (l.idleStep > 0) {
          l.idleStep = Math.max(0, l.idleStep - dt / 0.28); const u = 1 - l.idleStep;
          l.foot.set(lerp(l.idleFrom.x, l.idleTo.x, smoothstep(u)), Math.sin(Math.PI * u) * 0.12 * (l.L[0] + l.L[1]), lerp(l.idleFrom.z, l.idleTo.z, smoothstep(u)));
        } else if (d > 0.18 * (l.L[0] + l.L[1]) && !this.legs.some((o) => o.idleStep > 0)) { l.idleStep = 1; l.idleFrom.copy(l.foot); l.idleTo.copy(home); }
        else l.foot.y = 0;
        l.stance = true; return;
      }
      if (airborne) { // tuck (fronts forward, hinds back) relative to the body
        const tuck = this.V(n.x, act.y + (l.front ? 0.45 : 0.35) * (l.L[0] + l.L[1]), n.z + (l.front ? 0.15 : -0.2) * this.size);
        l.foot.lerp(this.toWorld(tuck), 1 - Math.exp(-dt * 14)); l.stance = false; return;
      }
      const inStance = ph < duty;
      if (inStance && !l.stance) { l.plant.copy(l.foot); l.plant.y = 0; } // touchdown
      if (!inStance && l.stance) l.lift.copy(l.foot); // liftoff
      l.stance = inStance;
      if (inStance) { l.foot.copy(l.plant); }
      else {
        const u = (ph - duty) / (1 - duty), tLeft = (1 - u) * (1 - duty) / freq;
        // land half a stance ahead of where the shoulder/hip will be at touchdown
        const ahead = this.V(n.x, 0, n.z + stride * duty * 0.5);
        const land = this.toWorld(ahead); land.x += this.vel.x * tLeft; land.z += this.vel.z * tLeft;
        const s = smoothstep(u);
        l.foot.set(lerp(l.lift.x, land.x, s), Math.sin(Math.PI * u) * lift * (l.L[0] + l.L[1]) * (0.5 + 0.5 * Math.min(1, vmag / vt)), lerp(l.lift.z, land.z, s));
      }
    });
    // ---------------- body height, pitch, roll, bob
    let hipY = sp.hip, shY = sp.sh, pitchAdd = 0;
    const bob = (mix.walk * 0.012 + mix.trot * 0.025 + mix.gallop * 0.05 + mix.hop * 0.12) * this.size * (moving ? 1 : 0);
    const bobW = Math.sin(TAU * this.phase * 2) * bob;
    hipY += bobW; shY += mix.gallop * Math.sin(TAU * (this.phase + 0.25)) * 0.05 * this.size + bobW * (1 - mix.gallop);
    if (this.stalk) { hipY *= 0.82; shY *= 0.78; }
    if (this.mode === 'sit') { hipY = sp.hip * 0.42; shY = sp.sh * 0.95; }
    if (this.mode === 'lie') { hipY = Math.max(sp.rH[1] * 0.9, sp.hip * 0.32); shY = Math.max(sp.rC[1] * 0.9, sp.sh * 0.34); }
    this.breath += dt * TAU * (0.3 + 0.9 * this.exert) / Math.sqrt(Math.max(0.3, this.size));
    this.exert += ((vmag / vg) - this.exert) * (1 - Math.exp(-dt / (vmag / vg > this.exert ? 4 : 12)));
    // actions
    let jawOpen = 0, headDown = 0, frontLift = 0, hindLift = 0, swipe = 0, lunge = 0;
    if (act) {
      act.t += dt; const u = act.t / act.dur;
      if (act.name === 'jump') {
        const crouchT = 0.18;
        if (act.t < crouchT) { hipY -= 0.12 * sp.hip * Math.sin(Math.PI * act.t / crouchT / 2); shY -= 0.1 * sp.sh * Math.sin(Math.PI * act.t / crouchT / 2); }
        else { if (!act.launched) { act.launched = true; act.vy = 3.2 * Math.sqrt(this.size); } act.vy -= 9.81 * dt; act.y = Math.max(0, act.y + act.vy * dt); if (act.y <= 0 && act.vy < 0) { act.done = true; } }
        hipY += act.y; shY += act.y + (act.vy > 0 ? 0.08 : -0.05) * this.size * (act.y > 0 ? 1 : 0);
        if (act.done && act.t > act.dur) this.action = null;
      } else {
        const e = Math.sin(Math.PI * clamp(u, 0, 1));
        if (act.name === 'bite') { jawOpen = e; headDown = 0.25 * e; lunge = e; }
        if (act.name === 'thump') { hindLift = e * 0.5; }
        if (act.name === 'swipe') { swipe = e; frontLift = 0.25 * e; jawOpen = 0.6 * e; }
        if (act.name === 'pounce') { const a = clamp(u / 0.45, 0, 1); frontLift = Math.sin(Math.PI * a) * 0.6; hipY -= (u < 0.25 ? Math.sin(Math.PI * u / 0.25) : 0) * 0.15 * sp.hip; jawOpen = e * 0.7; }
        if (act.name === 'rear') { frontLift = smoothstep(Math.min(u * 2.5, (1 - u) * 2.5)) * (sp.heavy ? 1.15 : 0.95); jawOpen = 0.5 * e; swipe = sp.heavy ? e : 0; }
        if (act.name === 'buck') { hindLift = smoothstep(Math.min(u * 3, (1 - u) * 3)) * 0.7; headDown = 0.5 * e; }
        if (act.name === 'defend') { hipY *= 1 - 0.18 * e; shY *= 1 - 0.25 * e; headDown = 0.35 * e; jawOpen = 0.35 * e; }
        if (act.name === 'charge') { headDown = 0.5 * e; lunge = e; }
        if (act.name === 'stomp') { frontLift = 0.5 * e; }
        if (u >= 1) this.action = null;
      }
    }
    // rear-up / buck rotate the body about the hind / front feet
    const bodyLen = sp.len, risePitch = Math.asin(clamp(frontLift * 0.9 * (sp.sh + 0.3 * bodyLen) / bodyLen, 0, 0.95));
    shY += Math.sin(risePitch) * bodyLen * 0.9; const shZshift = -(1 - Math.cos(risePitch)) * bodyLen;
    hipY += hindLift * 0.5 * sp.hip; pitchAdd = -hindLift * 0.4;
    const roll = mix.walk * Math.sin(TAU * this.phase) * 0.03 * (moving ? 1 : 0) - this.yawRate * Math.min(1, vmag / vt) * 0.06;
    // ---------------- spine (root-local): hips, mid, chest; lateral bend with turning, gallop flex
    const bend = clamp(this.yawRate * 0.18, -0.35, 0.35);
    const flex = mix.gallop * Math.sin(TAU * this.phase) * 0.1 + mix.hop * Math.sin(TAU * this.phase) * 0.15;
    const H = this.V(0, hipY, -bodyLen / 2), C = this.V(0, shY, bodyLen / 2 + shZshift);
    C.x += Math.sin(bend) * bodyLen * 0.3; H.x -= Math.sin(bend) * bodyLen * 0.12;
    const Mid = H.clone().add(C).multiplyScalar(0.5); Mid.y += flex * bodyLen * 0.3 - 0.02 * this.size; Mid.x += Math.sin(bend) * bodyLen * 0.1;
    const breathS = 1 + 0.025 * Math.sin(this.breath) * (0.5 + this.exert);
    const [rHx, rHy, rHz] = sp.rH, [rCx, rCy, rCz] = sp.rC;
    const T = this.THREE, place = (o, p, sx, sy, sz, dirFrom, dirTo, rollZ = 0) => {
      o.position.copy(p); o.scale.set(sx, sy, sz);
      if (dirFrom && dirTo) { const d = dirTo.clone().sub(dirFrom).normalize(); o.quaternion.setFromUnitVectors(this.V(0, 0, 1), d); if (rollZ) o.rotateZ(rollZ); }
    };
    place(this.tHip, H, rHx, rHy, rHz, H, Mid, roll); place(this.tChest, C, rCx * breathS, rCy * breathS, rCz, Mid, C, roll);
    place(this.tMid, Mid, (rHx + rCx) / 2 * breathS, (rHy + rCy) / 2 * 0.95 * breathS, bodyLen * 0.42, H, C, roll);
    const belly = Mid.clone(); belly.y -= (rHy + rCy) * 0.18; place(this.tBelly, belly, (rHx + rCx) * 0.42, (rHy + rCy) * 0.32, bodyLen * 0.38, H, C, roll);
    const cf = C.clone(); cf.z += rCz * 0.45; cf.y -= rCy * 0.15; place(this.tChestFront, cf, rCx * 0.72, rCy * 0.72, rCz * 0.6, Mid, C, roll);
    if (this.stripes) this.stripes.forEach((o, k) => { const p = H.clone().lerp(C, 0.25 + 0.25 * k); p.y += (rHy + rCy) * 0.38; place(o, p, (rHx + rCx) * 0.38, 0.012 * this.size + 0.01, 0.03 * this.size + 0.012, H, C); });
    // ---------------- neck & head: looks where it is going, scans when idle, stabilised against bob
    this.lookT -= dt;
    if (this.lookT <= 0) { this.lookT = 1.5 + Math.random() * 3; this.lookGoal = vmag > 0.3 ? [0, 0] : [(Math.random() - 0.5) * 1.4, (Math.random() - 0.4) * 0.4]; }
    const steer = clamp(turnErr, -0.8, 0.8) * (mag > 0.05 ? 1 : 0);
    this.lookYaw += ((vmag > 0.3 ? steer * 0.8 : this.lookGoal[0]) - this.lookYaw) * (1 - Math.exp(-dt * 4));
    this.lookPitch += ((vmag > 0.3 ? 0 : this.lookGoal[1]) - this.lookPitch) * (1 - Math.exp(-dt * 3));
    const [nl, na] = sp.neck, nA = na * Math.PI / 180 - headDown * 0.8 + this.lookPitch * 0.5 - frontLift * 0.3 + (this.stalk ? -0.5 : 0) + (this.mode === 'lie' ? -0.2 : 0);
    const nDir = this.V(Math.sin(this.lookYaw * 0.5) * Math.cos(nA), Math.sin(nA), Math.cos(this.lookYaw * 0.5) * Math.cos(nA)).normalize();
    const N0 = C.clone().add(this.V(0, rCy * 0.35, rCz * 0.55));
    N0.y -= bobW * 0.6; // stabilise
    const N1 = N0.clone().addScaledVector(nDir, nl * 0.5 + lunge * 0.15 * this.size), N2 = N0.clone().addScaledVector(nDir, nl + lunge * 0.25 * this.size);
    const neckR = (rCx * 0.55 + (sp.head[0]) * 0.5) * 0.7;
    this.neckParts.forEach((o, k) => { const a = k ? N1 : N0, b = k ? N2 : N1; this.limb(o, a, b, neckR * (1 - 0.15 * k)); });
    if (this.maneParts) this.maneParts.forEach((o, k) => { const p = N0.clone().lerp(N2, (k + 0.5) / 4); p.y += neckR * 0.85; place(o, p, neckR * 0.35, neckR * 0.45, nl * 0.16, N0, N2); });
    const [hr, hl, hw] = sp.head, hYaw = this.lookYaw, hPitch = -(sp.hp ?? 0.35) - headDown + this.lookPitch * 0.6 + (this.mode === 'lie' ? 0.2 : 0);
    const hDir = this.V(Math.sin(hYaw) * Math.cos(hPitch), Math.sin(hPitch), Math.cos(hYaw) * Math.cos(hPitch)).normalize();
    const S = N2.clone().addScaledVector(hDir, hr * 0.4); S.y += hr * 0.15;
    place(this.skull, S, hr * 1.0, hr * 0.92, hr * 1.1, S, S.clone().add(hDir));
    const snoutP = S.clone().addScaledVector(hDir, hr * 0.75 + hl * 0.45); snoutP.y -= hr * 0.18;
    place(this.snout, snoutP, hw * 1.05, hw * 0.95, hl * 0.6, S, snoutP);
    const jawP = snoutP.clone(); jawP.y -= hw * (0.55 + jawOpen * 0.6); jawP.addScaledVector(hDir, -hl * 0.05);
    place(this.jaw, jawP, hw * 0.85, hw * 0.4, hl * 0.52, S, snoutP.clone().addScaledVector(this.V(0, -1, 0), jawOpen * hl * 0.6));
    const noseP = snoutP.clone().addScaledVector(hDir, hl * 0.55); place(this.nose, noseP, hw * 0.42, hw * 0.34, hw * 0.3);
    if (sp.trunk) { this.nose.visible = false; this.snout.scale.multiplyScalar(0.9); }
    const right = this.V(Math.cos(hYaw), 0, -Math.sin(hYaw));
    this.eyes.forEach((o, k) => { const s = k ? 1 : -1, p = S.clone().addScaledVector(hDir, hr * 0.62).addScaledVector(right, s * hr * 0.55); p.y += hr * 0.22; place(o, p, hr * 0.13, hr * 0.13, hr * 0.1); });
    // ears (flick now and then)
    this.earT -= dt; if (this.earT <= 0) { this.earT = 1 + Math.random() * 4; this.earFlick[Math.random() < 0.5 ? 0 : 1] = 1; }
    this.ears.forEach((o, k) => {
      const s = k ? 1 : -1; this.earFlick[k] = Math.max(0, this.earFlick[k] - dt * 5); const fl = Math.sin(this.earFlick[k] * Math.PI) * 0.4;
      const es = sp.earSize, base = S.clone().addScaledVector(right, s * hr * 0.55); base.y += hr * 0.7;
      if (sp.ears === 'floppy') { const tip = base.clone().addScaledVector(right, s * es * 0.35); tip.y -= es * 0.9; tip.addScaledVector(hDir, -es * 0.1); this.limb(o, base, tip, es * 0.32, 0.4); }
      else if (sp.ears === 'elephant') { const p = S.clone().addScaledVector(right, s * hr * 1.0).addScaledVector(hDir, -hr * 0.3); p.y -= hr * 0.1; place(o, p, es * 0.15, es * 0.75, es * 0.6, S, S.clone().addScaledVector(right, s)); o.rotateY(s * (0.2 + 0.25 * Math.sin(this.time * 1.3 + k))); }
      else if (sp.ears === 'round') { place(o, base, es, es * 0.85, es * 0.5, S, base.clone().add(hDir)); }
      else { // cone ears pointing up (long for rabbits, oval for deer)
        const up = this.V(s * 0.35 + fl * s, 1, -0.2 + (sp.ears === 'long' ? -0.25 : 0)).normalize();
        o.position.copy(base).addScaledVector(up, es * 0.5); o.scale.set(es * (sp.ears === 'oval' ? 0.32 : 0.28), es, es * (sp.ears === 'long' ? 0.15 : 0.18));
        o.quaternion.setFromUnitVectors(this.V(0, 1, 0), up);
      }
    });
    if (this.maneHead) place(this.maneHead, S.clone().addScaledVector(hDir, -hr * 0.45), hr * 1.75, hr * 1.7, hr * 1.25, S, S.clone().add(hDir));
    if (this.antlers) { // two beams with tines
      this.antlers.forEach((o, k) => {
        const s = k < 4 ? 1 : -1, j = k % 4, base = S.clone().addScaledVector(right, s * hr * 0.35); base.y += hr * 0.75;
        const beamUp = this.V(s * 0.45, 1, -0.35).normalize(), a = base.clone().addScaledVector(beamUp, j === 0 ? 0 : 0.12 + 0.07 * j);
        const d = j === 0 ? beamUp : this.V(s * 0.25, 1, 0.45 - 0.2 * j).normalize(), b = a.clone().addScaledVector(d, j === 0 ? 0.32 : 0.13);
        this.limb(o, a, b, 0.012 + (j === 0 ? 0.006 : 0));
      });
    }
    if (this.trunk) { // hanging trunk that curls and swings
      let p = snoutP.clone().addScaledVector(hDir, hl * 0.4), d = hDir.clone(); const curl = 0.25 + 0.2 * Math.sin(this.time * 0.9) + jawOpen * 0.6;
      this.trunk.forEach((o, k) => { d = d.clone().lerp(this.V(Math.sin(this.time * 1.1 + k) * 0.15, -1, 0.15 + curl * k * 0.15), 0.45).normalize(); const q = p.clone().addScaledVector(d, 0.19); this.limb(o, p, q, 0.11 * (1 - k * 0.11)); p = q; });
    }
    if (this.tusks) this.tusks.forEach((o, k) => { const s = k ? 1 : -1, a = snoutP.clone().addScaledVector(right, s * hw * 0.8); a.y -= hw * 0.3; const b = a.clone().addScaledVector(hDir, 0.45).add(this.V(0, -0.12, 0)); b.addScaledVector(right, s * 0.05); this.limb(o, a, b, 0.045); });
    // ---------------- legs: two-bone IK from shoulder / hip to wrist / hock, then the foot segment
    this.legs.forEach((l, i) => {
      const n = this.neutral(l), attach = (l.front ? C : H).clone();
      attach.x += l.side * (l.front ? rCx : rHx) * 0.62; attach.y -= (l.front ? rCy : rHy) * 0.35; attach.z = (l.front ? C.z + rCz * 0.15 : H.z - rHz * 0.1);
      let footL = this.toLocal(l.foot);
      // actions that lift legs
      const swingUp = (l.front && frontLift > 0.02) || (!l.front && hindLift > 0.02) || (l.front && swipe > 0.02 && l.side === (act ? act.side : 1));
      if (swingUp) {
        const k = l.front ? Math.max(frontLift, swipe) : hindLift, reach = (l.L[0] + l.L[1]) * 0.85;
        const tgt = l.front ? this.V(n.x, attach.y - reach * (0.75 - 0.6 * k), attach.z + reach * (0.35 + 0.45 * k) + (swipe ? Math.sin(this.time * 10) * 0.15 * this.size : 0)) : this.V(n.x, attach.y - reach * 0.6 + reach * 0.5 * k, attach.z - reach * (0.4 + 0.5 * k));
        footL = footL.clone().lerp(tgt, Math.min(1, k * 2.5));
      }
      if (this.mode === 'sit' && !l.front) footL = this.V(n.x, 0, H.z + (l.L[0] + l.L[1]) * 0.25);
      if (this.mode === 'lie') footL = this.V(n.x * 1.3, 0, l.front ? attach.z + (l.L[0] + l.L[1]) * 0.7 : attach.z - (l.L[0] + l.L[1]) * 0.3);
      const [a, b, c] = l.L, sw = l.stance ? 0 : 1, liftAmt = clamp(footL.y / (0.3 * (a + b)), 0, 1);
      // foot segment direction (digitigrade): hind hock behind the paw, front wrist above it; folds in swing
      // direction from the foot up to the wrist (front) / hock (hind); the paw folds back during swing
      let fdir;
      if (sp.column) fdir = this.V(0, 1, 0);
      else if (l.front) { const phi = -0.15 + 1.3 * liftAmt * sw; fdir = this.V(0, Math.cos(phi), Math.sin(phi)); }
      else { const psi = 0.45 + 0.6 * liftAmt * sw; fdir = this.V(0, Math.cos(psi), -Math.sin(psi)); }
      const ankle = footL.clone().addScaledVector(fdir, c);
      const knee = this.ik(attach, ankle, a, b, l.front ? this.V(0, -0.2, -1) : this.V(0, -0.2, 1));
      const r = l.thick;
      this.limb(l.segs[0], attach, knee, r * 1.05); this.limb(l.segs[1], knee, ankle, r * 0.78); this.limb(l.segs[2], ankle, footL, r * (sp.hooves ? 0.62 : 0.55));
      const padP = footL.clone(); padP.y += r * 0.35; place(l.pad, padP, r * (sp.column ? 1.05 : 0.85), r * 0.5, r * (sp.column ? 1.05 : 1.15));
    });
    // ---------------- tail: spring chain behind the hips, wagging when idle
    const [tl, tA, tr, style] = sp.tail;
    const baseP = H.clone().add(this.V(0, rHy * 0.35, -rHz * 0.92));
    const wagA = sp.wag * (vmag < 0.3 ? 1 : 0.35) * Math.sin(this.time * (sp.wag > 0.8 ? 11 : 3)) * 0.6;
    const tailUp = (tA * Math.PI / 180) + (this.stalk ? -0.4 : 0) + (act && act.name === 'defend' ? 0.5 : 0);
    const restDir = this.V(Math.sin(wagA), Math.sin(tailUp), -Math.cos(tailUp)).normalize();
    const segL = tl / this.tailN;
    // simulate in world space so the tail lags behind body motion
    const wpts = this.tailPts, worldBase = this.toWorld(baseP); worldBase.y = baseP.y;
    if (!this.tailPrev) { this.tailPrev = wpts.map(() => worldBase.clone()); for (let k = 0; k <= this.tailN; k++) { const p = this.toWorld(baseP.clone().addScaledVector(restDir, segL * k)); p.y = baseP.y + restDir.y * segL * k; wpts[k].copy(p); this.tailPrev[k].copy(p); } }
    wpts[0].copy(worldBase);
    for (let k = 1; k <= this.tailN; k++) {
      const restL = baseP.clone().addScaledVector(restDir, segL * k), restW = this.toWorld(restL); restW.y = restL.y;
      const p = wpts[k], prev = this.tailPrev[k], v = p.clone().sub(prev).multiplyScalar(0.86);
      prev.copy(p); p.add(v); p.lerp(restW, 1 - Math.exp(-dt * (style === 'hair' ? 9 : 16)));
      p.y -= (style === 'hair' ? 2.5 : 0.6) * dt * dt * 30;
      const d = p.clone().sub(wpts[k - 1]); d.setLength(segL); p.copy(wpts[k - 1]).add(d);
    }
    this.tailParts.forEach((o, k) => {
      const a = this.toLocal(wpts[k]), b = this.toLocal(wpts[k + 1]); a.y = wpts[k].y; b.y = wpts[k + 1].y;
      const taper = style === 'bushy' ? (0.6 + Math.sin(Math.PI * (k + 0.5) / this.tailN) * 0.9) : style === 'puff' ? 1.6 : style === 'tuft' && k === this.tailN - 1 ? 2.4 : 1 - 0.6 * k / this.tailN;
      if (style === 'puff') { const m = a.clone().add(b).multiplyScalar(0.5); place(o, m, tr * taper, tr * taper, tr * taper, a, b); }
      else this.limb(o, a, b, tr * taper);
    });
    // root transform
    this.root.position.set(this.pos.x, 0, this.pos.z); this.root.rotation.set(0, this.yaw, 0);
  }
  // two-bone IK: joint between A and T with segment lengths l1, l2, bending toward pole
  ik(A, T, l1, l2, pole) {
    const d = T.clone().sub(A), dist = clamp(d.length(), Math.abs(l1 - l2) + 1e-4, l1 + l2 - 1e-4); d.normalize();
    const a = (l1 * l1 - l2 * l2 + dist * dist) / (2 * dist), h = Math.sqrt(Math.max(0, l1 * l1 - a * a));
    const p = pole.clone().addScaledVector(d, -pole.dot(d)); if (p.lengthSq() < 1e-8) p.set(0, 0, 1); p.normalize();
    return A.clone().addScaledVector(d, a).addScaledVector(p, h);
  }
  // place a unit capsule (radius 1, cylinder 1, total height 3) between two points with radius r
  limb(o, a, b, r, flat = 1) {
    const d = b.clone().sub(a), len = Math.max(1e-4, d.length());
    o.position.copy(a).add(b).multiplyScalar(0.5);
    o.quaternion.setFromUnitVectors(this.V(0, 1, 0), d.multiplyScalar(1 / len));
    o.scale.set(r, (len + 1.4 * r) / 3, r * flat);
  }
  get target() { return this.V(this.pos.x, Math.max(this.sp.hip, this.sp.sh) * 0.9, this.pos.z); }
}
