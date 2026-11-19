// Offline renderer for SOMA showcase videos: GPU-skinned NVIDIA SOMA body mesh driven by
// joint transforms that the real MotionBricks SOMA runtime generated (showcase/build/generate_motions.py).
import * as THREE from 'three';

const W = 1920, H = 1080;
const Q = Object.fromEntries(new URLSearchParams(location.search));
const SHADOW = +(Q.shadow || 2048), AA = Q.aa !== '0', CHEAP_GROUND = Q.ground === 'lambert';
const NJ = 77, STRIDE = NJ * 7 + 4; // per frame: 77 x (px py pz qx qy qz qw) + 4 foot contacts

function gridTexture() {
  const c = document.createElement('canvas'); c.width = c.height = 512;
  const g = c.getContext('2d');
  g.fillStyle = '#c9cfd6'; g.fillRect(0, 0, 512, 512);
  g.fillStyle = '#bfc6ce'; g.fillRect(0, 0, 256, 256); g.fillRect(256, 256, 256, 256);
  g.strokeStyle = '#9aa4ae'; g.lineWidth = 3; g.strokeRect(0, 0, 512, 512);
  g.strokeStyle = '#aeb6bf'; g.lineWidth = 1.5;
  g.beginPath(); g.moveTo(256, 0); g.lineTo(256, 512); g.moveTo(0, 256); g.lineTo(512, 256); g.stroke();
  const t = new THREE.CanvasTexture(c);
  t.wrapS = t.wrapT = THREE.RepeatWrapping; t.repeat.set(200, 200); // 512px = 2 m (1 m checks)
  t.anisotropy = 8; t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

function tileRects(layout) {
  if (layout === '1') return [[0, 0, W, H]];
  if (layout === '2') return [[0, 0, W / 2, H], [W / 2, 0, W / 2, H]];
  if (layout === '3') return [[0, 0, W / 3, H], [W / 3, 0, W / 3, H], [2 * W / 3, 0, W / 3, H]];
  return [[0, 0, W / 2, H / 2], [W / 2, 0, W / 2, H / 2], [0, H / 2, W / 2, H / 2], [W / 2, H / 2, W / 2, H / 2]];
}

function makeProp(p) {
  const g = new THREE.Group();
  const wood = new THREE.MeshStandardMaterial({ color: '#b5763f', roughness: 0.7 });
  const dark = new THREE.MeshStandardMaterial({ color: '#3d4650', roughness: 0.6 });
  if (p.type === 'hurdle') {
    const bar = new THREE.Mesh(new THREE.BoxGeometry(1.5, 0.08, 0.08), wood); bar.position.y = p.height; g.add(bar);
    for (const x of [-0.72, 0.72]) {
      const post = new THREE.Mesh(new THREE.BoxGeometry(0.06, p.height, 0.06), dark);
      post.position.set(x, p.height / 2, 0); g.add(post);
      const foot = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.03, 0.45), dark);
      foot.position.set(x, 0.015, 0); g.add(foot);
    }
  } else if (p.type === 'chair') {
    const s = p.seat, seat = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.05, 0.48), wood);
    seat.position.set(0, s - 0.025, 0); g.add(seat);
    for (const [x, z] of [[-0.22, -0.21], [0.22, -0.21], [-0.22, 0.21], [0.22, 0.21]]) {
      const leg = new THREE.Mesh(new THREE.BoxGeometry(0.045, s - 0.05, 0.045), dark);
      leg.position.set(x, (s - 0.05) / 2, z); g.add(leg);
    }
    const back = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.45, 0.04), wood);
    back.position.set(0, s + 0.25, -0.24); g.add(back);
  }
  g.position.set(p.x, 0, p.z); g.rotation.y = p.yaw;
  g.traverse(o => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
  return g;
}

async function loadMesh() {
  const meta = await (await fetch('../cache/soma_mesh.json')).json();
  const buf = await (await fetch('../cache/soma_mesh.bin')).arrayBuffer();
  const view = (k, T) => new T(buf, meta[k].offset, meta[k].length);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(view('positions', Float32Array), 3));
  geo.setIndex(new THREE.BufferAttribute(view('indices', Uint32Array), 1));
  geo.setAttribute('skinIndex', new THREE.Uint16BufferAttribute(view('skinIndex', Uint16Array), 4));
  geo.setAttribute('skinWeight', new THREE.BufferAttribute(view('skinWeight', Float32Array), 4));
  geo.computeVertexNormals();
  const inv = view('inverseBind', Float32Array);
  const inverses = [];
  for (let j = 0; j < NJ; j++) inverses.push(new THREE.Matrix4().fromArray(inv, j * 16));
  return { geo, inverses };
}

function buildTile(spec, tileSpec, rect, base, tex, idx, nTiles) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color('#dfe5ec');
  scene.fog = new THREE.Fog('#dfe5ec', 14, 48);
  scene.add(new THREE.HemisphereLight('#ffffff', '#7d858e', 1.35));
  const sun = new THREE.DirectionalLight('#fff6ea', 2.6);
  sun.castShadow = true; sun.shadow.mapSize.set(SHADOW, SHADOW);
  Object.assign(sun.shadow.camera, { left: -4, right: 4, top: 4, bottom: -4, near: 0.5, far: 30 });
  sun.shadow.bias = -0.0004; sun.shadow.normalBias = 0.02;
  scene.add(sun, sun.target);
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(400, 400), CHEAP_GROUND
    ? new THREE.MeshLambertMaterial({ map: tex }) : new THREE.MeshStandardMaterial({ map: tex, roughness: 0.95 }));
  ground.rotation.x = -Math.PI / 2; ground.receiveShadow = true; scene.add(ground);

  const bones = [];
  for (let j = 0; j < NJ; j++) { const b = new THREE.Bone(); bones.push(b); scene.add(b); }
  const mat = new THREE.MeshStandardMaterial({ color: '#c8bfb4', roughness: 0.52, metalness: 0.0 });
  const mesh = new THREE.SkinnedMesh(base.geo, mat);
  mesh.castShadow = true; mesh.receiveShadow = true; mesh.frustumCulled = false;
  mesh.bind(new THREE.Skeleton(bones, base.inverses), new THREE.Matrix4());
  scene.add(mesh);
  for (const p of tileSpec.props || []) scene.add(makeProp(p));

  // predicted foot contacts (model output) as soft rings under the feet
  const rings = [0, 1].map(() => {
    const r = new THREE.Mesh(new THREE.RingGeometry(0.07, 0.12, 32),
      new THREE.MeshBasicMaterial({ color: '#17b890', transparent: true, opacity: 0, depthWrite: false }));
    r.rotation.x = -Math.PI / 2; scene.add(r); return r;
  });
  let arrow = null;
  if (spec.arrows) {  // flat arrow on the ground = the gameplay input direction
    const sh = new THREE.Shape();
    sh.moveTo(-0.07, 0); sh.lineTo(0.07, 0); sh.lineTo(0.07, 0.9); sh.lineTo(0.2, 0.9);
    sh.lineTo(0, 1.25); sh.lineTo(-0.2, 0.9); sh.lineTo(-0.07, 0.9); sh.closePath();
    const geo = new THREE.ShapeGeometry(sh); geo.rotateX(Math.PI / 2);  // lie flat, pointing +Z
    arrow = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color: '#e0541b', transparent: true, opacity: 0.85,
      side: THREE.DoubleSide, depthWrite: false }));
    scene.add(arrow);
  }
  let prints = null;
  if (spec.footprints) {  // marks where the model reports a planted foot: sliding = streaks
    const n = 4000, geo = new THREE.CircleGeometry(0.04, 14).rotateX(-Math.PI / 2);
    prints = ['#0f9e7a', '#d9541c'].map(color => {  // left foot teal, right foot orange
      const m = new THREE.InstancedMesh(geo, new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.85,
        depthWrite: false }), n);
      m.count = 0; m.frustumCulled = false; scene.add(m); return m;
    });
  }
  const camera = new THREE.PerspectiveCamera(tileSpec.fov, rect[2] / rect[3], 0.05, 80);

  const el = document.createElement('div');
  el.className = 'tile' + (nTiles > 1 ? ' small' : '') + (rect[0] > 0 ? ' sep' : '') + (rect[1] > 0 ? ' sepTop' : '')
    + (nTiles > 1 && rect[1] === 0 ? ' titled' : '');
  Object.assign(el.style, { left: rect[0] + 'px', top: rect[1] + 'px', width: rect[2] + 'px', height: rect[3] + 'px' });
  el.innerHTML = `<div class="label"><b></b><span></span></div><div class="speed"></div><div class="caption"><div></div></div>`;
  el.querySelector('.label b').textContent = tileSpec.label;
  el.querySelector('.label span').textContent = tileSpec.sub;
  document.getElementById('overlay').appendChild(el);
  return { scene, camera, bones, sun, rings, arrow, prints, rect, el, spec: tileSpec, data: null };
}

let state = null;

async function init(specUrl) {
  const spec = await (await fetch(specUrl)).json();
  const renderer = new THREE.WebGLRenderer({ antialias: AA, preserveDrawingBuffer: true });
  renderer.setPixelRatio(1); renderer.setSize(W, H);
  renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.05;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.setScissorTest(true);
  document.body.prepend(renderer.domElement);
  const base = await loadMesh();
  const tex = gridTexture();
  const rects = tileRects(spec.layout);
  const dir = specUrl.substring(0, specUrl.lastIndexOf('/') + 1);
  const tiles = [];
  for (let i = 0; i < spec.tiles.length; i++) {
    const t = buildTile(spec, spec.tiles[i], rects[i], base, tex, i, spec.tiles.length);
    t.data = new Float32Array(await (await fetch(dir + spec.tiles[i].bin)).arrayBuffer());
    tiles.push(t);
  }
  const title = document.createElement('div'); title.id = 'title'; title.textContent = spec.title;
  if (spec.tiles.length > 1) document.getElementById('overlay').appendChild(title);
  const mark = document.createElement('div'); mark.id = 'mark';
  mark.textContent = 'Real NVIDIA MotionBricks SOMA model output  ·  30 fps  ·  rendered offline (three.js)';
  document.getElementById('overlay').appendChild(mark);
  state = { renderer, tiles, spec };
  window.SHOWCASE_READY = true;
  return spec.frames;
}

const tmpV = new THREE.Vector3();
function renderFrame(i) {
  const { renderer, tiles } = state;
  for (const t of tiles) {
    const n = t.spec.frames, f = Math.min(i, n - 1), o = f * STRIDE, d = t.data;
    for (let j = 0; j < NJ; j++) {
      const k = o + j * 7;
      t.bones[j].position.set(d[k], d[k + 1], d[k + 2]);
      t.bones[j].quaternion.set(d[k + 3], d[k + 4], d[k + 5], d[k + 6]);
    }
    const c = t.spec.camera[f];
    t.camera.position.set(c[0], c[1], c[2]); t.camera.lookAt(c[3], c[4], c[5]);
    const hx = d[o], hz = d[o + 2];
    t.sun.position.set(hx + 3.5, 7.5, hz + 2.5); t.sun.target.position.set(hx, 0, hz);
    const ct = o + NJ * 7; // contacts: LeftFoot, LeftToeBase, RightFoot, RightToeBase
    for (const [ri, jName, ci] of [[0, 'LeftFoot', 0], [1, 'RightFoot', 2]]) {
      const j = t.spec._jidx ? t.spec._jidx[jName] : null;
      if (j === null || j === undefined) continue;
      const k = o + j * 7;
      t.rings[ri].position.set(d[k], 0.004, d[k + 2]);
      t.rings[ri].material.opacity = Math.max(0, Math.min(1, (Math.max(d[ct + ci], d[ct + ci + 1]) - 0.4) * 1.6)) * 0.85;
    }
    if (t.arrow) {
      const g = t.spec.goal[f], m = Math.hypot(g[0], g[1]);
      t.arrow.visible = m > 0.05;
      if (m > 0.05) {
        t.arrow.position.set(hx, 0.012, hz);
        t.arrow.rotation.y = Math.atan2(g[0], g[1]);
        t.arrow.scale.setScalar(0.8 + 0.12 * m);
      }
    }
    if (t.prints && f > (t.lastPrint ?? -1)) {
      t.lastPrint = f;
      const m4 = new THREE.Matrix4();
      for (const [pi, jName, ci] of [[0, 'LeftFoot', 0], [1, 'RightFoot', 2]]) {
        const j = t.spec._jidx[jName], k = o + j * 7, pm = t.prints[pi];
        if (d[ct + ci] > 0.5 && pm.count < 4000) {
          m4.makeTranslation(d[k], 0.006, d[k + 2]);
          pm.setMatrixAt(pm.count++, m4);
          pm.instanceMatrix.needsUpdate = true;
        }
      }
    }
    t.el.querySelector('.caption div').textContent = t.spec.captions[f];
    t.el.querySelector('.speed').textContent = t.spec.speed[f].toFixed(1) + ' m/s';
    const [x, y, w, h] = t.rect;
    renderer.setViewport(x, H - y - h, w, h); renderer.setScissor(x, H - y - h, w, h);
    renderer.render(t.scene, t.camera);
  }
  return true;
}

window.SHOWCASE = {
  init: async (url) => {
    const frames = await init(url);
    const idx = {}; state.spec.joints.forEach((n, i) => { idx[n] = i; });
    for (const t of state.tiles) t.spec._jidx = idx;
    return frames;
  },
  renderFrame,
  sync: () => { const gl = state.renderer.getContext(); const px = new Uint8Array(4); gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px); },
};
