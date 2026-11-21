// Inference worker: downloads (and caches) the SOMA ONNX model, then runs whole generations
// (encode -> onnxruntime-web -> decode -> corrections -> anatomical limits) off the main thread.
import * as ort from '../vendor/ort/ort.wasm.min.mjs';
import { bind, bindLimits, encode, postProcess } from './codec.js';
import { parseGGUF, OnnxAssembler } from './gguf.js';

let session = null, binding = null, limits = null, CACHE = 'soma-models';

async function fetchPart(url, onBytes) {
  let res = null;
  try { const c = await caches.open(CACHE); res = await c.match(url); if (!res) { res = await fetch(url); if (res.ok) await c.put(url, res.clone()); } }
  catch { res = await fetch(url); }
  if (!res.ok) throw new Error(`download failed: ${url} (${res.status})`);
  const buf = new Uint8Array(await res.arrayBuffer());
  onBytes(buf.length);
  return buf;
}

async function load({ base, model, info, skeleton, jointLimits, threads }) {
  // one cache per model build (keyed by its checksum); caches of older builds are deleted
  CACHE = `soma-model-${model}-${info.sha256.slice(0, 16)}`;
  try { for (const k of await caches.keys()) if (k.startsWith('soma-model') && k !== CACHE) await caches.delete(k); } catch { /* no Cache Storage */ }
  binding = bind(skeleton);
  limits = bindLimits(jointLimits, skeleton);
  ort.env.wasm.wasmPaths = new URL('../vendor/ort/', self.location.href).href;
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.max(1, threads) : 1;
  let all, off = 0, done = 0;
  const t0 = performance.now();
  if (info.format === 'gguf') { // one split GGUF: shard 1 = metadata (+ animal net), the rest rebuilt into the ONNX file shard by shard
    let asm = null;
    for (const p of info.parts) {
      const b = await fetchPart(`${base}${model}/${p.file}`, (n) => { done += n; postMessage({ type: 'progress', done, total: info.bytes }); });
      const g = parseGGUF(b.buffer);
      if (!asm) { const meta = g.kv['soma.splice_json']; if (!meta) throw new Error('GGUF shard 1 has no SOMA metadata'); asm = new OnnxAssembler(JSON.parse(meta)); }
      asm.add(g);
    }
    if (!asm.complete) throw new Error('GGUF: some SOMA tensors are missing');
    all = asm.out;
  } else {
    all = new Uint8Array(info.bytes);
    for (const p of info.parts) {
      const b = await fetchPart(`${base}${model}/${p.file}`, (n) => { done += n; postMessage({ type: 'progress', done, total: info.bytes }); });
      all.set(b, off); off += b.length;
    }
  }
  postMessage({ type: 'status', text: 'Starting the SOMA network...' });
  if (session) { try { await session.release(); } catch { /* ignore */ } }
  session = await ort.InferenceSession.create(all, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
  postMessage({ type: 'ready', threads: ort.env.wasm.numThreads, seconds: (performance.now() - t0) / 1000 });
}

async function generate(msg) {
  const t0 = performance.now();
  const J = binding.sk.names.length, nT = msg.target.length;
  const guides = [...msg.context, ...msg.target];
  while (guides.length < 8) guides.push(guides[guides.length - 1]);
  const { feeds, align } = encode(binding, guides, nT, msg.allowed);
  const inputs = {};
  for (const [k, v] of Object.entries(feeds)) {
    const dims = { global_root_values: [1, 8, 5], local_root_values: [1, 8, 4], local_poses: [1, 8, 267], has_global_root_values: [1, 8],
      has_local_root_values: [1, 8], has_local_poses: [1, 8], allowed_pred_num_tokens: [1, 11] }[k];
    inputs[k] = new ort.Tensor(v instanceof Int32Array ? 'int32' : 'float32', v, dims);
  }
  const t1 = performance.now();
  const out = await session.run(inputs);
  const t2 = performance.now();
  const tokens = Number(out.pred_num_tokens.data[0]);
  const { frames, contacts } = postProcess(binding, msg.anatomical ? limits : null, out.pred_global_motions.data, tokens, align,
    guides, nT, { velocityMatching: msg.velocityMatching, correctEndpoints: true });
  const flat = new Float64Array(frames.length * J * 7);
  frames.forEach((f, i) => flat.set(f, i * J * 7));
  const cflat = new Float32Array(contacts.flat());
  postMessage({ type: 'result', id: msg.id, frames: flat, contacts: cflat, count: frames.length, tokens,
    ms: { encode: t1 - t0, network: t2 - t1, total: performance.now() - t0 } }, [flat.buffer, cflat.buffer]);
}

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'load') await load(m);
    else if (m.type === 'generate') await generate(m);
  } catch (err) {
    postMessage({ type: 'error', id: m.id, error: String(err && err.message || err) });
  }
};
