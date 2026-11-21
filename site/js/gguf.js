// Minimal GGUF (v2/v3) reader for the animal network file: returns the network's metadata and float16 tensors
// in the layout DogNet expects ({ meta: { ...json, tensors: [{ name, shape, offset }] }, buffer }).
export function readGGUF(buf) {
  const dv = new DataView(buf), dec = new TextDecoder(); let p = 0;
  const u32 = () => { const v = dv.getUint32(p, true); p += 4; return v; };
  const u64 = () => { const v = Number(dv.getBigUint64(p, true)); p += 8; return v; };
  const str = () => { const n = u64(), v = dec.decode(new Uint8Array(buf, p, n)); p += n; return v; };
  if (dec.decode(new Uint8Array(buf, 0, 4)) !== 'GGUF') throw new Error('not a GGUF file');
  p = 4; const version = u32(); if (version < 2) throw new Error(`GGUF v${version} unsupported`);
  const nT = u64(), nKV = u64(), kv = {};
  const SIZE = { 0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8 };
  const val = (t) => {
    if (t === 8) return str();
    if (t === 4) return u32(); if (t === 10) return u64(); if (t === 6) { const v = dv.getFloat32(p, true); p += 4; return v; }
    if (t === 9) { const et = u32(), n = u64(), a = []; for (let i = 0; i < n; i++) a.push(val(et)); return a; }
    const v = SIZE[t]; if (!v) throw new Error(`GGUF value type ${t}`); p += v; return null;
  };
  for (let i = 0; i < nKV; i++) { const k = str(); kv[k] = val(u32()); }
  const tensors = [];
  for (let i = 0; i < nT; i++) {
    const name = str(), nd = u32(), dims = []; for (let d = 0; d < nd; d++) dims.push(u64());
    const type = u32(), offset = u64(); if (type !== 1) throw new Error(`tensor ${name}: only float16 supported`);
    tensors.push({ name, shape: dims.reverse(), offset });
  }
  const align = kv['general.alignment'] || 32, data = Math.ceil(p / align) * align;
  const meta = JSON.parse(kv['mann-quadruped.meta_json']);
  meta.tensors = tensors;
  return { meta, buffer: buf.slice(data), kv };
}
