// GGUF (v2/v3) reading for the playground's single model file (tools/gguf/build_gguf.py): a split GGUF holding the
// SOMA human model as an ONNX graph with its INT8 matrix weights stored as ggml Q5_0 blocks; shard 1 also holds the
// metadata (soma.splice_json) that says how to put the ONNX file back together.

const SIZE = { 0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8 };

export function parseGGUF(buf) {
  const dv = new DataView(buf), dec = new TextDecoder(); let p = 0;
  const u32 = () => { const v = dv.getUint32(p, true); p += 4; return v; };
  const u64 = () => { const v = Number(dv.getBigUint64(p, true)); p += 8; return v; };
  const str = () => { const n = u64(), v = dec.decode(new Uint8Array(buf, p, n)); p += n; return v; };
  if (dec.decode(new Uint8Array(buf, 0, 4)) !== 'GGUF') throw new Error('not a GGUF file');
  p = 4; const version = u32(); if (version < 2) throw new Error(`GGUF v${version} unsupported`);
  const nT = u64(), nKV = u64(), kv = {};
  const val = (t) => {
    if (t === 8) return str();
    if (t === 4) return u32(); if (t === 10) return u64(); if (t === 2) { const v = dv.getUint16(p, true); p += 2; return v; }
    if (t === 5) { const v = dv.getInt32(p, true); p += 4; return v; } if (t === 6) { const v = dv.getFloat32(p, true); p += 4; return v; }
    if (t === 9) { const et = u32(), n = u64(), a = []; for (let i = 0; i < n; i++) a.push(val(et)); return a; }
    const v = SIZE[t]; if (!v) throw new Error(`GGUF value type ${t}`); p += v; return null;
  };
  for (let i = 0; i < nKV; i++) { const k = str(); kv[k] = val(u32()); }
  const tensors = [];
  for (let i = 0; i < nT; i++) {
    const name = str(), nd = u32(), dims = []; for (let d = 0; d < nd; d++) dims.push(u64());
    const type = u32(), offset = u64();
    tensors.push({ name, dims, type, offset });
  }
  const align = kv['general.alignment'] || 32, data = Math.ceil(p / align) * align;
  for (const t of tensors) t.offset += data;
  return { kv, tensors, buf };
}


// ggml Q5_0 block (32 weights: f16 scale, 32 high bits, 16 bytes of low nibbles) -> int8, rounded onto the int8 grid
const F16 = new Float32Array(65536);
for (let h = 0; h < 65536; h++) {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
  F16[h] = e === 0 ? s * m * 5.960464477539063e-8 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * Math.pow(2, e - 15);
}
export function q5_0ToInt8(src, srcOff, n, dst, dstOff) {
  // the packer's block scales are int8 values / 16 (exact in float16), so decoding is integer math:
  // round((q - 16) * m / 16) half up == ((q - 16) * m + 8) >> 4
  const nb = Math.ceil(n / 32);
  for (let b = 0; b < nb; b++) {
    const o = srcOff + b * 22, m = Math.round(F16[src[o] | (src[o + 1] << 8)] * 16) | 0, qh = (src[o + 2] | (src[o + 3] << 8) | (src[o + 4] << 16) | (src[o + 5] << 24)) >>> 0;
    const base = dstOff + b * 32, lim = Math.min(32, n - b * 32);
    if (lim === 32) {
      for (let j = 0; j < 16; j++) {
        const q = src[o + 6 + j];
        let v = (((q & 0x0f) | (((qh >>> j) & 1) << 4)) - 16) * m + 8 >> 4; dst[base + j] = v < -128 ? -128 : v > 127 ? 127 : v;
        v = (((q >> 4) | (((qh >>> (j + 16)) & 1) << 4)) - 16) * m + 8 >> 4; dst[base + j + 16] = v < -128 ? -128 : v > 127 ? 127 : v;
      }
    } else {
      for (let j = 0; j < 32; j++) {
        if (j >= lim) break;
        const q = src[o + 6 + (j & 15)], lo = j < 16 ? q & 0x0f : q >> 4;
        const v = ((lo | (((qh >>> j) & 1) << 4)) - 16) * m + 8 >> 4; dst[base + j] = v < -128 ? -128 : v > 127 ? 127 : v;
      }
    }
  }
}

// Rebuild the SOMA ONNX file shard by shard into one buffer (each shard can be dropped once copied).
export class OnnxAssembler {
  constructor(meta) {
    this.meta = meta; this.out = new Uint8Array(meta.onnx_bytes); this.out8 = new Int8Array(this.out.buffer);
    this.byTensor = Object.fromEntries(meta.splice.map((s) => [s.tensor, s]));
    // graph segments: graph range [g0, g1) lands at onnx offset o0
    this.seg = []; let g = 0, o = 0;
    for (const s of meta.splice) { const len = s.offset - o; this.seg.push([g, g + len, o]); g += len; o = s.offset + s.n; }
    this.seg.push([g, meta.graph_bytes, o]);
    this.chunk = meta.graph_chunk_bytes || 12 * 1024 * 1024; this.done = 0;
  }
  add(g) {
    const u8 = new Uint8Array(g.buf);
    for (const t of g.tensors) {
      if (t.name.startsWith('soma.graph.')) {
        const k = +t.name.slice(11), a = k * this.chunk, len = t.dims[0];
        for (const [g0, g1, o0] of this.seg) {
          const lo = Math.max(a, g0), hi = Math.min(a + len, g1);
          if (lo < hi) this.out.set(u8.subarray(t.offset + lo - a, t.offset + hi - a), o0 + lo - g0);
        }
        this.done++;
      } else if (this.byTensor[t.name]) {
        const s = this.byTensor[t.name]; if (t.type !== 6) throw new Error(`${t.name}: expected Q5_0`);
        q5_0ToInt8(u8, t.offset, s.n, this.out8, s.offset); this.done++;
      }
    }
  }
  get complete() { return this.done === this.meta.graph_chunks + this.meta.splice.length; }
}
