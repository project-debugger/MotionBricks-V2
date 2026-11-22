#!/usr/bin/env python3
"""Pack the playground's SOMA model into ONE GGUF model, split into shards like llama.cpp's gguf-split.

    .venv/bin/python tools/gguf/build_gguf.py   # -> site/models/balanced/motionbricks-000NN-of-000MM.gguf

Contents (every shard is a valid GGUF file with the split.no / split.count / split.tensors.count keys; shard 1 also
holds the metadata): the SOMA human model as an onnxruntime graph:
      - the dynamic-INT8 weights of the matrix multiplies (MatMulInteger, 136 M weights) re-coded with ggml's Q5_0
        blocks (5.5 bits per weight); at load they are decoded back onto the INT8 grid, so the runtime graph, speed and
        memory are those of the INT8 model. Eval set: MPJPE 7.07 cm vs 7.10 cm for the INT8 file (Q4_0: 7.27 cm);
      - the rest of the ONNX file ("soma.graph.NNN" byte chunks, GGML_TYPE_I8) with those weights cut out, and a
        splice table (soma.splice_json) saying where each decoded weight goes back in;
GGUF is only the container here: llama.cpp cannot run these networks (no ggml graph); the browser does (js/gguf.js).
"""
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

sys.path.insert(0, str(Path(__file__).parent))
from ggml_quant import CODECS  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "site/models/balanced/model.onnx.local"
OUT = ROOT / "site/models/balanced"
SHARD = 14 * 1024 * 1024
ALIGN = 32
T_U16, T_I32, T_U32, T_STR = 2, 5, 4, 8
GGML_F16, GGML_I8, GGML_Q5_0 = 1, 24, 6


def gstr(s: str) -> bytes:
    b = s.encode(); return struct.pack("<Q", len(b)) + b


def kv(key, typ, val):
    out = gstr(key) + struct.pack("<I", typ)
    return out + {T_STR: lambda: gstr(val), T_U16: lambda: struct.pack("<H", val), T_I32: lambda: struct.pack("<i", val), T_U32: lambda: struct.pack("<I", val)}[typ]()


def main():
    onnx_bytes = SRC.read_bytes()
    model = onnx.load_from_string(onnx_bytes)
    B = {n.input[1] for n in model.graph.node if n.op_type == "MatMulInteger"}
    enc = CODECS["q5_0"][0]
    weights = []  # (name, original offset, int8 bytes, dims)
    for t in model.graph.initializer:
        if t.name in B and t.data_type == onnx.TensorProto.INT8 and t.dims[-1] % 32 == 0 and t.raw_data:
            off = onnx_bytes.find(t.raw_data)
            assert off >= 0 and onnx_bytes.find(t.raw_data, off + 1) < 0, t.name
            weights.append((t.name, off, numpy_helper.to_array(t), list(t.dims)))
    weights.sort(key=lambda w: w[1])
    # graph = file minus the weight regions
    parts, pos, splice = [], 0, []
    for i, (name, off, w, dims) in enumerate(weights):
        parts.append(onnx_bytes[pos:off]); pos = off + w.size
        splice.append({"tensor": f"soma.w.{i:03d}", "offset": off, "n": int(w.size), "onnx_name": name})
    parts.append(onnx_bytes[pos:])
    graph = b"".join(parts)
    # tensors: (name, ggml dims, type, bytes)
    tensors = []
    CH = 12 * 1024 * 1024
    for k in range(0, len(graph), CH):
        tensors.append((f"soma.graph.{k // CH:03d}", [len(graph[k:k + CH])], GGML_I8, graph[k:k + CH]))
    for i, (name, off, w, dims) in enumerate(weights):
        tensors.append((f"soma.w.{i:03d}", list(reversed(dims)), GGML_Q5_0, enc(w)))
    # pack greedily into shards (tensor data + headroom for the header)
    shards, cur, size = [], [], 0
    for t in tensors:
        n = len(t[3]) + ALIGN
        if cur and size + n > SHARD - 512 * 1024:
            shards.append(cur); cur, size = [], 0
        cur.append(t); size += n
    shards.append(cur)
    meta = {"source": SRC.name, "onnx_bytes": len(onnx_bytes), "onnx_sha256": hashlib.sha256(onnx_bytes).hexdigest(), "graph_bytes": len(graph),
            "graph_chunks": sum(1 for t in tensors if t[0].startswith("soma.graph")), "graph_chunk_bytes": CH, "codec": "q5_0", "splice": splice}
    for old in OUT.glob("motionbricks-*.gguf"): old.unlink()
    files, total = [], 0
    for si, sh in enumerate(shards):
        kvs = [kv("split.no", T_U16, si), kv("split.count", T_U16, len(shards)), kv("split.tensors.count", T_I32, len(tensors)), kv("general.alignment", T_U32, ALIGN)]
        if si == 0:
            kvs += [kv("general.architecture", T_STR, "motionbricks"), kv("general.name", T_STR, "MotionBricks V2 playground: SOMA human motion model"),
                    kv("general.license", T_STR, "NVIDIA Open Model License (MotionBricks SOMA weights, quantized)"),
                    kv("soma.format", T_STR, "onnx graph + ggml Q5_0 MatMulInteger weights"), kv("soma.splice_json", T_STR, json.dumps(meta, separators=(",", ":")))]
        infos, datas, off = [], [], 0
        for name, dims, typ, raw in sh:
            pad = -off % ALIGN; datas.append(b"\0" * pad); off += pad
            infos.append(gstr(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims) + struct.pack("<IQ", typ, off))
            datas.append(raw); off += len(raw)
        head = b"GGUF" + struct.pack("<IQQ", 3, len(sh), len(kvs)) + b"".join(kvs) + b"".join(infos)
        head += b"\0" * (-len(head) % ALIGN)
        body = head + b"".join(datas)
        f = OUT / f"motionbricks-{si + 1:05d}-of-{len(shards):05d}.gguf"; f.write_bytes(body)
        files.append({"file": f.name, "bytes": len(body)}); total += len(body)
        assert len(body) <= 15_000_000, (f, len(body))
    h = hashlib.sha256()
    for f in files: h.update((OUT / f["file"]).read_bytes())
    print(f"{len(shards)} shards, {total / 1e6:.1f} MB (ONNX was {len(onnx_bytes) / 1e6:.1f} MB); {len(weights)} weights as Q5_0")
    return files, total, h.hexdigest()


if __name__ == "__main__":
    import argparse
    argparse.ArgumentParser(description=__doc__.split("\n")[0]).parse_args()
    files, total, sha = main()
    man = ROOT / "site/models/models.json"; m = json.loads(man.read_text())
    b = m["models"]["balanced"]
    b.update({"format": "gguf", "parts": files, "bytes": total, "sha256": sha, "label": "Balanced - INT8 pose transformer (Q5_0 storage)"})
    man.write_text(json.dumps(m, indent=1)); print("manifest updated")
