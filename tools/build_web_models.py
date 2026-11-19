#!/usr/bin/env python3
"""Build browser-sized SOMA models for onnxruntime-web and split them into small parts.

For each source variant:
  * quantized layers (MatMulInteger / MatMulNBits) are kept as they are;
  * float32 MatMul / Conv weights are stored as int8 with a per-output-channel scale (Cast * scale,
    folded back to float32 at session creation: compute is unchanged, eval-set MPJPE 7.10 vs 7.08 cm);
  * every other float32 initializer with >= 4096 elements is stored as float16 followed by a
    Cast to float32 (ONNX Runtime constant-folds the Cast when the session is created, so the
    compute precision is unchanged; only the download shrinks). Small tensors such as the
    normalisation statistics stay float32;
  * the file is split into <= 14 MB parts (GitHub Pages / artifact friendly) with a manifest.

  .venv/bin/python tools/build_web_models.py            # int8_pose -> site/models/balanced/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "models/soma/optimized/onnx"
OUT = ROOT / "site/models"
PART = 14 * 1024 * 1024
# The INT4 variant (soma_int4_pose) is not shipped: its MatMulNBits kernels are ~3x slower than INT8
# in onnxruntime-web on one WASM thread (0.75-1.2 s vs 0.26-0.31 s per generation, Chromium, this CPU).
VARIANTS = {
    "balanced": ("soma_int8_pose.onnx", "Balanced - INT8 pose transformer"),
}


def fp16_storage(model: onnx.ModelProto, min_elems: int = 4096) -> tuple[onnx.ModelProto, int]:
    g = model.graph
    cast_nodes, converted = [], 0
    rename = {}
    keep = []
    for t in g.initializer:
        if t.data_type == TensorProto.FLOAT and int(np.prod(t.dims)) >= min_elems:
            arr = numpy_helper.to_array(t).astype(np.float16)
            new_name = t.name + "__fp16"
            keep.append(numpy_helper.from_array(arr, new_name))
            cast_nodes.append(helper.make_node("Cast", [new_name], [t.name], name=t.name + "__cast", to=TensorProto.FLOAT))
            rename[t.name] = new_name
            converted += 1
        else:
            keep.append(t)
    del g.initializer[:]
    g.initializer.extend(keep)
    nodes = list(g.node)
    del g.node[:]
    g.node.extend(cast_nodes + nodes)
    return model, converted


def int8_storage(model: onnx.ModelProto, min_elems: int = 4096) -> tuple[onnx.ModelProto, int]:
    """Store float MatMul / Conv weights as int8 with one scale per output channel.

    Each weight W becomes Cast(int8 -> float) * scale. Both are constant inputs, so ONNX Runtime folds
    them back into an ordinary float32 weight when the session is created: the network computes at
    float32 exactly as before, with weights rounded to 255 levels per output channel.
    """
    g = model.graph
    users = {}
    for n in g.node:
        for i, x in enumerate(n.input):
            users.setdefault(x, set()).add((n.op_type, i))
    nodes, keep, converted = [], [], 0
    for t in g.initializer:
        u = users.get(t.name, set())
        if t.data_type == TensorProto.FLOAT and int(np.prod(t.dims)) >= min_elems and len(t.dims) in (2, 3) and u in ({("MatMul", 1)}, {("Conv", 1)}):
            w = numpy_helper.to_array(t).astype(np.float32)
            axes = (0,) if u == {("MatMul", 1)} else tuple(range(1, w.ndim))  # per output column / output channel
            scale = np.maximum(np.abs(w).max(axis=axes, keepdims=True), 1e-12) / 127.0
            q = np.clip(np.rint(w / scale), -127, 127).astype(np.int8)
            keep += [numpy_helper.from_array(q, t.name + "__q8"), numpy_helper.from_array(scale.astype(np.float32), t.name + "__scale")]
            nodes += [helper.make_node("Cast", [t.name + "__q8"], [t.name + "__deq"], name=t.name + "__cast", to=TensorProto.FLOAT),
                      helper.make_node("Mul", [t.name + "__deq", t.name + "__scale"], [t.name], name=t.name + "__mul")]
            converted += 1
        else:
            keep.append(t)
    del g.initializer[:]
    g.initializer.extend(keep)
    rest = list(g.node)
    del g.node[:]
    g.node.extend(nodes + rest)
    return model, converted


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("variants", nargs="*", default=list(VARIANTS))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    index = json.loads((OUT / "models.json").read_text()) if (OUT / "models.json").exists() else {"models": {}}
    index["models"] = {k: v for k, v in index["models"].items() if k in VARIANTS}
    for key in args.variants:
        src_name, label = VARIANTS[key]
        model = onnx.load(str(SRC / src_name))
        model, n8 = int8_storage(model)
        model, n = fp16_storage(model)
        onnx.checker.check_model(model)
        data = model.SerializeToString()
        d = OUT / key
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("part*.bin"):
            old.unlink()
        parts = []
        for i in range(0, len(data), PART):
            chunk = data[i:i + PART]
            name = f"part{i // PART:03d}.bin"
            (d / name).write_bytes(chunk)
            parts.append({"file": name, "bytes": len(chunk), "sha256": sha256(chunk)})
        (d / "model.onnx.local").write_bytes(data)  # local copy for validation (git-ignored)
        index["models"][key] = {"label": label, "source": src_name, "bytes": len(data), "sha256": sha256(data),
                                "int8_initializers": n8, "fp16_initializers": n, "parts": parts}
        print(f"{key}: {src_name} -> {len(data) / 1e6:.1f} MB in {len(parts)} parts ({n8} int8-stored, {n} fp16-stored)")
    (OUT / "models.json").write_text(json.dumps(index, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
