#!/usr/bin/env python3
"""Build browser-sized SOMA models for onnxruntime-web and split them into small parts.

For each source variant:
  * quantized layers (MatMulInteger / MatMulNBits) are kept as they are;
  * every other float32 initializer with >= 4096 elements is stored as float16 followed by a
    Cast to float32 (ONNX Runtime constant-folds the Cast when the session is created, so the
    compute precision is unchanged; only the download shrinks). Small tensors such as the
    normalisation statistics stay float32;
  * the file is split into <= 14 MB parts (GitHub Pages / artifact friendly) with a manifest.

  .venv/bin/python tools/build_web_models.py            # int8_pose + int4_pose -> site/models/
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
VARIANTS = {
    "balanced": ("soma_int8_pose.onnx", "Balanced - INT8 pose transformer"),
    "light": ("soma_int4_pose.onnx", "Light - INT4 pose transformer"),
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


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("variants", nargs="*", default=list(VARIANTS))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    index = json.loads((OUT / "models.json").read_text()) if (OUT / "models.json").exists() else {"models": {}}
    for key in args.variants:
        src_name, label = VARIANTS[key]
        model = onnx.load(str(SRC / src_name))
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
                                "fp16_initializers": n, "parts": parts}
        print(f"{key}: {src_name} -> {len(data) / 1e6:.1f} MB in {len(parts)} parts ({n} initializers stored as fp16)")
    (OUT / "models.json").write_text(json.dumps(index, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
