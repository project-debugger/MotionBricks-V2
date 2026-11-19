#!/usr/bin/env python3
"""Produce optimized SOMA model variants from the FP32 export. Originals are never touched.

Variants (all written to models/soma/optimized/onnx/):
  soma_fp32.onnx            FP32 export (tools/export_onnx.sh, upstream exporter)
  soma_fp16.onnx            mixed FP16 export (upstream converter: networks FP16, geometry FP32)
  soma_fp32_ortopt.onnx     FP32 graph pre-optimized offline by ONNX Runtime (faster session start)
  soma_int8.onnx            INT8 dynamic quantization of all network weight MatMul/Gemm (per-channel)
  soma_int8_pose.onnx       INT8 only on the 16-layer pose transformer; root model + decoder stay FP32
  soma_int4.onnx            INT4 weight-only (MatMulNBits, block 32, symmetric) on network MatMuls
  soma_int4_pose.onnx       INT4 only on the pose transformer
Usage:
  .venv/bin/python tools/optimize_models.py all
  .venv/bin/python tools/optimize_models.py int8 int4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import onnx

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models" / "soma" / "optimized" / "onnx"
SRC = OUT / "soma_fp32.onnx"
MANIFEST = ROOT / "models" / "soma" / "optimized" / "manifest.json"

NETWORK_PREFIXES = ("/mmm_net/", "/decoder/")
POSE_PREFIX = ("/mmm_net/_transformer_model/",)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def weight_matmuls(model: onnx.ModelProto, prefixes) -> tuple[list[str], list[str]]:
    """Return (included, excluded) MatMul/Gemm node names whose B input is a constant initializer."""
    inits = {t.name for t in model.graph.initializer}
    inc, exc = [], []
    for n in model.graph.node:
        if n.op_type not in ("MatMul", "Gemm"):
            continue
        const_b = len(n.input) > 1 and n.input[1] in inits
        (inc if const_b and n.name.startswith(prefixes) else exc).append(n.name)
    return inc, exc


def all_nodes_except(model, keep):
    keep = set(keep)
    return [n.name for n in model.graph.node if n.name not in keep]


def make_ortopt(dst: Path) -> dict:
    import onnxruntime as ort
    so = ort.SessionOptions()
    # EXTENDED keeps the file portable across CPUs (ALL may bake in layout-specific kernels).
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    so.optimized_model_filepath = str(dst)
    ort.InferenceSession(str(SRC), so, providers=["CPUExecutionProvider"])
    return {"method": "onnxruntime offline graph optimization (ORT_ENABLE_EXTENDED)",
            "onnxruntime": ort.__version__}


def make_int8(dst: Path, prefixes) -> dict:
    from onnxruntime.quantization import QuantType, quantize_dynamic
    model = onnx.load(str(SRC))
    inc, _ = weight_matmuls(model, prefixes)
    exclude = all_nodes_except(model, inc)
    del model
    quantize_dynamic(str(SRC), str(dst), op_types_to_quantize=["MatMul", "Gemm"], per_channel=True,
                     weight_type=QuantType.QInt8, nodes_to_exclude=exclude,
                     extra_options={"MatMulConstBOnly": True})
    return {"method": "onnxruntime dynamic quantization, INT8 weights per-channel, INT8 activations "
                      "quantized at runtime (VNNI/AMX-INT8 kernels on x86)",
            "quantized_nodes": len(inc), "scope": list(prefixes)}


def gemm_to_matmul(model: onnx.ModelProto, prefixes) -> int:
    """Rewrite Gemm(A, W, b) with constant W into MatMul(A, W') + Add(b) (exact, no precision change).

    MatMulNBits only quantizes MatMul, so this lets INT4 reach the Gemm projection layers too.
    """
    from onnx import helper, numpy_helper
    inits = {t.name: t for t in model.graph.initializer}
    new_nodes, count = [], 0
    for n in model.graph.node:
        if n.op_type != "Gemm" or not n.name.startswith(prefixes) or n.input[1] not in inits:
            new_nodes.append(n)
            continue
        attr = {a.name: helper.get_attribute_value(a) for a in n.attribute}
        if attr.get("transA", 0) or attr.get("alpha", 1.0) != 1.0 or attr.get("beta", 1.0) != 1.0:
            new_nodes.append(n)
            continue
        w = numpy_helper.to_array(inits[n.input[1]])
        if attr.get("transB", 0):
            w = w.T.copy()
        wname = n.input[1] + "__matmul"
        model.graph.initializer.append(numpy_helper.from_array(w, wname))
        mm_out = n.output[0] + "__mm" if len(n.input) > 2 and n.input[2] else n.output[0]
        new_nodes.append(helper.make_node("MatMul", [n.input[0], wname], [mm_out], name=n.name + "/MatMul"))
        if mm_out != n.output[0]:
            new_nodes.append(helper.make_node("Add", [mm_out, n.input[2]], [n.output[0]], name=n.name + "/Add"))
        count += 1
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    used = {i for n in model.graph.node for i in n.input}
    keep = [t for t in model.graph.initializer if t.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep)
    return count


def make_int4(dst: Path, prefixes, block_size: int = 32) -> dict:
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
    model = onnx.load(str(SRC))
    rewritten = gemm_to_matmul(model, prefixes)
    inc, _ = weight_matmuls(model, prefixes)
    inc = [n for n in inc if next(x for x in model.graph.node if x.name == n).op_type == "MatMul"]
    exclude = all_nodes_except(model, inc)
    q = MatMulNBitsQuantizer(model, block_size=block_size, is_symmetric=True, accuracy_level=4,
                             nodes_to_exclude=exclude)
    q.process()
    q.model.save_model_to_file(str(dst), use_external_data_format=False)
    return {"method": f"weight-only INT4 (MatMulNBits, block {block_size}, symmetric, accuracy_level 4 "
                      "= INT8 compute); constant Gemm rewritten to MatMul+Add first",
            "quantized_nodes": len(inc), "gemm_rewritten": rewritten, "scope": list(prefixes)}


VARIANTS = {
    "ortopt": ("soma_fp32_ortopt.onnx", lambda d: make_ortopt(d)),
    "int8": ("soma_int8.onnx", lambda d: make_int8(d, NETWORK_PREFIXES)),
    "int8_pose": ("soma_int8_pose.onnx", lambda d: make_int8(d, POSE_PREFIX)),
    "int4": ("soma_int4.onnx", lambda d: make_int4(d, NETWORK_PREFIXES)),
    "int4_pose": ("soma_int4_pose.onnx", lambda d: make_int4(d, POSE_PREFIX)),
}


def update_manifest() -> dict:
    data = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {"variants": {}}
    for f in sorted(OUT.glob("*.onnx")):
        entry = data["variants"].setdefault(f.name, {})
        entry.update({"bytes": f.stat().st_size, "sha256": sha256(f)})
    data["source_checkpoints"] = "models/soma/original (verified against upstream SHA256SUMS)"
    data["variants"].setdefault("soma_fp32.onnx", {}).setdefault(
        "method", "upstream motionbricks-onnx exporter @54fc563, FP32, opset 17")
    data["variants"].setdefault("soma_fp16.onnx", {}).setdefault(
        "method", "upstream motionbricks-onnx exporter @54fc563, mixed FP16 (LayerNorm/Softmax/geometry FP32)")
    MANIFEST.write_text(json.dumps(data, indent=2))
    (ROOT / "models" / "soma" / "checksums" / "SHA256SUMS.optimized").write_text(
        "".join(f"{v['sha256']}  optimized/onnx/{k}\n" for k, v in sorted(data["variants"].items())
                if "sha256" in v))
    return data


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("variants", nargs="+", help="all | " + " | ".join(VARIANTS))
    args = ap.parse_args()
    names = list(VARIANTS) if "all" in args.variants else args.variants
    if not SRC.exists():
        print("missing FP32 export; run tools/export_onnx.sh first", file=sys.stderr)
        return 1
    data = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {"variants": {}}
    for name in names:
        fname, fn = VARIANTS[name]
        dst = OUT / fname
        t0 = time.time()
        print(f"[optimize] {name} -> {dst.name}", flush=True)
        info = fn(dst)
        info["build_seconds"] = round(time.time() - t0, 1)
        info["source"] = "soma_fp32.onnx"
        data["variants"].setdefault(fname, {}).update(info)
        MANIFEST.write_text(json.dumps(data, indent=2))
        print(f"[optimize] {name} done in {info['build_seconds']} s, {dst.stat().st_size/1e6:.1f} MB", flush=True)
    update_manifest()
    return 0


if __name__ == "__main__":
    sys.exit(main())
