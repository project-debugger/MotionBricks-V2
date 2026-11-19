"""Inference backends for the batched SOMA graph.

- ``OrtBackend``  : ONNX Runtime (CPU, CUDA, TensorRT, DirectML, CoreML, OpenVINO... whatever
                    the installed onnxruntime build exposes). Used for FP32, mixed FP16, INT8, INT4.
- ``TorchBackend``: PyTorch eager graph built from the original SafeTensors checkpoints with the
                    upstream (Apache-2.0) model code. Used for BF16 autocast (AMX on Intel CPUs,
                    Tensor Cores on GPUs) and as an independent numerical reference.

Both return ``(pred_global_motions [N,64,366] float32, pred_num_tokens [N] int32)``.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

from . import contract as C

REPO_ROOT = Path(__file__).resolve().parents[2]


class Backend:
    name = "base"
    precision = "fp32"

    def run(self, inputs: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - trivial
        pass

    @property
    def device(self) -> str:
        return "cpu"


class OrtBackend(Backend):
    name = "onnxruntime"

    def __init__(self, model_path: str | Path, *, precision: str = "fp32", intra_threads: int = 0,
                 inter_threads: int = 1, graph_optimization: str = "all", providers: list | None = None,
                 enable_mem_pattern: bool = True, enable_cpu_mem_arena: bool = True,
                 optimized_model_path: str | Path | None = None, execution_mode: str = "sequential",
                 allow_spinning: bool = True, use_io_binding: bool = True) -> None:
        import onnxruntime as ort

        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")
        self.precision = precision
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(intra_threads)
        so.inter_op_num_threads = int(inter_threads)
        so.graph_optimization_level = {
            "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
            "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        }[graph_optimization]
        so.execution_mode = (ort.ExecutionMode.ORT_PARALLEL if execution_mode == "parallel"
                             else ort.ExecutionMode.ORT_SEQUENTIAL)
        so.enable_mem_pattern = enable_mem_pattern
        so.enable_cpu_mem_arena = enable_cpu_mem_arena
        if not allow_spinning:
            so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        if optimized_model_path:
            so.optimized_model_filepath = str(optimized_model_path)
        available = ort.get_available_providers()
        self.providers = [p for p in (providers or ["CPUExecutionProvider"]) if
                          (p[0] if isinstance(p, tuple) else p) in available] or ["CPUExecutionProvider"]
        t0 = time.perf_counter()
        self.session = ort.InferenceSession(str(self.model_path), so, providers=self.providers)
        self.load_seconds = time.perf_counter() - t0
        self._out_names = [o.name for o in self.session.get_outputs()]
        self._use_io_binding = use_io_binding
        self._lock = threading.Lock()

    @property
    def device(self) -> str:
        return self.session.get_providers()[0]

    def run(self, inputs):
        feeds = {k: np.ascontiguousarray(inputs[k]) for k in C.INPUT_NAMES}
        with self._lock:  # one session may be shared by a worker thread and the main thread
            out = self.session.run(self._out_names, feeds)
        res = dict(zip(self._out_names, out))
        return res["pred_global_motions"], res["pred_num_tokens"].astype(np.int32)


class TorchBackend(Backend):
    """Eager PyTorch graph from the original checkpoints (requires third_party upstream code)."""

    name = "torch"

    def __init__(self, checkpoint_root: str | Path, *, precision: str = "fp32", threads: int = 0,
                 device: str = "cpu", compile: bool = False) -> None:
        import torch

        pkgs = REPO_ROOT / "third_party" / "pkgs"
        if not (pkgs / "motionbricks_onnx").exists():
            raise RuntimeError("Upstream model code missing: run `python3 tools/fetch_upstream.py` "
                               "and `bash tools/export_onnx.sh` (creates third_party/pkgs).")
        if str(pkgs) not in sys.path:
            sys.path.insert(0, str(pkgs))
        from motionbricks_onnx.checkpoints import checkpoint_root as _root, load_sources
        from motionbricks_onnx.checksums import CHECKPOINTS
        from motionbricks_onnx.models.soma import build_soma

        if threads:
            torch.set_num_threads(int(threads))
        import logging
        # Upstream warns on every call because the last context slot carries no root velocity;
        # that is by design of the conditioning masks (see codec.conditions), not an input error.
        logging.getLogger("motionbricks_onnx.models.soma.inference").setLevel(logging.ERROR)
        t0 = time.perf_counter()
        sources = load_sources(_root(Path(checkpoint_root), CHECKPOINTS["soma"]), CHECKPOINTS["soma"])
        self.model = build_soma(sources).eval().to(device)
        del sources
        self.load_seconds = time.perf_counter() - t0
        self.precision = precision
        self._device = device
        self._torch = torch
        self._dtype = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}[precision]
        if self._dtype is not None:
            # The fused nn.TransformerEncoderLayer fast path bypasses autocast and fails on mixed
            # dtypes; the regular path runs Linear/MatMul in BF16 (AMX/AVX512-BF16 on Intel CPUs).
            torch.backends.mha.set_fastpath_enabled(False)
        if compile:
            self.model = torch.compile(self.model)

    @property
    def device(self) -> str:
        return self._device

    def run(self, inputs):
        torch = self._torch
        args = [torch.from_numpy(np.ascontiguousarray(inputs[k])).to(self._device) for k in C.INPUT_NAMES]
        with torch.inference_mode():
            if self._dtype is None:
                pred, tok = self.model(*args)
            else:
                with torch.autocast(device_type=self._device.split(":")[0], dtype=self._dtype):
                    pred, tok = self.model(*args)
        return pred.float().cpu().numpy(), tok.cpu().numpy().astype(np.int32).reshape(-1)


def available_providers() -> list[str]:
    try:
        import onnxruntime as ort
        return list(ort.get_available_providers())
    except Exception:  # pragma: no cover
        return []


def make_backend(cfg: dict) -> Backend:
    """Create a backend from an ``inference`` config section (see configs/presets/*.yaml)."""
    kind = cfg.get("backend", "onnxruntime")
    root = REPO_ROOT
    threads = int(cfg.get("threads", 0)) or 0
    if kind == "onnxruntime":
        path = Path(cfg["model"])
        path = path if path.is_absolute() else root / path
        return OrtBackend(path, precision=cfg.get("precision", "fp32"), intra_threads=threads,
                          inter_threads=int(cfg.get("inter_threads", 1)),
                          graph_optimization=cfg.get("graph_optimization", "all"),
                          providers=cfg.get("providers"),
                          enable_mem_pattern=bool(cfg.get("memory_pattern", True)),
                          enable_cpu_mem_arena=bool(cfg.get("memory_arena", True)),
                          allow_spinning=bool(cfg.get("allow_spinning", True)))
    if kind == "torch":
        ck = Path(cfg.get("checkpoints", "models/soma/original"))
        return TorchBackend(ck if ck.is_absolute() else root / ck, precision=cfg.get("precision", "fp32"),
                            threads=threads, device=cfg.get("device", "cpu"))
    raise ValueError(f"unknown backend {kind!r}")


def cpu_count() -> int:
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
