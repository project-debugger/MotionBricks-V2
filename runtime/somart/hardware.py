"""Hardware detection and automatic profile / model selection.

Never assumes an NVIDIA GPU: it inspects the onnxruntime execution providers actually
installed (CUDA, TensorRT, DirectML, CoreML, ROCm, OpenVINO ...), optional torch CUDA,
and CPU SIMD features (AVX-512 VNNI / AMX make INT8 the clear CPU winner).
"""
from __future__ import annotations

import os
import platform
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .backends import REPO_ROOT, available_providers

GPU_PROVIDERS = ("TensorrtExecutionProvider", "CUDAExecutionProvider", "ROCMExecutionProvider",
                 "MIGraphXExecutionProvider", "DmlExecutionProvider", "CoreMLExecutionProvider")


@dataclass
class HardwareInfo:
    system: str
    machine: str
    cpu: str
    logical_cpus: int
    physical_cpus: int
    ram_gb: float
    simd: list[str] = field(default_factory=list)
    ort_providers: list[str] = field(default_factory=list)
    gpu_providers: list[str] = field(default_factory=list)
    torch_cuda: bool = False

    @property
    def has_gpu(self) -> bool:
        return bool(self.gpu_providers) or self.torch_cuda

    @property
    def fast_int8(self) -> bool:
        return any(f in self.simd for f in ("avx512_vnni", "avx_vnni", "amx_int8")) or self.machine in (
            "arm64", "aarch64")

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(has_gpu=self.has_gpu, fast_int8=self.fast_int8)
        return d


def detect() -> HardwareInfo:
    cpu, simd, phys = platform.processor() or platform.machine(), [], 0
    try:
        txt = Path("/proc/cpuinfo").read_text()
        for line in txt.splitlines():
            if line.startswith("model name") and cpu in ("", platform.machine(), "x86_64"):
                cpu = line.split(":", 1)[1].strip()
            if line.startswith("flags") and not simd:
                simd = sorted(f for f in line.split(":", 1)[1].split()
                              if f.startswith(("avx512", "amx", "avx2", "avx_vnni", "f16c", "fma")))
        cores = {(l.split(":")[1].strip()) for l in txt.splitlines() if l.startswith("core id")}
        phys = len(cores) or 0
    except OSError:
        pass
    ram = 0.0
    try:
        ram = int(Path("/proc/meminfo").read_text().split()[1]) / 2**20
    except (OSError, IndexError, ValueError):
        try:
            ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
        except (ValueError, OSError, AttributeError):
            pass
    logical = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    provs = available_providers()
    torch_cuda = False
    try:  # optional
        import torch
        torch_cuda = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        pass
    return HardwareInfo(platform.system(), platform.machine(), cpu, logical, phys or logical, round(ram, 1), simd,
                        provs, [p for p in provs if p in GPU_PROVIDERS], torch_cuda)


def recommend_profile(hw: HardwareInfo | None = None) -> str:
    hw = hw or detect()
    if hw.has_gpu:
        return "ultra_quality" if hw.ram_gb >= 8 else "balanced"
    if hw.ram_gb and hw.ram_gb < 3:
        return "low_vram"
    return "balanced" if hw.fast_int8 else "cpu_only"


def resolve_inference(section: dict, hw: HardwareInfo | None = None) -> dict:
    """Pick the device-appropriate model and thread count for an ``inference`` config section."""
    hw = hw or detect()
    s = dict(section)
    by_dev = s.get("models_by_device") or {}
    uses_gpu = hw.has_gpu and any(p in GPU_PROVIDERS for p in (s.get("providers") or []))
    if by_dev:
        s["model"] = by_dev.get("gpu" if uses_gpu else "cpu", s["model"])
    if not uses_gpu:
        s["providers"] = ["CPUExecutionProvider"]
    if not int(s.get("threads", 0)):
        # leave cores for the game/render threads when generating asynchronously
        reserve = int(s.get("reserve_cores", 1)) if s.get("async", True) else 0
        s["threads"] = max(1, hw.physical_cpus - reserve)
    limit = float(s.get("memory_limit_mb", 0) or 0)
    if limit and s.get("backend", "onnxruntime") == "onnxruntime":
        model = Path(s["model"])
        model = model if model.is_absolute() else REPO_ROOT / model
        need = model.stat().st_size / 2**20 * 1.6  # measured RSS / file size ratio (see benchmarks)
        if need > limit:
            raise MemoryError(f"{model.name} needs ~{need:.0f} MB resident, above memory_limit_mb={limit:.0f}; "
                              "use the low_vram preset (INT4) or raise the limit")
    return s
