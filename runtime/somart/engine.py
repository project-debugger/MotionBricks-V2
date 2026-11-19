"""SOMA generation engine: guide poses in, 30 Hz motion out.

Pipeline per request (all NumPy except the network):
  local poses -> FK -> align -> encode -> [backend] -> decode -> endpoint correction

Supports single requests, batched requests (many characters, one network call) and
asynchronous requests (a background worker thread, so the game loop never blocks).
"""
from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from . import codec
from . import contract as C
from .backends import Backend
from .skeleton import Skeleton


@dataclass
class GenerationRequest:
    context: np.ndarray            # [4, J, 7] local poses, root in world (model) space
    target: np.ndarray             # [1..4, J, 7] local poses, root in world (model) space
    allowed_tokens: tuple[int, ...] = C.DEFAULT_TOKEN_COUNTS
    heading: tuple[float, float] = (1.0, 0.0)
    correct_endpoints: bool = True
    match_velocity: bool = False   # velocity-continuous joins (somart addition, see codec.velocity_matching)
    velocity_tau: float = 4.0      # frames


@dataclass
class GeneratedMotion:
    local: np.ndarray              # [F, J, 7]
    contacts: np.ndarray           # [F, 4] (LeftFoot, LeftToeBase, RightFoot, RightToeBase)
    tokens: int
    fps: float = C.SAMPLE_RATE
    timings_ms: dict = field(default_factory=dict)
    raw: np.ndarray | None = None  # [64, 366] raw network output (optional)

    @property
    def num_frames(self) -> int:
        return self.local.shape[0]


class SomaEngine:
    def __init__(self, backend: Backend, skeleton: Skeleton, *, keep_raw: bool = False) -> None:
        self.backend = backend
        self.binding = codec.SkeletonBinding.bind(skeleton)
        self.skeleton = skeleton
        self.keep_raw = keep_raw
        self._pool: ThreadPoolExecutor | None = None
        self.last_timings: dict = {}

    # ------------------------------------------------------------------ encoding
    def build_inputs(self, requests: list[GenerationRequest]):
        sk = self.skeleton
        guides, ntargets, allowed, aligns = [], [], [], []
        for r in requests:
            ctx = np.asarray(r.context, dtype=np.float64)
            if ctx.shape[0] != C.POSES_PER_TOKEN:
                raise ValueError("context token must contain 4 poses")
            tgt, k = codec.pad_token(r.target)
            guides.append(np.concatenate([ctx, tgt], axis=0))
            ntargets.append(k)
            allowed.append(tuple(r.allowed_tokens))
        local = np.stack(guides)                 # [N, 8, J, 7]
        world = sk.fk(local)
        heading = requests[0].heading
        if any(r.heading != heading for r in requests):
            aligns = np.concatenate([codec.guide_alignment(world[i:i + 1], self.binding, r.heading)
                                     for i, r in enumerate(requests)])
        else:
            aligns = codec.guide_alignment(world, self.binding, heading)
        feats = codec.encode(world, self.binding, aligns)
        feats.update(codec.conditions(np.array(ntargets), allowed))
        feats["_num_targets"] = np.array(ntargets)  # not a model input; returned per call (thread-safe)
        return feats, aligns, local

    # ---------------------------------------------------------------- generation
    def generate_batch(self, requests: list[GenerationRequest]) -> list[GeneratedMotion]:
        t0 = time.perf_counter()
        feats, aligns, local = self.build_inputs(requests)
        ntargets = feats.pop("_num_targets")
        t1 = time.perf_counter()
        pred, tokens = self.backend.run(feats)
        t2 = time.perf_counter()
        poses, contacts, counts = codec.decode(pred, tokens, self.binding, aligns)
        t3 = time.perf_counter()
        need = [i for i, r in enumerate(requests) if r.correct_endpoints]
        if need:
            corr = codec.endpoint_correction(poses[need], counts[need], local[need, C.POSES_PER_TOKEN - 1],
                                             local[need, C.POSES_PER_TOKEN])
            poses[need] = corr
        for i, r in enumerate(requests):
            if r.match_velocity:
                k = int(ntargets[i])
                poses[i:i + 1] = codec.velocity_matching(
                    poses[i:i + 1], counts[i:i + 1], local[i:i + 1, C.POSES_PER_TOKEN - 2],
                    local[i:i + 1, C.POSES_PER_TOKEN - 1], local[i:i + 1, C.POSES_PER_TOKEN],
                    local[i:i + 1, C.POSES_PER_TOKEN + 1] if k >= 2 else None, r.velocity_tau)
        t4 = time.perf_counter()
        timings = {"encode": (t1 - t0) * 1e3, "inference": (t2 - t1) * 1e3, "decode": (t3 - t2) * 1e3,
                   "post_process": (t4 - t3) * 1e3, "total": (t4 - t0) * 1e3, "batch": len(requests)}
        self.last_timings = timings
        out = []
        for i, r in enumerate(requests):
            tok = int(tokens[i])
            if tok not in r.allowed_tokens:
                raise RuntimeError(f"model chose {tok} tokens outside allowed {r.allowed_tokens}")
            c = int(counts[i])
            out.append(GeneratedMotion(poses[i, :c].copy(), contacts[i, :c].copy(), tok, timings_ms=timings,
                                       raw=pred[i].copy() if self.keep_raw else None))
        return out

    def generate(self, context: np.ndarray, target: np.ndarray, **kw) -> GeneratedMotion:
        return self.generate_batch([GenerationRequest(context, target, **kw)])[0]

    # ------------------------------------------------------------------- async
    def submit(self, requests: list[GenerationRequest]) -> Future:
        """Run ``generate_batch`` on a background worker; returns a Future."""
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="soma-gen")
        return self._pool.submit(self.generate_batch, requests)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
        self.backend.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
