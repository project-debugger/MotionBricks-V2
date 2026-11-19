#!/usr/bin/env python3
"""Quick start: 1) one transition between two mocap poses, 2) a short interactive-style session.

    .venv/bin/python examples/quickstart.py            # writes benchmarks/output/quickstart_*.bvh
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime"))

from somart import SomaEngine                       # noqa: E402
from somart.backends import make_backend           # noqa: E402
from somart.bvh_writer import write_bvh            # noqa: E402
from somart.clips import ClipLibrary, place        # noqa: E402
from somart.config import Config                   # noqa: E402
from somart.controller import CharacterController, Command  # noqa: E402
from somart.hardware import detect, recommend_profile, resolve_inference  # noqa: E402

hw = detect()
profile = recommend_profile(hw)
print(f"hardware: {hw.cpu} | GPU providers: {hw.gpu_providers or 'none'} | recommended profile: {profile}")
cfg = Config.load(profile)
engine_cfg = resolve_inference(cfg.section("inference"), hw)
print(f"model: {engine_cfg['model']} on {engine_cfg['providers']} x{engine_cfg['threads']} threads")

lib = ClipLibrary(ROOT / cfg.get("clips.root")).load("locomotion/**/*.bvh")
lib.index_windows(["locomotion/bs_1d_linear", "locomotion/idle"])
engine = SomaEngine(make_backend(engine_cfg), lib.skeleton)

# 1) Raw engine: idle -> running target 3 m ahead, turned 90 degrees.
idle = place(lib.window_at(lib.find(cfg.get("actions.idle.source"))[0], 30).local, np.zeros(3), 0.0)
run = lib.match(np.array([0.0, 3.4]))
target = place(run.local, np.array([3.0, 0.0, 1.0]), np.pi / 2)
g = engine.generate(idle, target, match_velocity=True)
print(f"transition: {g.num_frames} frames ({g.num_frames / 30:.2f} s), tokens {g.tokens}, "
      f"timings {{{', '.join(f'{k}: {v:.1f} ms' for k, v in g.timings_ms.items() if k != 'batch')}}}")
out = ROOT / "benchmarks/output"
write_bvh(out / "quickstart_transition.bvh", lib.skeleton, g.local)

# 2) Controller: run forward 2 s, then turn right 2 s (synchronous for a deterministic script).
cfg.set("locomotion.run_speed", 3.8)          # runtime change, validated
ctrl = CharacterController(engine, lib, cfg, synchronous=True)
poses = [ctrl.update(Command(move=(0.0, 1.0), gait="run"), 1 / 30) for _ in range(60)]
poses += [ctrl.update(Command(move=(1.0, 0.0), gait="run"), 1 / 30) for _ in range(60)]
write_bvh(out / "quickstart_session.bvh", lib.skeleton, np.array(poses))
print(f"session: {len(poses)} frames, {ctrl.stats.generations} generations, "
      f"hips moved to {np.round(poses[-1][0, [0, 2]], 2)} m -> {out}/quickstart_*.bvh")
engine.close()
