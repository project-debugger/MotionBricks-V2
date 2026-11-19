import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from conftest import ROOT, needs_clips, needs_model

ORIG = ROOT / "models/soma/original"
REPORT = ROOT / "models/soma/checksums/verification_report.json"


@pytest.mark.skipif(not REPORT.exists(), reason="run tools/download_soma.py first")
def test_soma_download_verified_and_complete():
    rep = json.loads(REPORT.read_text())
    assert rep["all_ok"] and rep["subfolder"] == "soma" and not rep["unexpected_local_files"]
    assert len(rep["files"]) == 25
    assert not any("g1" in k.lower() for k in rep["files"])  # never downloaded G1
    # re-hash the small files (the large ones are re-verified by `download_soma.py --verify-only`)
    for rel, info in rep["files"].items():
        p = ORIG / rel
        if p.stat().st_size < 100_000:
            assert hashlib.sha256(p.read_bytes()).hexdigest() == info["expected"], rel


@pytest.mark.skipif(not (ROOT / "models/soma/optimized/manifest.json").exists(), reason="no optimized models")
def test_optimized_manifest_lists_existing_files():
    man = json.loads((ROOT / "models/soma/optimized/manifest.json").read_text())
    for name, v in man["variants"].items():
        p = ROOT / "models/soma/optimized/onnx" / name
        if p.exists():
            assert p.stat().st_size == v["bytes"], name


@needs_model
@needs_clips
def test_controller_runs_and_follows_input():
    from somart.backends import make_backend
    from somart.clips import ClipLibrary
    from somart.config import Config
    from somart.controller import CharacterController, Command
    from somart.engine import SomaEngine
    from somart.hardware import resolve_inference

    cfg = Config.load("balanced", overrides={"inference": {"threads": 2}})
    lib = ClipLibrary(ROOT / cfg.get("clips.root")).load("locomotion/**/*.bvh")
    lib.index_windows(["locomotion/bs_1d_linear", "locomotion/idle"])
    eng = SomaEngine(make_backend(resolve_inference(cfg.section("inference"))), lib.skeleton)
    ctrl = CharacterController(eng, lib, cfg, synchronous=True)
    start = ctrl.sim_pos.copy()
    poses = [ctrl.update(Command(move=(0.0, 1.0), gait="run"), 1 / 30) for _ in range(90)]
    world = lib.skeleton.fk(np.array(poses))
    assert np.isfinite(world).all()
    assert ctrl.stats.generations >= 3
    moved = world[-1, 0, [0, 2]] - start[[0, 2]]
    assert moved[1] > 2.0 and abs(moved[0]) < 1.0  # ran forward (+Z) several metres
    # live parameter change is picked up
    cfg.set("animation.transition_duration", 0.3)
    assert abs(ctrl.inertializer.halflife - 0.3 / 2.5) < 1e-9
    eng.close()
