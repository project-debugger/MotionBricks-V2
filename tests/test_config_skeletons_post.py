import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from somart import math3d as m3
from somart.config import Config, ConfigError
from somart.postprocess import FootLocker, Inertializer, two_bone_ik
from somart.skeleton import Skeleton
from somart.skeletons import FeatureLayout, Retargeter, SkeletonProfile, model_for

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- config
@pytest.mark.parametrize("preset", Config.presets())
def test_presets_load_and_validate(preset):
    cfg = Config.load(preset)
    assert cfg.get("profile") == preset
    assert Path(ROOT / cfg.get("inference.model")).suffix == ".onnx"


def test_required_presets_exist():
    assert {"ultra_quality", "balanced", "low_vram", "extreme_performance", "cpu_only"} <= set(Config.presets())


def test_validation_rejects_typos_types_ranges():
    cfg = Config.load("balanced")
    with pytest.raises(ConfigError):
        cfg.set("locomotion.run_sped", 3.0)
    with pytest.raises(ConfigError):
        cfg.set("locomotion.run_speed", "fast")
    with pytest.raises(ConfigError):
        cfg.set("animation.smoothing", 5.0)
    with pytest.raises(ConfigError):
        cfg.set("inference.precision", "int2")


def test_runtime_change_notifies_listeners():
    cfg = Config.load("balanced")
    seen = []
    cfg.on_change(lambda k, v: seen.append((k, v)))
    cfg.set("locomotion.run_speed", 4.25)
    assert cfg.get("locomotion.run_speed") == 4.25 and seen == [("locomotion.run_speed", 4.25)]


def test_character_and_action_overrides():
    cfg = Config.load("custom/example_heavy_character.yaml")
    heavy = cfg.for_character("heavy_guard")
    assert heavy.get("locomotion.inertia") == 1.6 and cfg.get("locomotion.inertia") == 1.0
    assert heavy.get("inference.model").endswith("soma_int8_pose.onnx")
    sprint = cfg.for_action("sprint")
    assert sprint.get("generation.prediction_horizon") == 0.6


def test_save_load_roundtrip(tmp_path):
    cfg = Config.load("low_vram")
    cfg.set("locomotion.turn_rate", 400.0)
    p = cfg.save(tmp_path / "mine.yaml")
    again = Config.load(p)
    assert again.get("locomotion.turn_rate") == 400.0
    assert again.get("inference.model") == cfg.get("inference.model")
    assert "walk_speed" not in yaml.safe_load(p.read_text()).get("locomotion", {})  # only diffs saved


# ------------------------------------------------------------- skeletons
def test_soma_layout_matches_released_tensors():
    prof = SkeletonProfile.load("soma")
    assert prof.validate() == []
    lay = FeatureLayout.for_profile(prof)
    assert (lay.local_pose_dim, lay.global_motion_dim) == (267, 366)
    assert model_for(prof).status == "available"


@pytest.mark.parametrize("animal", ["dog", "cat", "horse", "wolf", "bear", "bird"])
def test_animal_templates_valid_but_not_claimed(animal):
    prof = SkeletonProfile.load(animal)
    assert prof.validate() == []
    with pytest.raises(NotImplementedError):
        model_for(prof)


def test_retargeter_identity():
    prof = SkeletonProfile.load("soma")
    sk = prof.to_skeleton()
    rng = np.random.default_rng(1)
    local = np.broadcast_to(sk.reference, (5,) + sk.reference.shape).copy()
    local[:, :, 3:] = m3.normalize(rng.normal(size=(5, sk.num_joints, 4)))
    local[:, 0, :3] = rng.normal(size=(5, 3))
    out = Retargeter(sk, sk, {n: n for n in sk.names})(local)
    assert np.allclose(m3.quat_angle_between(sk.fk(out)[..., 3:], sk.fk(local)[..., 3:]), 0, atol=1e-5)


# ----------------------------------------------------------- postprocess
def _leg():
    names = ["Hips", "LeftLeg", "LeftShin", "LeftFoot"]
    ref = np.concatenate([np.array([[0, 1.0, 0], [0.1, -0.05, 0], [0, -0.45, 0.02], [0, -0.45, -0.02]]),
                          m3.quat_identity((4,))], -1)
    return Skeleton(names, np.array([-1, 0, 1, 2]), ref)


def test_two_bone_ik_reaches_target():
    sk = _leg()
    pose = sk.reference.copy()
    world = sk.fk(pose)
    target = world[3, :3] + np.array([0.05, 0.12, 0.08])
    out = two_bone_ik(pose, world, sk, 1, 2, 3, target)
    assert np.linalg.norm(sk.fk(out)[3, :3] - target) < 1e-4
    # bone lengths preserved
    assert np.allclose(out[:, :3], pose[:, :3])


def test_inertializer_decays_to_zero():
    sk = _leg()
    a = sk.reference.copy()
    b = a.copy()
    b[1, 3:] = m3.quat_from_axis_angle(np.array([1.0, 0, 0]), 0.6)
    ine = Inertializer(4, halflife=0.05)
    ine.transition(a, a, b, b, 1 / 60)
    first = ine.apply(b, 1 / 60)
    assert m3.quat_angle_between(first[1, 3:], a[1, 3:]) < 0.6  # starts near the old pose
    for _ in range(120):
        out = ine.apply(b, 1 / 60)
    assert m3.quat_angle_between(out[1, 3:], b[1, 3:]) < 1e-3 and not ine.active


def test_foot_locker_removes_sliding():
    sk = _leg()
    fl = FootLocker(sk, [["LeftLeg", "LeftShin", "LeftFoot"]], [0], max_correction=0.2, blend_in=1e-3)
    contacts = np.array([1.0, 0, 0, 0])
    positions = []
    for i in range(20):
        pose = sk.reference.copy()
        pose[0, 0] = 0.004 * i  # hips (and foot) drift 4 mm/frame while "planted"
        out = fl.apply(pose, contacts, 1 / 30)
        positions.append(sk.fk(out)[3, :3])
    drift = np.linalg.norm(np.diff(np.array(positions)[2:, [0, 2]], axis=0), axis=-1).mean()
    assert drift < 0.0005
