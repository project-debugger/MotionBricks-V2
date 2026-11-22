"""Left/right mirroring used to augment the web playground's target library."""
import json
from pathlib import Path

import numpy as np
import pytest

from somart.mirror import Mirror
from somart.skeleton import load_bvh

ROOT = Path(__file__).resolve().parents[1]
CLIP = ROOT / "third_party/motionbricks/assets/motions/soma/locomotion/bs_1d_linear/run_loop_180_R_003__A326.bvh"
MESH = ROOT / "site/data/soma_mesh.json"


@pytest.fixture(scope="module")
def setup():
    if not CLIP.exists() or not MESH.exists():
        pytest.skip("needs the upstream sample clips and the extracted SOMA mesh")
    clip = load_bvh(CLIP).resample(30)
    meta = json.loads(MESH.read_text())
    raw = (MESH.parent / "soma_mesh.bin").read_bytes()
    ib = meta["inverseBind"]
    inv = np.frombuffer(raw, np.float32, count=ib["length"], offset=ib["offset"]).reshape(-1, 16)
    sk = clip.skeleton
    loc = clip.local.astype(np.float64).copy()
    loc[:, 1:, :3] = sk.reference[None, 1:, :3]
    return sk, Mirror(sk, Mirror.bind_from_inverse_bind(inv)), loc


def test_mirrored_motion_is_the_reflection(setup):
    sk, mirror, loc = setup
    m = mirror(loc)
    m[:, 1:, :3] = sk.reference[None, 1:, :3]
    expect = sk.fk(loc)[:, mirror.swap, :3] * np.array([-1.0, 1.0, 1.0])
    assert np.abs(sk.fk(m)[..., :3] - expect).max() < 0.01  # metres


def test_mirror_twice_is_identity(setup):
    _, mirror, loc = setup
    back = mirror(mirror(loc))
    dots = np.abs((back[..., 3:] * loc[..., 3:]).sum(-1))
    assert np.degrees(2 * np.arccos(np.clip(dots, 0, 1))).max() < 1e-3
    assert np.abs(back[:, 0, :3] - loc[:, 0, :3]).max() < 1e-9
