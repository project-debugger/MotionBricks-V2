"""CMU ASF/AMC -> SOMA retargeting (tools/mocap/cmu.py)."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/mocap"))
import cmu  # noqa: E402
import style100  # noqa: E402

RAW = cmu.CACHE / "raw"
needs_data = pytest.mark.skipif(not (RAW / "09_01.amc").exists(), reason="run tools/mocap/cmu.py fetch first")


def test_euler_order():
    """CMU Euler angles: about fixed x, then y, then z."""
    from somart import math3d as m3
    q = cmu._euler(np.array([90.0, 0, 90.0]))
    v = m3.quat_rotate(q, np.array([0, 1.0, 0]))  # x: y -> z, then z: z stays
    assert np.allclose(v, [0, 0, 1], atol=1e-9)


@needs_data
def test_asf_rest_pose():
    """Rest pose = identity rotations; leg bones point down, legs about 0.9 m long."""
    sk, _ = cmu.load_asf(RAW / "09.asf")
    w = sk.fk(sk.reference)
    hip, knee, ankle = (w[sk.index(n), :3] for n in ("lfemur", "ltibia", "lfoot"))
    assert (knee - hip)[1] < 0 and (ankle - knee)[1] < 0
    assert 0.7 < np.linalg.norm(knee - hip) + np.linalg.norm(ankle - knee) < 1.1


@needs_data
def test_retarget_geometry():
    """Bones point the same way as the performer's; the running pelvis stays at a normal height."""
    sk, _ = style100.soma_skeleton()
    src = cmu.load_trial("09_01").resample(30.0)
    local = cmu.CMURetargeter(sk, src.skeleton)(src.local)
    w, sw = sk.fk(local), src.skeleton.fk(src.local)
    for d, (s, dc, sc) in cmu.MAP.items():
        # Hips: CMU's root -> upperback vector also bends with the lower back, so it is no single-bone direction
        if isinstance(sc, tuple) or d in cmu.CMURetargeter.FLAT or d == "Hips":
            continue
        a = w[:, sk.index(dc), :3] - w[:, sk.index(d), :3]
        b = sw[:, src.skeleton.index(sc), :3] - sw[:, src.skeleton.index(s), :3]
        cos = np.sum(a * b, -1) / np.linalg.norm(a, axis=-1) / np.linalg.norm(b, axis=-1)
        assert np.degrees(np.arccos(np.clip(cos, -1, 1))).mean() < 3.0, d
    hips = w[:, 0, 1] + style100.STANCE_HEIGHT - style100.stance_height(sk, local)
    assert 0.9 < hips.mean() < 1.05
