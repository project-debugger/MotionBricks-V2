"""100STYLE -> SOMA retargeting (tools/mocap/style100.py)."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/mocap"))
import style100 as S  # noqa: E402
from somart import math3d as m3  # noqa: E402

RAW = S.CACHE / "raw" / "Neutral_FW.bvh"
MESH = ROOT / "showcase/build/cache/soma_mesh.json"
needs_data = pytest.mark.skipif(not (RAW.exists() and MESH.exists()),
                                reason="run tools/mocap/style100.py fetch (and the showcase mesh export) first")


def test_quat_between():
    rng = np.random.default_rng(0)
    for _ in range(50):
        a, b = rng.normal(size=3), rng.normal(size=3)
        q = S.quat_between(a, b)
        out = m3.quat_rotate(q, a / np.linalg.norm(a))
        assert np.allclose(out, b / np.linalg.norm(b), atol=1e-9)
    q = S.quat_between(np.array([0, 1.0, 0]), np.array([0, -1.0, 0]))  # opposite vectors
    assert np.allclose(m3.quat_rotate(q, np.array([0, 1.0, 0])), [0, -1, 0], atol=1e-9)


@needs_data
def test_retarget_geometry():
    """Bones point the same way as the performer's; stance feet stay flat; the pelvis height is plausible."""
    from somart.skeleton import load_bvh
    sk, target_h = S.soma_skeleton()
    src = load_bvh(RAW, drop_static_root=None).resample(30.0)
    rt = S.Style100Retargeter(sk, src.skeleton)
    local = rt(src.local[300:900])
    w, sw = sk.fk(local), src.skeleton.fk(src.local[300:900])
    for d, (s, dc, sc) in S.MAP.items():
        if isinstance(sc, tuple) or d in S.FLAT:
            continue
        a = w[:, sk.index(dc), :3] - w[:, sk.index(d), :3]
        b = sw[:, src.skeleton.index(sc), :3] - sw[:, src.skeleton.index(s), :3]
        cos = np.sum(a * b, -1) / np.linalg.norm(a, axis=-1) / np.linalg.norm(b, axis=-1)
        assert np.degrees(np.arccos(np.clip(cos, -1, 1))).mean() < 2.0, d
    # flat-footed stance: foot pitch close to the SOMA bind pose (about 21 degrees)
    F, T = sk.index("LeftFoot"), sk.index("LeftToeBase")
    d = w[:, T, :3] - w[:, F, :3]
    pitch = np.degrees(np.arctan2(-d[:, 1], np.linalg.norm(d[:, [0, 2]], axis=1)))
    low = (w[:, T, 1] < np.percentile(w[:, T, 1], 30)) & (w[:, F, 1] < np.percentile(w[:, F, 1], 30))
    assert 15 < np.median(pitch[low]) < 28
    # after the ground shift used by retarget_all, the walking pelvis sits at a normal height (NVIDIA walks: 0.96-0.99 m)
    hips = w[:, 0, 1] + target_h - S.stance_height(sk, local)
    assert 0.93 < hips.mean() < 1.03
