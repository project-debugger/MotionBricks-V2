import numpy as np
import pytest

from conftest import OFFICIAL_FP32, needs_eval, needs_model
from somart import GenerationRequest, codec
from somart import contract as C
from somart import math3d as m3


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_condition_masks_match_upstream_semantics(k):
    c = codec.conditions(np.array([k]), C.DEFAULT_TOKEN_COUNTS)
    given = (np.arange(8) < 4 + k).astype(int)
    assert (c["has_global_root_values"][0] == given).all()
    assert (c["has_local_poses"][0] == given).all()
    lr = given.copy()
    lr[3] = 0
    lr[4 + k - 1] = 0
    assert (c["has_local_root_values"][0] == lr).all()
    assert c["allowed_pred_num_tokens"][0].tolist() == [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]


def test_allowed_token_mask_validation():
    with pytest.raises(ValueError):
        C.allowed_token_mask([5])
    with pytest.raises(ValueError):
        C.allowed_token_mask([])


@needs_eval
def test_encode_alignment_faces_plus_x(eval_data):
    d, sk = eval_data
    b = codec.SkeletonBinding.bind(sk)
    local = np.concatenate([d["context"][:5], d["target"][:5]], 1).astype(np.float64)
    world = sk.fk(local)
    al = codec.guide_alignment(world, b)
    f = codec.encode(world, b, al)
    g = f["global_root_values"]
    # last context pose: ground position at origin, heading = +X  (cos=0, sin=1 in (x, z) heading convention)
    assert np.allclose(g[:, 3, [0, 2]], 0, atol=1e-5)
    assert np.allclose(g[:, 3, 3], 0, atol=1e-4) and np.allclose(g[:, 3, 4], 1, atol=1e-4)
    assert f["local_poses"].shape == (5, 8, 267)


@needs_model
@needs_eval
def test_endpoint_correction_is_exact(eval_data, fp32_engine):
    d, _ = eval_data
    reqs = [GenerationRequest(d["context"][i].astype(float), d["target"][i].astype(float)) for i in range(3)]
    for r, g in zip(reqs, fp32_engine.generate_batch(reqs)):
        assert np.abs(g.local[0, :, :3] - r.context[-1, :, :3]).max() < 1e-6
        assert np.abs(g.local[-1, :, :3] - r.target[0, :, :3]).max() < 1e-6
        assert m3.quat_angle_between(g.local[0, :, 3:], r.context[-1, :, 3:]).max() < 1e-5
        assert m3.quat_angle_between(g.local[-1, :, 3:], r.target[0, :, 3:]).max() < 1e-5
        assert C.MIN_TOKENS <= g.tokens <= C.MAX_TOKENS and g.num_frames == C.valid_frame_count(g.tokens)


@needs_model
@needs_eval
def test_pinned_token_count_is_respected(eval_data, fp32_engine):
    d, _ = eval_data
    g = fp32_engine.generate(d["context"][0].astype(float), d["target"][0].astype(float), allowed_tokens=(9,))
    assert g.tokens == 9 and g.num_frames == 30


@needs_model
@needs_eval
def test_batched_equals_single(eval_data, fp32_engine):
    d, _ = eval_data
    reqs = [GenerationRequest(d["context"][i].astype(float), d["target"][i].astype(float)) for i in range(4)]
    batch = fp32_engine.generate_batch(reqs)
    for r, b in zip(reqs, batch):
        s = fp32_engine.generate_batch([r])[0]
        assert s.tokens == b.tokens
        assert np.abs(s.local[..., :3] - b.local[..., :3]).max() < 1e-3


@needs_model
@needs_eval
def test_matches_official_upstream_onnx(eval_data, fp32_engine):
    if not OFFICIAL_FP32.exists():
        pytest.skip("official ONNX not fetched (tools/fetch_upstream.py --groups onnx)")
    from somart import SomaEngine
    from somart.backends import OrtBackend
    d, sk = eval_data
    ref = SomaEngine(OrtBackend(OFFICIAL_FP32, intra_threads=2), sk, keep_raw=True)
    for i in range(3):
        a = fp32_engine.generate(d["context"][i].astype(float), d["target"][i].astype(float))
        b = ref.generate(d["context"][i].astype(float), d["target"][i].astype(float))
        assert a.tokens == b.tokens
        assert np.abs(a.raw - b.raw).max() < 1e-3


@needs_model
@needs_eval
def test_async_submit(eval_data, fp32_engine):
    d, _ = eval_data
    fut = fp32_engine.submit([GenerationRequest(d["context"][0].astype(float), d["target"][0].astype(float))])
    res = fut.result(timeout=60)
    assert res[0].num_frames > 0 and np.isfinite(res[0].local).all()
