import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime"))

ONNX_DIR = ROOT / "models/soma/optimized/onnx"
FP32 = ONNX_DIR / "soma_fp32.onnx"
OFFICIAL_FP32 = ROOT / "third_party/motionbricks/motionbricks/assets/onnx/soma/mmm_v6_soma_fp32.onnx"
EVAL = ROOT / "datasets/soma_eval/eval_requests.npz"
CLIPS = ROOT / "third_party/motionbricks/assets/motions/soma"

needs_model = pytest.mark.skipif(not FP32.exists(), reason="run tools/export_onnx.sh first")
needs_eval = pytest.mark.skipif(not EVAL.exists(), reason="needs the upstream SOMA evaluation set (NVIDIA sample data, not included)")
needs_clips = pytest.mark.skipif(not CLIPS.exists(), reason="run tools/fetch_upstream.py first")


@pytest.fixture(scope="session")
def eval_data():
    from somart.skeleton import Skeleton
    d = np.load(EVAL)
    sk = Skeleton.from_dict(json.loads((ROOT / "datasets/soma_eval/skeleton.json").read_text()))
    return {k: d[k] for k in d.files}, sk


@pytest.fixture(scope="session")
def fp32_engine(eval_data):
    from somart import SomaEngine
    from somart.backends import OrtBackend
    _, sk = eval_data
    return SomaEngine(OrtBackend(FP32, intra_threads=2), sk, keep_raw=True)
