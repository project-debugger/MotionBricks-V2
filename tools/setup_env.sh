#!/usr/bin/env bash
# Create .venv (Python 3.12, the version the upstream exporter pins) with the tested packages.
# Uses uv when available (fast, fetches Python 3.12 if missing); falls back to python3.12 -m venv.
set -euo pipefail
cd "$(dirname "$0")/.."
if command -v uv >/dev/null 2>&1; then
  uv venv --python 3.12 .venv
  PIP="uv pip install --python .venv/bin/python"
else
  python3.12 -m venv .venv
  PIP=".venv/bin/python -m pip install"
fi
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cpu}
$PIP --index-url "$TORCH_INDEX" "torch==2.14.0+cpu"
$PIP -r requirements.txt
.venv/bin/python -c "import torch, onnx, onnxruntime, numpy; print('torch', torch.__version__, '| onnx', onnx.__version__, '| ort', onnxruntime.__version__, onnxruntime.get_available_providers())"
