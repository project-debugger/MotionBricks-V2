#!/usr/bin/env bash
# Export SOMA checkpoints -> ONNX with the *unmodified* upstream exporter (pinned commit).
# Prerequisites: python3 tools/fetch_upstream.py && bash tools/setup_env.sh
# Usage: bash tools/export_onnx.sh            (fp32 and fp16)
#        PRECISIONS=fp32 bash tools/export_onnx.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p third_party/pkgs
ln -sfn ../motionbricks/tools/onnx third_party/pkgs/motionbricks_onnx
export PYTHONPATH="$PWD/third_party/pkgs${PYTHONPATH:+:$PYTHONPATH}"
OUT=models/soma/optimized/onnx
mkdir -p "$OUT" benchmarks/logs
for p in ${PRECISIONS:-fp32 fp16}; do
  echo "[export] precision=$p"
  start=$(date +%s)
  $PY -m motionbricks_onnx.export --model soma --checkpoints models/soma/original \
      --precision "$p" --output "$OUT/soma_${p}.onnx" --force --verbose
  echo "[export] $p done in $(( $(date +%s) - start )) s"
done
