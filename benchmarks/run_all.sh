#!/usr/bin/env bash
# Reproduce every report in benchmarks/ (about 35 min on a 4-vCPU machine).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
$PY datasets/build_eval_set.py
$PY benchmarks/ablation_postprocess.py --model soma_int8_pose
$PY benchmarks/make_postprocess_report.py
$PY benchmarks/run_controller_matrix.py --repeats 3
$PY benchmarks/run_benchmarks.py
