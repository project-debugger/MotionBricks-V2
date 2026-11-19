#!/usr/bin/env bash
# Render all showcase videos with 3 parallel headless-Chromium processes (SwiftShader uses ~1.5 cores each).
set -euo pipefail
cd "$(dirname "$0")/../.."
export SHOWCASE_QUALITY="${SHOWCASE_QUALITY:-shadow=1024&ground=lambert}"
export NODE_PATH="$(npm root -g)"
mkdir -p showcase/build/cache/logs
node showcase/build/render.mjs cmp_model_variants_quality 07_animation_speed 01_walk_run_sprint 02_turns_and_reversals > showcase/build/cache/logs/render_a.log 2>&1 &
node showcase/build/render.mjs cmp_model_variants_aggressive cmp_fp32_vs_int8 03_crouch 04_jump 05_sit_and_stand > showcase/build/cache/logs/render_b.log 2>&1 &
node showcase/build/render.mjs cmp_responsiveness cmp_smoothness cmp_foot_locking 06_transitions_medley > showcase/build/cache/logs/render_c.log 2>&1 &
wait
cat showcase/build/cache/logs/render_*.log
