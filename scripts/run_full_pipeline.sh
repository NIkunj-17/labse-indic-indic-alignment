#!/usr/bin/env bash
# Runs Task 1 end to end: Phase 1 (1024 ex/pair) -> Phase 2 (weighted) -> Evaluation.
# Intended to be launched inside a tmux session so it survives SSH disconnects.
#
# Usage:
#   bash scripts/run_full_pipeline.sh
#
# Safe to re-run: each stage checks for its own completed output and skips
# if already done (see training.py's final_model_dir check).
set -euo pipefail

cd "$(dirname "$0")/.."   # repo root
OUTPUT_ROOT="${OUTPUT_ROOT:-./labse_research_output}"

echo "=== [1/3] Phase 1: balanced 1024-example fine-tuning ==="
python3 scripts/run_phase1.py \
    --config configs/phase1_1024.json \
    --output-root "${OUTPUT_ROOT}"

echo "=== [2/3] Phase 2: weak-pair weighted fine-tuning ==="
python3 scripts/run_phase2.py \
    --phase1-model-dir "${OUTPUT_ROOT}/phase1_balanced_1024/best_model" \
    --output-root "${OUTPUT_ROOT}" \
    --examples-per-pair 1024 \
    --batch-size 32 \
    --learning-rate 2e-6 \
    --max-pair-weight 2.0 \
    --run-name phase2_weighted_1024

echo "=== [3/3] Evaluation on IN22-Conv ==="
python3 scripts/run_evaluate.py \
    --output-root "${OUTPUT_ROOT}" \
    --phase1-model-dir "${OUTPUT_ROOT}/phase1_balanced_1024/best_model" \
    --phase2-model-dir "${OUTPUT_ROOT}/phase2_weighted_1024/best_model" \
    --phase1-name phase1_balanced_1024 \
    --phase2-name phase2_weighted_1024

echo "=== Pipeline complete. Results in ${OUTPUT_ROOT}/evaluation_results ==="
