#!/usr/bin/env bash
# Run offline throughput benchmark for Qwen3-4B: baseline vs 4-bit
# Usage: bash run_bench.sh
#
# Requires:
#   - HF_TOKEN env var set (for private model access)
#   - sc1 dataset at datasets/sc1_delta_v2.jsonl (same as gemma4_moe_fp8)

set -euo pipefail
cd "$(dirname "$0")"

export HF_TOKEN="${HF_TOKEN:?ERROR: Set HF_TOKEN for private model access}"

SCENARIO="sc1"
MAX_NUM_SEQS="128"
REPS=1
echo "=== Qwen3-4B Baseline (bf16) ==="
python bench_offline.py \
    --scenario "$SCENARIO" \
    --model-tag baseline \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --reps "$REPS" \
    --output-dir bench_results_baseline

echo ""
echo "=== Qwen3-4B 4-bit (Xinyi0625/train_maiprofile4b_w4) ==="
python bench_offline.py \
    --scenario "$SCENARIO" \
    --model-tag 4bit \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --reps "$REPS" \
    --output-dir bench_results_4bit

echo ""
echo "=== Done. Compare bench_results_baseline/ vs bench_results_4bit/ ==="
