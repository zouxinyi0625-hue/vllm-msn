#!/usr/bin/env bash
# Smoke-test the 26B MoE-target DSpark inference path in ONE command before the
# full per-layer benchmark. Starts the server on the BF16 config, runs a tiny
# single-layer online bench, and prints whether an acceptance rate came back.
#
# WHY: the 26B DSpark draft is DENSE over a MoE target (design A). The inference
# code (gemma4.py MoE target + Gemma4DSparkForCausalLM draft + DSparkSpeculator)
# is target-agnostic in theory, but the first run is where KV-sharing wiring,
# aux-layer ids (dspark_target_layer_ids +1 -> [4,12,20,29]) and draft weight
# loading actually get exercised. Keep this cheap so failures surface fast.
#
# RUN ON THE SERVER (GPU + Azure ML mount), inside the vLLM venv, from this dir.
#
# USAGE
#   # default: BF16 config, 20 prompts, the first short-layer eval set
#   export GEMMA4_SPECULATOR_MODEL=/tmp/models/dspark/26b_finetune
#   EVAL_DIR="$AZURE_ML_INPUT_msndni/shares/users/zxy/maiprofile/prepared_prompts/20260615/short_layers/eval_datasets" \
#     bash smoke_dspark_26b.sh
#
#   # point at a specific single layer + more prompts:
#   LAYERS=layer1_actual NUM_PROMPTS=50 bash smoke_dspark_26b.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="${CONFIG:-configs/26b_dspark_bf16.json}"
LAYERS="${LAYERS:-short}"
NUM_PROMPTS="${NUM_PROMPTS:-20}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-16}"
SUMMARY_JSON="${SUMMARY_JSON:-maiprofile_dspark_26b_smoke_summary.json}"

echo "=============================================="
echo " 26B DSpark SMOKE (single-layer, ${NUM_PROMPTS} prompts)"
echo "=============================================="
echo "  config    : $CONFIG"
echo "  draft     : ${GEMMA4_SPECULATOR_MODEL:-<from config>}"
echo "  layers    : $LAYERS"
echo ""
echo "  If this prints a non-null acceptance rate, the 26B MoE-target DSpark"
echo "  inference path is WIRED CORRECTLY. Then run the full bench with"
echo "  run_maiprofile_online.sh CONFIG=configs/26b_dspark.json (FP8)."
echo ""

# Only ONE layer for the smoke: if LAYERS=short expands to many, the harness
# runs them all — so for a true smoke, pass a single LAYERS=<name>. We still
# keep NUM_PROMPTS tiny to bound the time even if multiple layers run.
CONFIG="$CONFIG" LAYERS="$LAYERS" NUM_PROMPTS="$NUM_PROMPTS" \
  MAX_CONCURRENCY="$MAX_CONCURRENCY" SUMMARY_JSON="$SUMMARY_JSON" \
  bash run_maiprofile_online.sh

echo ""
echo "Smoke done. Inspect $SUMMARY_JSON — acceptance_rate must be non-null."
