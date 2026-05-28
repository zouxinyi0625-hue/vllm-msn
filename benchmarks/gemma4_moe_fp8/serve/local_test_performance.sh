#!/bin/bash
# Local performance test: request-rate=inf with varying max-concurrency.
# Results are saved to individual files in OUTPUT_DIR.
#
# Usage:
#   export _ModelDataPath_=/path/to/models
#   bash local_test_performance.sh [--base-url http://host:port] [--num-prompts N] [--model NAME]
set -e

BASE_URL=""
NUM_PROMPTS=1000
MODEL_NAME=gemma4

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-url) BASE_URL="$2"; shift 2 ;;
    --num-prompts) NUM_PROMPTS="$2"; shift 2 ;;
    --model) MODEL_NAME="$2"; shift 2 ;;
    *) shift ;;
  esac
done

if [[ -z "$BASE_URL" ]]; then
  BASE_URL="http://localhost:8100"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_PATH="${DATASET_PATH:-${SCRIPT_DIR}/../datasets/sc1_delta_v2.jsonl}"

if [[ -z "${_ModelDataPath_}" ]]; then
  TOKENIZER_PATH="${GEMMA4_TEXT_ONLY_MODEL_PATH:-$(pwd)/INPUT_model_dir/text_only}"
else
  TOKENIZER_PATH="${GEMMA4_TEXT_ONLY_MODEL_PATH:-${_ModelDataPath_}/text_only}"
fi

OUTPUT_DIR="${SCRIPT_DIR}/bench_results"
mkdir -p "$OUTPUT_DIR"

echo "=== Local Concurrency Benchmark ==="
echo "  Target: ${BASE_URL}"
echo "  Model: ${MODEL_NAME}"
echo "  Dataset: ${DATASET_PATH}"
echo "  Tokenizer: ${TOKENIZER_PATH}"
echo "  Prompts: ${NUM_PROMPTS}"
echo "  Output dir: ${OUTPUT_DIR}"
echo ""

# Wait for server ready
echo "Waiting for server to be ready..."
MAX_WAIT=600
WAITED=0
until curl -s "${BASE_URL}/health" > /dev/null 2>&1; do
  sleep 5
  WAITED=$((WAITED + 5))
  if [[ $WAITED -ge $MAX_WAIT ]]; then
    echo "ERROR: Server not ready after ${MAX_WAIT}s"
    exit 1
  fi
done
echo "Server ready (waited ${WAITED}s)."
echo ""

COMMON_ARGS=(
  --backend openai-chat
  --base-url "$BASE_URL"
  --endpoint /v1/chat/completions
  --model "$MODEL_NAME"
  --tokenizer "$TOKENIZER_PATH"
  --dataset-name custom
  --dataset-path "$DATASET_PATH"
  --num-prompts "$NUM_PROMPTS"
  --output-len 8192
  --request-rate inf
)

# Test 1: No max-concurrency (unlimited)
echo "========================================"
echo "  TEST: rate=inf, max-concurrency=unlimited"
echo "========================================"
vllm bench serve "${COMMON_ARGS[@]}" \
  2>&1 | tee "$OUTPUT_DIR/result_unlimited.txt"
echo ""

# Tests with max-concurrency 16, 64, 128
for CONC in 16 64 128; do
  echo "========================================"
  echo "  TEST: rate=inf, max-concurrency=${CONC}"
  echo "========================================"
  vllm bench serve "${COMMON_ARGS[@]}" \
    --max-concurrency "$CONC" \
    2>&1 | tee "$OUTPUT_DIR/result_concurrency_${CONC}.txt"
  echo ""
done

echo "=== All tests complete ==="
echo "Results saved in: ${OUTPUT_DIR}/"
