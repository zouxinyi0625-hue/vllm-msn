#!/bin/bash
# Performance test for Gemma4 E011 serve config.
# Runs vllm bench serve at multiple request rates to measure throughput and latency.
#
# Usage:
#   bash test_performance.sh [HOST] [PORT] [NUM_PROMPTS]
#
# Prerequisites: server must be running (serve_e011.sh)
set -e

HOST=${1:-localhost}
PORT=${2:-8100}
NUM_PROMPTS=${3:-1000}
MODEL_NAME=${4:-gemma4}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_PATH="${DATASET_PATH:-${SCRIPT_DIR}/../datasets/sc1_delta_v2.jsonl}"

# Tokenizer path: use text_only model dir for tokenizer
if [[ -z "${_ModelDataPath_}" ]]; then
  TOKENIZER_PATH="${GEMMA4_TEXT_ONLY_MODEL_PATH:-$(pwd)/INPUT_model_dir/text_only}"
else
  TOKENIZER_PATH="${GEMMA4_TEXT_ONLY_MODEL_PATH:-${_ModelDataPath_}/text_only}"
fi

BASE_URL="http://${HOST}:${PORT}"

echo "=== Gemma4 E011 Online Serving Benchmark ==="
echo "  Target: ${BASE_URL}"
echo "  Model: ${MODEL_NAME}"
echo "  Dataset: ${DATASET_PATH}"
echo "  Prompts: ${NUM_PROMPTS}"
echo ""

# --- Wait for server ready ---
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

# --- Throughput test (request_rate=inf) ---
echo "========================================"
echo "  TEST 1: Max throughput (rate=inf)"
echo "========================================"
vllm bench serve \
  --backend openai-chat \
  --base-url "$BASE_URL" \
  --endpoint /v1/chat/completions \
  --model "$MODEL_NAME" \
  --tokenizer "$TOKENIZER_PATH" \
  --dataset-name custom \
  --dataset-path "$DATASET_PATH" \
  --num-prompts "$NUM_PROMPTS" \
  --output-len 8192 \
  --request-rate inf

echo ""

# --- Latency-sensitive tests at various QPS ---
for RATE in 10 5 2 1; do
  echo "========================================"
  echo "  TEST: Request rate = ${RATE} req/s"
  echo "========================================"
  vllm bench serve \
    --backend openai-chat \
    --base-url "$BASE_URL" \
    --endpoint /v1/chat/completions \
    --model "$MODEL_NAME" \
    --tokenizer "$TOKENIZER_PATH" \
    --dataset-name custom \
    --dataset-path "$DATASET_PATH" \
    --num-prompts "$NUM_PROMPTS" \
    --output-len 8192 \
    --request-rate "$RATE"
  echo ""
done

echo "=== All tests complete ==="
