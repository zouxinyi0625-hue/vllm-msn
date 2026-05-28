#!/bin/bash
# Performance test for Gemma4 E011 serve config.
# Runs vllm bench serve at multiple request rates to measure throughput and latency.
#
# Usage:
#   bash test_performance.sh --base-url http://host:port [--num-prompts N] [--model NAME]
#   bash test_performance.sh [HOST] [PORT] [NUM_PROMPTS] [MODEL_NAME]
#
# Prerequisites: server must be running (serve_e011.sh)
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
    *)
      # Legacy positional: HOST PORT NUM_PROMPTS MODEL_NAME
      if [[ -z "$_POS1" ]]; then _POS1="$1"
      elif [[ -z "$_POS2" ]]; then _POS2="$1"
      elif [[ -z "$_POS3" ]]; then NUM_PROMPTS="$1"
      elif [[ -z "$_POS4" ]]; then MODEL_NAME="$1"
      fi
      shift ;;
  esac
done

# Build BASE_URL from positional args if --base-url not provided
if [[ -z "$BASE_URL" ]]; then
  HOST=${_POS1:-localhost}
  PORT=${_POS2:-8100}
  BASE_URL="http://${HOST}:${PORT}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_PATH="${DATASET_PATH:-${SCRIPT_DIR}/../datasets/sc1_delta_v2.jsonl}"

# Tokenizer path: use text_only model dir for tokenizer
if [[ -z "${_ModelDataPath_}" ]]; then
  TOKENIZER_PATH="${GEMMA4_TEXT_ONLY_MODEL_PATH:-$(pwd)/INPUT_model_dir/text_only}"
else
  TOKENIZER_PATH="${GEMMA4_TEXT_ONLY_MODEL_PATH:-${_ModelDataPath_}/text_only}"
fi

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
# for RATE in 10 5 2 1; do
#   echo "========================================"
#   echo "  TEST: Request rate = ${RATE} req/s"
#   echo "========================================"
#   vllm bench serve \
#     --backend openai-chat \
#     --base-url "$BASE_URL" \
#     --endpoint /v1/chat/completions \
#     --model "$MODEL_NAME" \
#     --tokenizer "$TOKENIZER_PATH" \
#     --dataset-name custom \
#     --dataset-path "$DATASET_PATH" \
#     --num-prompts "$NUM_PROMPTS" \
#     --output-len 8192 \
#     --request-rate "$RATE"
#   echo ""
# done

echo "=== All tests complete ==="
