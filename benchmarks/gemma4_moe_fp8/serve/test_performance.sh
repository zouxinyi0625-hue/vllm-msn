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
NUM_PROMPTS=${3:-500}
MODEL_NAME=${4:-gemma4}

BASE_URL="http://${HOST}:${PORT}"

echo "=== Gemma4 E011 Online Serving Benchmark ==="
echo "  Target: ${BASE_URL}"
echo "  Model: ${MODEL_NAME}"
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
  --model "$MODEL_NAME" \
  --dataset-name random \
  --random-input-len 4000 \
  --random-output-len 1300 \
  --num-prompts "$NUM_PROMPTS" \
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
    --model "$MODEL_NAME" \
    --dataset-name random \
    --random-input-len 4000 \
    --random-output-len 1300 \
    --num-prompts "$NUM_PROMPTS" \
    --request-rate "$RATE"
  echo ""
done

echo "=== All tests complete ==="
