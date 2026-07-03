#!/bin/bash
# bench_online_sglang.sh — Start SGLang server, run benchmark, stop server.
#
# Usage:
#   ./sglang/bench_online_sglang.sh                    # defaults
#   ./sglang/bench_online_sglang.sh --num-prompts 100  # quick test
#   ./sglang/bench_online_sglang.sh --no-spec          # disable speculative
#
# Environment variables:
#   GEMMA4_MODEL_PATH              full model path
#   GEMMA4_ASSISTANT_MODEL_PATH    EAGLE3 draft model for speculative decoding
set -euo pipefail
cd "$(dirname "$0")/.."

# --- Defaults ---
PORT=8200
MEM_FRAC=0.85
MAX_RUNNING=32
MAX_LEN=24576
NUM_PROMPTS=1000
MAX_TOKENS=8192
MAX_CONCURRENCY=32
HEALTH_TIMEOUT=300
NO_SPEC=false

# --- Parse args ---
EXTRA_BENCH_ARGS=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --mem-frac) MEM_FRAC="$2"; shift 2 ;;
    --max-running) MAX_RUNNING="$2"; shift 2 ;;
    --num-prompts) NUM_PROMPTS="$2"; shift 2 ;;
    --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
    --max-concurrency) MAX_CONCURRENCY="$2"; shift 2 ;;
    --no-spec) NO_SPEC=true; shift ;;
    *) EXTRA_BENCH_ARGS="$EXTRA_BENCH_ARGS $1"; shift ;;
  esac
done

# --- Model paths ---
: "${GEMMA4_MODEL_PATH:=/tmp/models/gemma-4-26B-A4B-it}"
: "${GEMMA4_ASSISTANT_MODEL_PATH:=${GEMMA4_MODEL_PATH}-assistant}"
export GEMMA4_MODEL_PATH GEMMA4_ASSISTANT_MODEL_PATH

if [[ "$NO_SPEC" == "true" ]]; then
  unset GEMMA4_ASSISTANT_MODEL_PATH
  export GEMMA4_ASSISTANT_MODEL_PATH=""
fi

SERVER_PID=""
cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Stopping SGLang server (PID=$SERVER_PID)..."
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

# --- 1. Start server ---
echo "=================================================="
echo "  SGLang Online Benchmark (A100 80GB)"
echo "  Port=$PORT  MemFrac=$MEM_FRAC  MaxRunning=$MAX_RUNNING"
echo "  NumPrompts=$NUM_PROMPTS  MaxTokens=$MAX_TOKENS  Concurrency=$MAX_CONCURRENCY"
echo "  Spec=$(if [[ "$NO_SPEC" == "true" ]]; then echo DISABLED; else echo ENABLED; fi)"
echo "  $(date)"
echo "=================================================="

LOG_FILE="/tmp/sglang_bench_server_${PORT}.log"
bash sglang/serve_sglang.sh "$PORT" "$MEM_FRAC" "$MAX_RUNNING" "$MAX_LEN" \
  > "$LOG_FILE" 2>&1 &
SERVER_PID=$!
echo "Server PID=$SERVER_PID, log=$LOG_FILE"

# --- 2. Wait for health ---
echo "Waiting for server to be ready (timeout=${HEALTH_TIMEOUT}s)..."
SECONDS=0
while [[ $SECONDS -lt $HEALTH_TIMEOUT ]]; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "ERROR: Server process died. Last 20 lines of log:"
    tail -20 "$LOG_FILE"
    exit 1
  fi
  if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
    echo "Server ready after ${SECONDS}s"
    break
  fi
  sleep 2
done

if [[ $SECONDS -ge $HEALTH_TIMEOUT ]]; then
  echo "ERROR: Server not ready after ${HEALTH_TIMEOUT}s. Last 20 lines of log:"
  tail -20 "$LOG_FILE"
  exit 1
fi

# --- 3. Run benchmark ---
echo ""
echo "Running benchmark..."
python3 serve/test_performance.py \
  --host localhost \
  --port "$PORT" \
  --model default \
  --num-prompts "$NUM_PROMPTS" \
  --max-tokens "$MAX_TOKENS" \
  --max-concurrency "$MAX_CONCURRENCY" \
  --request-rate inf \
  --no-stream \
  $EXTRA_BENCH_ARGS

echo ""
echo "Server log tail:"
tail -5 "$LOG_FILE"
echo ""
echo "Done. Full server log at: $LOG_FILE"
