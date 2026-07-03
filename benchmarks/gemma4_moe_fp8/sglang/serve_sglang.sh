#!/bin/bash
# Serve Gemma4 with SGLang — matches S003 config (BF16 + CUDA graphs + NEXTN spec k=5)
set -xe

# --- 1. Positional Arguments ---
PORT=${1:-8200}
MEM_FRAC=${2:-0.85}
MAX_RUNNING=${3:-32}
MAX_LEN=${4:-24576}

# --- 2. Model Path Resolution ---
if [[ -z "${_ModelDataPath_}" ]]; then
  base_dir="${GEMMA4_MODEL_PATH:-/tmp/models/gemma-4-26B-A4B-it}"
else
  base_dir="${_ModelDataPath_}"
fi

model="${GEMMA4_MODEL_PATH:-${base_dir}}"
spec_model="${GEMMA4_ASSISTANT_MODEL_PATH:-${model}-assistant}"

[[ ! -d "$model" ]] && model="google/gemma-4-26B-A4B-it"
[[ ! -d "$spec_model" ]] && spec_model=""

# --- 3. Build sglang serve command ---
SPEC_ARGS=""
if [[ -n "$spec_model" && -d "$spec_model" ]]; then
  SPEC_ARGS="--speculative-algorithm NEXTN \
    --speculative-draft-model-path $spec_model \
    --speculative-num-steps 5 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 6"
fi

echo "Starting SGLang server..."
echo "  Model: $model"
echo "  Spec model: ${spec_model:-DISABLED}"
echo "  Port: $PORT, MemFrac: $MEM_FRAC, MaxRunning: $MAX_RUNNING, MaxLen: $MAX_LEN"

sglang serve \
  --model-path "$model" \
  --port "$PORT" \
  --host 0.0.0.0 \
  --dtype bfloat16 \
  --context-length "$MAX_LEN" \
  --mem-fraction-static "$MEM_FRAC" \
  --chunked-prefill-size 16384 \
  --max-running-requests "$MAX_RUNNING" \
  --disable-radix-cache \
  --trust-remote-code \
  --log-level info \
  $SPEC_ARGS
