#!/bin/bash
# Serve Gemma4 MoE with E011 optimal config (best from ablation: 2020 tok/s offline)
# Config: FP8 + CUDA graphs + MTP k=5 + text-only + gpu_mem=0.95
set -xe

# --- 1. Positional Arguments ---
PORT=${1:-8100}
TP_SIZE=${2:-1}
MAX_LEN=${3:-24576}
GPU_UTIL=${4:-0.95}
SERVED_NAME=${5:-"gemma4"}
MAX_NUM_SEQS=${6:-128}
MAX_BATCHED_TOKENS=${7:-16384}

# --- 1.1 Optional tuning args (env-overridable) ---
# Mirrors new benchmark config knobs used in align scripts.
QUANTIZATION=${QUANTIZATION:-fp8}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-auto}
MOE_BACKEND=${MOE_BACKEND:-auto}
SPECULATIVE_METHOD=${SPECULATIVE_METHOD:-mtp}
SPEC_TOKENS=${SPEC_TOKENS:-5}
DRAFT_MODEL_SUBDIR=${DRAFT_MODEL_SUBDIR:-assistant}
ENFORCE_EAGER=${ENFORCE_EAGER:-false}
ENABLE_CHUNKED_PREFILL=${ENABLE_CHUNKED_PREFILL:-true}
ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING:-true}
OPTIMIZATION_LEVEL=${OPTIMIZATION_LEVEL:-3}
PERFORMANCE_MODE=${PERFORMANCE_MODE:-throughput}

# --- 2. Model Path Resolution ---
# Expects 3 folders under _ModelDataPath_:
#   model/       - full model (unused for E011)
#   text_only/   - text-only model (main serving model)
#   assistant/   - MTP draft model (speculative decoding)
if [[ -z "${_ModelDataPath_}" ]]; then
  echo "Assuming local environment"
  base_dir="$(pwd)/INPUT_model_dir"
else
  base_dir="${_ModelDataPath_}"
fi

model="${GEMMA4_TEXT_ONLY_MODEL_PATH:-${base_dir}/text_only}"
spec_model="${GEMMA4_ASSISTANT_MODEL_PATH:-${base_dir}/${DRAFT_MODEL_SUBDIR}}"

# Fallback to HuggingFace if local paths don't exist
[[ ! -d "$model" ]] && model="google/gemma-4-26B-A4B-it"
[[ ! -d "$spec_model" ]] && spec_model=""

# --- 3. Environment ---
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}

# A100 (sm_80): FlashInfer FP8 MoE requires Hopper; use Marlin fallback
GPU_ARCH=$(python3 -c "import torch; print(torch.cuda.get_device_capability(0)[0])" 2>/dev/null || echo "8")
if [[ "$GPU_ARCH" -lt 9 ]]; then
  export VLLM_USE_FLASHINFER_MOE_FP8=${VLLM_USE_FLASHINFER_MOE_FP8:-0}
  export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
fi

# --- 4. Build vllm serve command ---
SPEC_ARGS=()
if [[ "${SPECULATIVE_METHOD}" == "mtp" ]]; then
  if [[ -n "$spec_model" && -d "$spec_model" ]]; then
    SPEC_ARGS+=(--spec-model "$spec_model" --spec-tokens "$SPEC_TOKENS")
  fi
elif [[ "${SPECULATIVE_METHOD}" == "dspark" || "${SPECULATIVE_METHOD}" == "eagle3" ]]; then
  SPECULATOR_MODEL="${GEMMA4_SPECULATOR_MODEL:-$spec_model}"
  if [[ -n "$SPECULATOR_MODEL" ]]; then
    SPEC_CONFIG_JSON=$(python3 - "$SPECULATOR_MODEL" "$SPEC_TOKENS" "$SPECULATIVE_METHOD" <<'PY'
import json, sys
print(json.dumps({
    "model": sys.argv[1],
    "num_speculative_tokens": int(sys.argv[2]),
    "method": sys.argv[3],
}))
PY
)
    SPEC_ARGS+=(--speculative-config "$SPEC_CONFIG_JSON")
  fi
fi

EXTRA_ARGS=()
if [[ -n "$QUANTIZATION" && "$QUANTIZATION" != "None" ]]; then
  EXTRA_ARGS+=(--quantization "$QUANTIZATION")
fi
if [[ -n "$KV_CACHE_DTYPE" ]]; then
  EXTRA_ARGS+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
fi
if [[ -n "$MOE_BACKEND" ]]; then
  EXTRA_ARGS+=(--moe-backend "$MOE_BACKEND")
fi
if [[ -n "$OPTIMIZATION_LEVEL" ]]; then
  EXTRA_ARGS+=(--optimization-level "$OPTIMIZATION_LEVEL")
fi
if [[ -n "$PERFORMANCE_MODE" ]]; then
  EXTRA_ARGS+=(--performance-mode "$PERFORMANCE_MODE")
fi
if [[ "$ENFORCE_EAGER" == "true" || "$ENFORCE_EAGER" == "True" ]]; then
  EXTRA_ARGS+=(--enforce-eager)
fi
if [[ "$ENABLE_CHUNKED_PREFILL" == "true" || "$ENABLE_CHUNKED_PREFILL" == "True" ]]; then
  EXTRA_ARGS+=(--enable-chunked-prefill)
fi
if [[ "$ENABLE_PREFIX_CACHING" == "false" || "$ENABLE_PREFIX_CACHING" == "False" ]]; then
  EXTRA_ARGS+=(--no-enable-prefix-caching)
fi

echo "Starting vLLM server (E011 optimal config)..."
echo "  Model: $model"
echo "  Spec model: ${spec_model:-DISABLED}"
echo "  Port: $PORT, TP: $TP_SIZE, MaxLen: $MAX_LEN, GPU_UTIL: $GPU_UTIL"
echo "  MaxNumSeqs: $MAX_NUM_SEQS, MaxBatchedTokens: $MAX_BATCHED_TOKENS"
echo "  SpecMethod: $SPECULATIVE_METHOD, SpecTokens: $SPEC_TOKENS"
echo "  Quant: $QUANTIZATION, KV: $KV_CACHE_DTYPE, MoE: $MOE_BACKEND"
echo "  OptLevel: $OPTIMIZATION_LEVEL, PerfMode: $PERFORMANCE_MODE"

vllm serve "$model" \
  --served-model-name "$SERVED_NAME" \
  --port "$PORT" \
  --tensor-parallel-size "$TP_SIZE" \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --dtype auto \
  --trust-remote-code \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
  --async-scheduling \
  --no-enable-log-requests \
  "${EXTRA_ARGS[@]}" \
  "${SPEC_ARGS[@]}"
