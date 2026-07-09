#!/usr/bin/env bash
# End-to-end: Gemma4-26B-A4B on sc1_delta_v2, four runs, all logs saved.
#   1) offline MTP        (configs/26b_e011_mtp.json)
#   2) offline no-MTP     (configs/26b_e011_no_mtp.json)
#   3) online  MTP        (configs/26b_e011_mtp.json)
#   4) online  no-MTP     (configs/26b_e011_no_mtp.json)
#
# Fire-and-forget: runs sequentially, each run's stdout+stderr is tee'd to its
# own log AND appended to a combined log. A failing run does NOT abort the rest
# (each is guarded); a STATUS line is recorded per run so you can see what
# passed/failed when you come back.
#
# TP=1 on purpose: 26B-A4B fits for plain inference on one GPU here; tp>2 was
# only needed for hidden-states extraction (a different, memory-heavier path).
#
# USAGE (run on the server, in the vLLM venv):
#   bash run_26b_maiprofile_e2e.sh
#
# Skip the (slow) model download if the model is already on disk:
#   SKIP_DOWNLOAD=1 bash run_26b_maiprofile_e2e.sh
#
# Everything below matches the manual commands; edit the vars at top to taste.
set -uo pipefail   # NOTE: no -e — we want to continue past a failing run.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ================= Configuration =================
NUM_PROMPTS="${NUM_PROMPTS:-1000}"
REPS="${REPS:-1}"
PORT="${PORT:-8100}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-32}"        # online client concurrency
TP_SIZE="${TP_SIZE:-1}"                          # 26B-A4B inference on 1 GPU
export TP_SIZE

MODEL_ROOT="${MODEL_ROOT:-/tmp/models/gemma4}"
HF_REPO="${HF_REPO:-Xinyi0625/Gemma-4-26B-A4B-it-deploy}"
# Token: prefer an already-exported HF_TOKEN; fall back to the one you provided.
HF_TOKEN="${HF_TOKEN:-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}"

# Model path env consumed by serve_align.sh AND bench_offline_align.py.
export GEMMA4_26B_TEXT_ONLY_MODEL_PATH="${MODEL_ROOT}/text_only"
export GEMMA4_26B_ASSISTANT_MODEL_PATH="${MODEL_ROOT}/assistant"
export _ModelDataPath_="${MODEL_ROOT}"

# Logs
LOG_DIR="${LOG_DIR:-26b_maiprofile_logs}"
mkdir -p "$LOG_DIR"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
COMBINED_LOG="${LOG_DIR}/ALL_${RUN_ID}.log"
STATUS_LOG="${LOG_DIR}/STATUS_${RUN_ID}.log"
# =================================================

log()  { echo "$@" | tee -a "$COMBINED_LOG"; }
sep()  { log ""; log "############################################################"; log "# $*"; log "############################################################"; }

# run_step <name> <logfile> <command...> : run guarded, tee to per-step + combined,
# record STATUS. Never aborts the script on failure.
run_step() {
  local name="$1"; shift
  local logf="$1"; shift
  sep "START: ${name}  (log: ${logf})"
  local t0 rc
  t0="$(date +%s)"
  # shellcheck disable=SC2069
  { "$@" 2>&1; echo "STEP_EXIT_CODE=$?"; } | tee "$logf" | tee -a "$COMBINED_LOG"
  # recover the real exit code from the marker (pipe hides $? of the command)
  rc="$(grep -oE 'STEP_EXIT_CODE=[0-9]+' "$logf" | tail -1 | cut -d= -f2)"
  rc="${rc:-1}"
  local dt=$(( $(date +%s) - t0 ))
  if [[ "$rc" == "0" ]]; then
    echo "PASS  ${name}  (${dt}s)  ${logf}" | tee -a "$STATUS_LOG" "$COMBINED_LOG"
  else
    echo "FAIL  ${name}  rc=${rc}  (${dt}s)  ${logf}" | tee -a "$STATUS_LOG" "$COMBINED_LOG"
  fi
}

sep "26B-A4B MAI Profile E2E — run ${RUN_ID}"
log "  num_prompts     : ${NUM_PROMPTS}"
log "  reps (offline)  : ${REPS}"
log "  online conc.    : ${MAX_CONCURRENCY}"
log "  tp size         : ${TP_SIZE}"
log "  model root      : ${MODEL_ROOT}"
log "  text_only       : ${GEMMA4_26B_TEXT_ONLY_MODEL_PATH}"
log "  assistant       : ${GEMMA4_26B_ASSISTANT_MODEL_PATH}"
log "  combined log    : ${COMBINED_LOG}"
log "  status log      : ${STATUS_LOG}"

# ---- Step 0: download model (skippable) ----
if [[ "${SKIP_DOWNLOAD:-0}" == "1" ]]; then
  sep "SKIP download (SKIP_DOWNLOAD=1)"
else
  export HF_HUB_DISABLE_XET=1
  run_step "download_model" "${LOG_DIR}/00_download_${RUN_ID}.log" \
    hf download "$HF_REPO" --local-dir "$MODEL_ROOT" --token "$HF_TOKEN" --max-workers 8
fi

# Sanity: confirm the model dirs exist before benchmarking.
if [[ ! -d "${GEMMA4_26B_TEXT_ONLY_MODEL_PATH}" ]]; then
  log "[FATAL] text_only model dir missing: ${GEMMA4_26B_TEXT_ONLY_MODEL_PATH}"
  log "        Check the download step / HF token, then re-run with SKIP_DOWNLOAD=0."
  echo "FAIL  preflight_model_dir_missing" | tee -a "$STATUS_LOG"
  exit 1
fi

# ---- Step 1: offline MTP ----
run_step "offline_mtp" "${LOG_DIR}/01_offline_mtp_${RUN_ID}.log" \
  bash run_offline_align.sh configs/26b_e011_mtp.json --reps "$REPS" --prompts "$NUM_PROMPTS"

# ---- Step 2: offline no-MTP ----
run_step "offline_no_mtp" "${LOG_DIR}/02_offline_no_mtp_${RUN_ID}.log" \
  bash run_offline_align.sh configs/26b_e011_no_mtp.json --reps "$REPS" --prompts "$NUM_PROMPTS"

# ---- Step 3: online MTP (auto server up/bench/down) ----
run_step "online_mtp" "${LOG_DIR}/03_online_mtp_${RUN_ID}.log" \
  bash run_online_align.sh configs/26b_e011_mtp.json \
    --port "$PORT" --num-prompts "$NUM_PROMPTS" --max-concurrency "$MAX_CONCURRENCY"

# ---- Step 4: online no-MTP ----
run_step "online_no_mtp" "${LOG_DIR}/04_online_no_mtp_${RUN_ID}.log" \
  bash run_online_align.sh configs/26b_e011_no_mtp.json \
    --port "$PORT" --num-prompts "$NUM_PROMPTS" --max-concurrency "$MAX_CONCURRENCY"

# ---- Summary ----
sep "DONE — status summary"
cat "$STATUS_LOG" | tee -a "$COMBINED_LOG"
log ""
log "Per-run logs:"
log "  offline MTP     : ${LOG_DIR}/01_offline_mtp_${RUN_ID}.log"
log "  offline no-MTP  : ${LOG_DIR}/02_offline_no_mtp_${RUN_ID}.log"
log "  online  MTP     : ${LOG_DIR}/03_online_mtp_${RUN_ID}.log"
log "  online  no-MTP  : ${LOG_DIR}/04_online_no_mtp_${RUN_ID}.log"
log "  offline JSON    : offline_results/*.json"
log "  online  txt     : online_results/*.txt"
log "  combined        : ${COMBINED_LOG}"
log ""
log "Paste these logs back and I'll fill in the results table."
