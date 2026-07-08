#!/usr/bin/env bash
# Download the official RedHatAI EAGLE-3 speculator (draft) for the 26B-A4B
# target, so benchmarks load it from local disk instead of hitting the HF Hub
# at serve time (faster, no rate limits, reproducible).
#
# Usage:
#   bash download_eagle3_speculator.sh [/tmp/models/gemma4/eagle3_speculator]
#
# Optional:
#   export HF_TOKEN=...   # if the environment needs authenticated HF access
set -euo pipefail

DEST=${1:-/tmp/models/gemma4/eagle3_speculator}
SPECULATOR_REPO=${EAGLE3_SPECULATOR_REPO:-RedHatAI/gemma-4-26B-A4B-it-speculator.eagle3}

mkdir -p "$DEST"

echo "=== Download Gemma4 26B-A4B EAGLE-3 speculator ==="
echo "Speculator repo: $SPECULATOR_REPO"
echo "Local dir      : $DEST"
echo ""

if ! command -v hf >/dev/null 2>&1; then
  echo "ERROR: 'hf' command not found. Install Hugging Face Hub CLI first." >&2
  echo "Example: python3 -m pip install huggingface_hub[cli] --break-system-packages" >&2
  exit 1
fi

hf download "$SPECULATOR_REPO" \
  --local-dir "$DEST"

echo ""
echo "=== Done ==="
echo "EAGLE-3 speculator: $DEST"
echo ""
echo "Point the benchmark at the local copy (overrides the config default):"
echo "export GEMMA4_SPECULATOR_MODEL=$DEST"
echo ""
echo "Then run, e.g.:"
echo "  export GEMMA4_MODEL_PATH=/tmp/models/gemma4/text_only   # 26B target"
echo "  bash run_online_align.sh configs/26b_eagle3.json --max-concurrency none --num-prompts 1000"
