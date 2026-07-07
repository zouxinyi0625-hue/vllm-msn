#!/usr/bin/env bash
# Run offline alignment benchmark from a config JSON.
set -euo pipefail
cd "$(dirname "$0")"

CONFIG=${1:-configs/26b_e011_mtp.json}
shift || true

python3 bench_offline_align.py --config "$CONFIG" "$@"
