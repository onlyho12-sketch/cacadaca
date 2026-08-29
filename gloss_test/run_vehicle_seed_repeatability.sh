#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${ROOT_DIR}/results/vehicle_seed_repeatability"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/repeatability_$(date '+%Y%m%d_%H%M%S').txt"

echo "[Vehicle Seed Repeatability] full log: ${LOG_PATH}"
PYTHONUNBUFFERED=1 python3 \
  "${ROOT_DIR}/scripts/vehicle_seed_repeatability.py" \
  --geometry-csv \
    "${ROOT_DIR}/results/vehicle_multi_region_benchmark/vehicle_multi_region_rtx_cells.csv" \
  --output-dir "${OUTPUT_DIR}" "$@" 2>&1 | tee "${LOG_PATH}"
