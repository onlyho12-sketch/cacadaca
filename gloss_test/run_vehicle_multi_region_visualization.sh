#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAAC_PYTHON="${ISAAC_PYTHON:-/home/rokey/isaacsim-6.0.1/python.sh}"
OUTPUT_DIR="${ROOT_DIR}/results/vehicle_multi_region_visualization"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/inspection_playback_$(date '+%Y%m%d_%H%M%S').txt"

echo "[Vehicle Inspection Visualization] full log: ${LOG_PATH}"
PYTHONUNBUFFERED=1 "${ISAAC_PYTHON}" \
  "${ROOT_DIR}/scripts/play_vehicle_multi_region_inspection.py" \
  --asset "${ROOT_DIR}/../scan_obj/car.usd" \
  --cells-csv \
    "${ROOT_DIR}/results/vehicle_multi_region_benchmark/vehicle_multi_region_rtx_cells.csv" \
  --output-dir "${OUTPUT_DIR}" "$@" 2>&1 | tee "${LOG_PATH}"
