#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAAC_PYTHON="${ISAAC_PYTHON:-/home/rokey/isaacsim-6.0.1/python.sh}"
CELLS_CSV="${VEHICLE_RL_150_CELLS:-${ROOT_DIR}/results/rl_vehicle_150_states/vehicle_rl_150_render_cells.csv}"
OUTPUT_DIR="${ROOT_DIR}/results/rl_vehicle_150_visualization"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/rl_inspection_$(date '+%Y%m%d_%H%M%S').txt"

echo "[Vehicle RL 150 Visualization] full log: ${LOG_PATH}"
PYTHONUNBUFFERED=1 "${ISAAC_PYTHON}" \
  "${ROOT_DIR}/scripts/play_vehicle_rl_150_inspection.py" \
  --asset "${ROOT_DIR}/../scan_obj/car.usd" \
  --cells-csv "${CELLS_CSV}" \
  --output-dir "${OUTPUT_DIR}" "$@" 2>&1 | tee "${LOG_PATH}"
