#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEED="${1:-20260828}"
OUTPUT_CSV="${2:-${ROOT_DIR}/results/rl_vehicle_150_input/vehicle_150_cells.csv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" "${ROOT_DIR}/scripts/export_vehicle_rl_input_150.py" \
  --geometry-csv \
    "${ROOT_DIR}/results/vehicle_multi_region_local_20/vehicle_multi_region_local_20.csv" \
  --state-csv \
    "${ROOT_DIR}/results/vehicle_seed_repeatability/vehicle_seed_repeatability_cells.csv" \
  --output-csv "${OUTPUT_CSV}" \
  --seed "${SEED}"
