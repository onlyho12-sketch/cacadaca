#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_CSV="${1:-${ROOT_DIR}/results/rl_vehicle_150_adapter/validated/rl_vehicle_150_cells_normalized.csv}"
OUTPUT_DIR="${2:-${ROOT_DIR}/results/rl_vehicle_150_states}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" "${ROOT_DIR}/scripts/prepare_vehicle_rl_150_states.py" \
  --input-csv "${INPUT_CSV}" \
  --output-dir "${OUTPUT_DIR}"
