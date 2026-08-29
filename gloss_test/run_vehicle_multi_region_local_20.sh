#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAAC_PYTHON="${ISAAC_PYTHON:-/home/rokey/isaacsim-6.0.1/python.sh}"
OUTPUT_DIR="${ROOT_DIR}/results/vehicle_multi_region_local_20"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/vehicle_multi_region_$(date '+%Y%m%d_%H%M%S').txt"

echo "[Vehicle Multi-Region Local 20] full log: ${LOG_PATH}"
PYTHONUNBUFFERED=1 "${ISAAC_PYTHON}" \
  "${ROOT_DIR}/scripts/validate_vehicle_multi_region_local_20.py" \
  --asset "${ROOT_DIR}/../scan_obj/car.usd" \
  --output-dir "${OUTPUT_DIR}" "$@" 2>&1 | tee "${LOG_PATH}"

SUMMARY_PATH="${OUTPUT_DIR}/vehicle_multi_region_local_20_summary.json"
python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["passed"] else 1)' \
  "${SUMMARY_PATH}"
