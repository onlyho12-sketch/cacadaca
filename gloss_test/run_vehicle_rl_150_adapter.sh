#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEOMETRY_CSV="${ROOT_DIR}/results/vehicle_multi_region_local_20/vehicle_multi_region_local_20.csv"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODE="${1:-template}"

if [[ "${MODE}" == "template" ]]; then
  OUTPUT_CSV="${2:-${ROOT_DIR}/results/rl_vehicle_150_adapter/rl_vehicle_150_input_template.csv}"
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/adapt_vehicle_rl_150.py" template \
    --geometry-csv "${GEOMETRY_CSV}" \
    --output-csv "${OUTPUT_CSV}"
elif [[ "${MODE}" == "validate" ]]; then
  if [[ $# -lt 2 ]]; then
    echo "사용법: $0 validate /받은/RL.csv [출력폴더] [episode_id]" >&2
    exit 2
  fi
  INPUT_CSV="$2"
  OUTPUT_DIR="${3:-${ROOT_DIR}/results/rl_vehicle_150_adapter/validated}"
  COMMAND=(
    "${PYTHON_BIN}" "${ROOT_DIR}/scripts/adapt_vehicle_rl_150.py" validate
    --geometry-csv "${GEOMETRY_CSV}"
    --input-csv "${INPUT_CSV}"
    --output-dir "${OUTPUT_DIR}"
  )
  if [[ $# -ge 4 ]]; then
    COMMAND+=(--episode-id "$4")
  fi
  "${COMMAND[@]}"
else
  echo "알 수 없는 모드: ${MODE} (template 또는 validate)" >&2
  exit 2
fi
