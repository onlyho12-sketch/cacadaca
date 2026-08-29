#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEED="${1:-20260828}"
BASE_DIR="${ROOT_DIR}/results/vehicle_seed_repeatability"
PLAN="${BASE_DIR}/representative_rtx_plan.csv"
STATE_ROOT="${BASE_DIR}/states"

if [ ! -f "${PLAN}" ]; then
  echo "대표 RTX 계획이 없습니다. 먼저 ./run_vehicle_seed_repeatability.sh 를 실행하세요." >&2
  exit 1
fi

for REGION in \
  hood roof negative_x_door positive_x_door \
  negative_x_front_fender positive_x_front_fender; do
  CELLS="$(python3 "${ROOT_DIR}/scripts/aggregate_vehicle_seed_representative_rtx.py" \
    cells --plan "${PLAN}" --seed "${SEED}" --region "${REGION}")"
  STATE="${STATE_ROOT}/seed_${SEED}/${REGION}_state_maps.npz"
  if [ ! -f "${STATE}" ]; then
    echo "상태 파일이 없습니다: ${STATE}" >&2
    exit 1
  fi
  echo "[Representative RTX] seed=${SEED}, region=${REGION}, cells=${CELLS}"
  for PHASE in before after; do
    "${ROOT_DIR}/run_vehicle_mesh_rtx_scan.sh" \
      --region-profile "${REGION}" \
      --vehicle-state-npz "${STATE}" \
      --vehicle-state-phase "${PHASE}" \
      --roughness-seed "${SEED}" \
      --measurement-cells "${CELLS}" \
      --tag "vehicle_seed_${SEED}_${REGION}_${PHASE}_representative"
  done
done

python3 "${ROOT_DIR}/scripts/aggregate_vehicle_seed_representative_rtx.py" \
  aggregate \
  --results-root "${ROOT_DIR}/results" \
  --state-root "${STATE_ROOT}" \
  --plan "${PLAN}" \
  --seed "${SEED}" \
  --output-dir "${BASE_DIR}/representative_rtx_seed_${SEED}"
