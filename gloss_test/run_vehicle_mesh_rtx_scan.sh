#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAAC_PYTHON="${ISAAC_PYTHON:-/home/rokey/isaacsim-6.0.1/python.sh}"
ASSET_DEFAULT="${ROOT_DIR}/../scan_obj/car.usd"
TAG="vehicle_mesh_hood_rtx_5x5"
PREVIOUS=""
for ARG in "$@"; do
  if [ "${PREVIOUS}" = "--tag" ]; then
    TAG="${ARG}"
    break
  fi
  PREVIOUS="${ARG}"
done
TAG="${TAG//[^a-zA-Z0-9_.-]/_}"
LOG_DIR="${ROOT_DIR}/results/${TAG}/logs"
mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/vehicle_mesh_rtx_$(date '+%Y%m%d_%H%M%S').txt"

HAS_ASSET=0
for ARG in "$@"; do
  if [ "${ARG}" = "--asset" ]; then
    HAS_ASSET=1
    break
  fi
done

echo "[Vehicle Mesh RTX] full log: ${LOG_PATH}"
set +e
if [ "${HAS_ASSET}" -eq 1 ]; then
  PYTHONUNBUFFERED=1 "${ISAAC_PYTHON}" \
    "${ROOT_DIR}/scripts/run_vehicle_mesh_rtx_scan.py" "$@" 2>&1 | tee "${LOG_PATH}"
else
  PYTHONUNBUFFERED=1 "${ISAAC_PYTHON}" \
    "${ROOT_DIR}/scripts/run_vehicle_mesh_rtx_scan.py" \
    --asset "${ASSET_DEFAULT}" "$@" 2>&1 | tee "${LOG_PATH}"
fi
RUN_STATUS=${PIPESTATUS[0]}
set -e
STATUS_JSON="${ROOT_DIR}/results/${TAG}/run_status.json"
if [ ! -f "${STATUS_JSON}" ] || ! grep -q '"success": true' "${STATUS_JSON}"; then
  RUN_STATUS=1
fi
echo "[Vehicle Mesh RTX] process exit code: ${RUN_STATUS}" | tee -a "${LOG_PATH}"
exit "${RUN_STATUS}"
