#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAAC_PYTHON="${ISAAC_PYTHON:-/home/rokey/isaacsim-6.0.1/python.sh}"
TAG="curved_freeform_rtx_5x5"
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
LOG_PATH="${LOG_DIR}/curved_rtx_$(date '+%Y%m%d_%H%M%S').txt"

echo "[Curved RTX] full log: ${LOG_PATH}"
set +e
PYTHONUNBUFFERED=1 "${ISAAC_PYTHON}" \
  "${ROOT_DIR}/scripts/run_curved_rtx_scan.py" "$@" 2>&1 | tee "${LOG_PATH}"
RUN_STATUS=${PIPESTATUS[0]}
set -e
STATUS_JSON="${ROOT_DIR}/results/${TAG}/run_status.json"
if [ ! -f "${STATUS_JSON}" ] || ! grep -q '"success": true' "${STATUS_JSON}"; then
  RUN_STATUS=1
fi
echo "[Curved RTX] process exit code: ${RUN_STATUS}" | tee -a "${LOG_PATH}"
exit "${RUN_STATUS}"
