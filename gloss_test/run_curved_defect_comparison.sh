#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAAC_PYTHON="${ISAAC_PYTHON:-/home/rokey/isaacsim-6.0.1/python.sh}"
BEFORE_DIR="${ROOT_DIR}/results/curved_distributed_initial"
AFTER_DIR="${ROOT_DIR}/results/curved_distributed_improved"
OUTPUT_DIR="${ROOT_DIR}/results/curved_distributed_comparison"

"${ROOT_DIR}/run_curved_rtx_scan.sh" \
  --distributed-roughness initial --tag curved_distributed_initial
"${ROOT_DIR}/run_curved_rtx_scan.sh" \
  --distributed-roughness improved --tag curved_distributed_improved
"${ISAAC_PYTHON}" "${ROOT_DIR}/scripts/compare_curved_distributed_states.py" \
  --before-dir "${BEFORE_DIR}" --after-dir "${AFTER_DIR}" --output-dir "${OUTPUT_DIR}"
"${ISAAC_PYTHON}" "${ROOT_DIR}/scripts/compute_gu_proxy.py" \
  --input-csv "${OUTPUT_DIR}/before_after_cells.csv" --output-dir "${OUTPUT_DIR}"
