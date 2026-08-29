#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAAC_PYTHON="${ISAAC_PYTHON:-/home/rokey/isaacsim-6.0.1/python.sh}"
BEFORE_DIR="${ROOT_DIR}/results/vehicle_hood_defect_initial"
AFTER_DIR="${ROOT_DIR}/results/vehicle_hood_defect_improved"
OUTPUT_DIR="${ROOT_DIR}/results/vehicle_hood_defect_comparison"

"${ROOT_DIR}/run_vehicle_mesh_rtx_scan.sh" \
  --distributed-roughness initial --tag vehicle_hood_defect_initial
"${ROOT_DIR}/run_vehicle_mesh_rtx_scan.sh" \
  --distributed-roughness improved --tag vehicle_hood_defect_improved
"${ISAAC_PYTHON}" "${ROOT_DIR}/scripts/compare_vehicle_mesh_states.py" \
  --before-dir "${BEFORE_DIR}" --after-dir "${AFTER_DIR}" --output-dir "${OUTPUT_DIR}"
"${ISAAC_PYTHON}" "${ROOT_DIR}/scripts/compute_gu_proxy.py" \
  --input-csv "${OUTPUT_DIR}/before_after_cells.csv" \
  --output-dir "${OUTPUT_DIR}" \
  --good-anchor-gu 88.8 --high-gloss-anchor-gu 88.8

echo "Vehicle hood defect comparison completed."
