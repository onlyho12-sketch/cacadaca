#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAAC_PYTHON="${ISAAC_PYTHON:-/home/rokey/isaacsim-6.0.1/python.sh}"
MATERIAL_PROFILE="white_automotive_literature_composite_v1"

for SURFACE_PROFILE in cylinder sphere; do
  BEFORE_TAG="${SURFACE_PROFILE}_white_clearcoat_initial"
  AFTER_TAG="${SURFACE_PROFILE}_white_clearcoat_improved"
  BEFORE_DIR="${ROOT_DIR}/results/${BEFORE_TAG}"
  AFTER_DIR="${ROOT_DIR}/results/${AFTER_TAG}"
  OUTPUT_DIR="${ROOT_DIR}/results/${SURFACE_PROFILE}_white_clearcoat_comparison"

  "${ROOT_DIR}/run_curved_rtx_scan.sh" \
    --surface-profile "${SURFACE_PROFILE}" \
    --material-profile "${MATERIAL_PROFILE}" \
    --distributed-roughness initial --tag "${BEFORE_TAG}"
  "${ROOT_DIR}/run_curved_rtx_scan.sh" \
    --surface-profile "${SURFACE_PROFILE}" \
    --material-profile "${MATERIAL_PROFILE}" \
    --distributed-roughness improved --tag "${AFTER_TAG}"
  "${ISAAC_PYTHON}" "${ROOT_DIR}/scripts/compare_curved_distributed_states.py" \
    --before-dir "${BEFORE_DIR}" --after-dir "${AFTER_DIR}" --output-dir "${OUTPUT_DIR}"
  "${ISAAC_PYTHON}" "${ROOT_DIR}/scripts/compute_gu_proxy.py" \
    --input-csv "${OUTPUT_DIR}/before_after_cells.csv" \
    --output-dir "${OUTPUT_DIR}" \
    --good-anchor-gu 88.8 --high-gloss-anchor-gu 88.8
done

echo "Cylinder and sphere white-clearcoat comparisons completed."
