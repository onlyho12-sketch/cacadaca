#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${1:-${ROOT_DIR}/results/surface_generalization_benchmark}"

python3 "${ROOT_DIR}/scripts/surface_generalization_benchmark.py" \
  --output-dir "${OUTPUT_DIR}" --seed 20260828
python3 "${ROOT_DIR}/scripts/aggregate_curved_rtx_evidence.py" \
  --results-root "${ROOT_DIR}/results" --output-dir "${OUTPUT_DIR}"

echo "[Surface Generalization] completed: ${OUTPUT_DIR}"
