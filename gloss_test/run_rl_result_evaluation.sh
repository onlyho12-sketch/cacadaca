#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_CSV="${1:-${ROOT_DIR}/examples/rl_polishing_output_example.csv}"
OUTPUT_DIR="${2:-${ROOT_DIR}/results/rl_bridge_smoke_test}"
ISAAC_PYTHON="${ISAAC_PYTHON:-/home/rokey/isaacsim-6.0.1/python.sh}"

"${ISAAC_PYTHON}" "${ROOT_DIR}/scripts/evaluate_rl_output.py" \
  --input-csv "${INPUT_CSV}" \
  --output-dir "${OUTPUT_DIR}"
