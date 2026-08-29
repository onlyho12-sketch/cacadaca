#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${1:-${ROOT_DIR}/results/curved_local_20_validation}"
ISAAC_PYTHON="${ISAAC_PYTHON:-/home/rokey/isaacsim-6.0.1/python.sh}"

"${ISAAC_PYTHON}" "${ROOT_DIR}/scripts/validate_curved_local_20.py" \
  --output-dir "${OUTPUT_DIR}"
