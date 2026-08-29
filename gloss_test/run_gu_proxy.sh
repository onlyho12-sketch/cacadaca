#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ISAAC_PYTHON="${ISAAC_PYTHON:-$HOME/isaacsim-6.0.1/python.sh}"
COMPARISON_DIR="${1:-$HERE/results/distributed_roughness_comparison}"

"$ISAAC_PYTHON" "$HERE/scripts/compute_gu_proxy.py" \
  --input-csv "$COMPARISON_DIR/before_after_cells.csv" \
  --output-dir "$COMPARISON_DIR"
