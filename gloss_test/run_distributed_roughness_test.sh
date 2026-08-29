#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ISAAC_PYTHON="${ISAAC_PYTHON:-$HOME/isaacsim-6.0.1/python.sh}"

"$HERE/run_test.sh" --spatial-grid 5 --scan-roughness 0.10 \
  --distributed-roughness initial --roughness-seed 20260827 \
  --tag distributed_roughness_initial

"$HERE/run_test.sh" --spatial-grid 5 --scan-roughness 0.10 \
  --distributed-roughness improved --roughness-seed 20260827 \
  --tag distributed_roughness_improved

"$ISAAC_PYTHON" "$HERE/scripts/compare_distributed_states.py" \
  "$HERE/results/distributed_roughness_initial" \
  "$HERE/results/distributed_roughness_improved" \
  "$HERE/results/distributed_roughness_comparison"

"$HERE/run_gu_proxy.sh" "$HERE/results/distributed_roughness_comparison"
