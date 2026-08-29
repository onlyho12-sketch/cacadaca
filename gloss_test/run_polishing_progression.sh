#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

run_stage() {
  local tag="$1"
  local defect_roughness="$2"
  local scratch_strength="$3"
  shift 3
  "$HERE/run_test.sh" \
    --spatial-grid 5 --scan-roughness 0.10 \
    --defect-cell 4,4 --defect-roughness "$defect_roughness" \
    --scratch-strength "$scratch_strength" --defect-size-m 0.030 \
    --tag "$tag" "$@"
}

run_stage polish_stage00_initial 0.30 2.80
run_stage polish_stage01_pass1 0.23 1.80
run_stage polish_stage02_pass2 0.16 0.90
run_stage polish_stage03_pass3 0.12 0.35
run_stage polish_stage04_target 0.10 0.00 --no-expect-defect

python3 "$HERE/scripts/plot_polishing_progression.py" \
  "$HERE/results/polishing_progression" \
  "Initial=$HERE/results/polish_stage00_initial" \
  "Pass 1=$HERE/results/polish_stage01_pass1" \
  "Pass 2=$HERE/results/polish_stage02_pass2" \
  "Pass 3=$HERE/results/polish_stage03_pass3" \
  "Target=$HERE/results/polish_stage04_target"

echo "[Polishing] verified progression complete"
