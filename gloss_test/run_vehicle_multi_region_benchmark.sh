#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${ROOT_DIR}/run_vehicle_multi_region_local_20.sh"
for REGION in \
  hood roof negative_x_door positive_x_door \
  negative_x_front_fender positive_x_front_fender; do
  "${ROOT_DIR}/run_vehicle_mesh_rtx_scan.sh" \
    --region-profile "${REGION}" \
    --tag "vehicle_multi_region_${REGION}_rtx"
done

python3 "${ROOT_DIR}/scripts/aggregate_vehicle_multi_region_results.py" \
  --results-root "${ROOT_DIR}/results" \
  --output-dir "${ROOT_DIR}/results/vehicle_multi_region_benchmark"
