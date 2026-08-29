#!/usr/bin/env bash
set -euo pipefail

GLOSS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${GLOSS_DIR}/.." && pwd)"
ISAAC_PYTHON="${ISAAC_PYTHON:-/home/rokey/isaacsim-6.0.1/python.sh}"
WORKERS="${RL_EXPORT_WORKERS:-8}"
INITIAL_CSV="${1:-${GLOSS_DIR}/results/rl_vehicle_150_input/vehicle_150_cells.csv}"
OUTPUT_DIR="${2:-${GLOSS_DIR}/results/learning_rl_vehicle_150}"
BRIDGE="${GLOSS_DIR}/scripts/bridge_learning_vehicle_150.py"
LEARNING_EXPORTER="${REPO_DIR}/learning/vehicle_export/export_vehicle_results.py"
GEOMETRY_CSV="${GLOSS_DIR}/results/vehicle_multi_region_local_20/vehicle_multi_region_local_20.csv"
BC_CKPT="${REPO_DIR}/learning/rl/champion/model_bc.pt"
TERMINAL_CKPT="${REPO_DIR}/learning/rl/champion/model_terminal_ppo_it400.pt"

mkdir -p "${OUTPUT_DIR}"

"${ISAAC_PYTHON}" "${BRIDGE}" to-learning \
  --input-csv "${INITIAL_CSV}" \
  --output-csv "${OUTPUT_DIR}/learning_vehicle_150_input.csv"

run_policy() {
  local policy_id="$1"
  local checkpoint="$2"
  local policy_dir="${OUTPUT_DIR}/${policy_id}"
  mkdir -p "${policy_dir}"

  "${ISAAC_PYTHON}" "${LEARNING_EXPORTER}" \
    --checkpoint "${checkpoint}" \
    --input "${OUTPUT_DIR}/learning_vehicle_150_input.csv" \
    --output "${policy_dir}/learning_raw_results.csv" \
    --summary "${policy_dir}/learning_raw_summary.json" \
    --workers "${WORKERS}"

  "${ISAAC_PYTHON}" "${BRIDGE}" from-learning \
    --initial-csv "${INITIAL_CSV}" \
    --learning-csv "${policy_dir}/learning_raw_results.csv" \
    --output-csv "${policy_dir}/vehicle_150_contract.csv" \
    --policy-id "${policy_id}" \
    --checkpoint "${checkpoint}" \
    --learning-summary "${policy_dir}/learning_raw_summary.json"

  "${ISAAC_PYTHON}" "${GLOSS_DIR}/scripts/adapt_vehicle_rl_150.py" validate \
    --geometry-csv "${GEOMETRY_CSV}" \
    --input-csv "${policy_dir}/vehicle_150_contract.csv" \
    --output-dir "${policy_dir}/validated"

  "${ISAAC_PYTHON}" "${GLOSS_DIR}/scripts/prepare_vehicle_rl_150_states.py" \
    --input-csv "${policy_dir}/validated/rl_vehicle_150_cells_normalized.csv" \
    --output-dir "${policy_dir}/render_states" \
    --ra-min-um 0.0 \
    --ra-max-um 0.20 \
    --rz-max-um 2.0
}

run_policy "bc_champion" "${BC_CKPT}"
run_policy "terminal_ppo_it400" "${TERMINAL_CKPT}"

"${ISAAC_PYTHON}" "${BRIDGE}" compare \
  --bc-summary "${OUTPUT_DIR}/bc_champion/vehicle_150_contract.summary.json" \
  --terminal-summary "${OUTPUT_DIR}/terminal_ppo_it400/vehicle_150_contract.summary.json" \
  --output-json "${OUTPUT_DIR}/policy_comparison.json"

echo "[learning RL 150] complete: ${OUTPUT_DIR}"
