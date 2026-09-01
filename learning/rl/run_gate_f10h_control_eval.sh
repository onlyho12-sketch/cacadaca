#!/usr/bin/env bash
set -euo pipefail

repo=/home/rokey/cacadaca
isaac=/home/rokey/isaacsim-6.0.1/python.sh
parent="$repo/learning/rl/robot/releases/gate_e7_seed1_it211_20260831_201801/model_gate_e7_seed1_it211.pt"
root="$repo/learning/rl/robot/results/gate_f10h_fixed_eval_20260901_231000"

for kind in cylinder freeform; do
  for direction in same_xx cross_xy; do
    "$isaac" "$repo/learning/rl/gate_f10h_residual_eval.py" \
      --parent-checkpoint "$parent" --arm control_parent \
      --surface-kind "$kind" --direction-mode "$direction" \
      --envs-per-profile 1 --surface-seed-base 60000 \
      --physics-seed 20260921 --policy-seed 20260922 \
      --out-dir "$root/control_parent_${kind}_${direction}" --headless
  done
done
