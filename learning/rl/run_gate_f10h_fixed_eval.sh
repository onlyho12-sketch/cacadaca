#!/usr/bin/env bash
set -euo pipefail

repo=/home/rokey/cacadaca
isaac=/home/rokey/isaacsim-6.0.1/python.sh
parent="$repo/learning/rl/robot/releases/gate_e7_seed1_it211_20260831_201801/model_gate_e7_seed1_it211.pt"
residual="$repo/learning/rl/robot/results/gate_f10h_residual_ppo_train_20260901_223000/stage_3_freeform_cross_xy/residual_model.pt"
root="$repo/learning/rl/robot/results/gate_f10h_fixed_eval_20260901_231000"

mkdir -p "$root"
for kind in cylinder freeform; do
  for direction in same_xx cross_xy; do
    "$isaac" "$repo/learning/rl/gate_f10_curvature_physx_v2_eval.py" \
      --checkpoint "$parent" --shield-mode control \
      --surface-kind "$kind" --direction-mode "$direction" \
      --envs-per-profile 1 --surface-seed-base 60000 \
      --physics-seed 20260921 --policy-seed 20260922 \
      --out-dir "$root/control_${kind}_${direction}" --headless
    "$isaac" "$repo/learning/rl/gate_f10h_residual_eval.py" \
      --parent-checkpoint "$parent" --arm deterministic_g3 \
      --surface-kind "$kind" --direction-mode "$direction" \
      --envs-per-profile 1 --surface-seed-base 60000 \
      --physics-seed 20260921 --policy-seed 20260922 \
      --out-dir "$root/deterministic_g3_${kind}_${direction}" --headless
    "$isaac" "$repo/learning/rl/gate_f10h_residual_eval.py" \
      --parent-checkpoint "$parent" --residual-checkpoint "$residual" \
      --arm residual_ppo --surface-kind "$kind" --direction-mode "$direction" \
      --envs-per-profile 1 --surface-seed-base 60000 \
      --physics-seed 20260921 --policy-seed 20260922 \
      --out-dir "$root/residual_ppo_${kind}_${direction}" --headless
  done
done
