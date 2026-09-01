#!/usr/bin/env bash
set -euo pipefail
cd /home/rokey/cacadaca
PY=/home/rokey/isaacsim-6.0.1/python.sh
CKPT=learning/rl/robot/releases/gate_e7_seed1_it211_20260831_201801/model_gate_e7_seed1_it211.pt
EXPECTED_SHA=c733763bfebd7fcead01cf2574bbd1621c4660b9347143cccb35e0e83449d052
PLAN=learning/rl/robot/results/gate_f10g_plan_20260901_170500
actual=$(sha256sum "$CKPT" | cut -d' ' -f1)
test "$actual" = "$EXPECTED_SHA"
for kind in flat cylinder freeform; do
  for dm in same_xx cross_xy; do
    short=${dm%%_*}
    out="learning/rl/robot/results/gate_f10g_curvature_safety_${kind}_${short}_seed40000_20260901_170500"
    if [ -d "$out" ]; then
      test -f "$out/metadata.json" && continue
      echo "incomplete result directory requires review: $out" >&2
      exit 2
    fi
    "$PY" learning/rl/gate_f10_curvature_physx_eval.py \
      --checkpoint "$CKPT" --shield-mode curvature_safety \
      --surface-kind "$kind" --direction-mode "$dm" \
      --surface-seed-base 40000 --physics-seed 20260901 --policy-seed 20260831 \
      --curvature-radius-m 0.60 --envs-per-profile 1 --max-control-steps 8000 \
      --out-dir "$out" --headless
    test -f "$out/metadata.json" -a -f "$out/sequences.csv"
    echo "$out" >> "$PLAN/candidate_run_dirs.txt"
  done
done
