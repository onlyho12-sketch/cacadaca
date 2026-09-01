#!/usr/bin/env bash
set -euo pipefail
cd /home/rokey/cacadaca
PY=/home/rokey/isaacsim-6.0.1/python.sh
CKPT=learning/rl/robot/releases/gate_e7_seed1_it211_20260831_201801/model_gate_e7_seed1_it211.pt
EXPECTED_CKPT=c733763bfebd7fcead01cf2574bbd1621c4660b9347143cccb35e0e83449d052
EXPECTED_SUPERVISOR=14d641b14838be85f300044f887774dc4420b6fc64b4ca3d05feb0989dc3b5c5
EXPECTED_EVALUATOR=1986db1ffa312cf4308706f2b3a2a2992197b8072b6aa8b96535183ff28f90e2
test "$(sha256sum "$CKPT" | cut -d' ' -f1)" = "$EXPECTED_CKPT"
test "$(sha256sum learning/rl/gate_f10_substep_supervisor.py | cut -d' ' -f1)" = "$EXPECTED_SUPERVISOR"
test "$(sha256sum learning/rl/gate_f10_curvature_physx_v3_eval.py | cut -d' ' -f1)" = "$EXPECTED_EVALUATOR"
for item in flat:same_xx flat:cross_xy cylinder:cross_xy freeform:same_xx freeform:cross_xy; do
  kind=${item%%:*}
  direction=${item##*:}
  short=${direction%%_*}
  out="learning/rl/robot/results/gate_f10g3_curvature_safety_v3_${kind}_${short}_seed40000_20260901_210000"
  test ! -e "$out"
  "$PY" learning/rl/gate_f10_curvature_physx_v3_eval.py \
    --checkpoint "$CKPT" --shield-mode curvature_safety_v3 \
    --surface-kind "$kind" --direction-mode "$direction" --curvature-radius-m 0.60 \
    --surface-seed-base 40000 --physics-seed 20260901 --policy-seed 20260831 \
    --envs-per-profile 1 --max-control-steps 8000 --out-dir "$out" --headless
  test -f "$out/metadata.json" -a -f "$out/sequences.csv"
done
