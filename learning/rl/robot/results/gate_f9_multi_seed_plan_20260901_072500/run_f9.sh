#!/usr/bin/env bash
# Gate F9 multi-seed paired validation runner (frozen plan 20260901_072500).
# Reuses learning/rl/gate_f8_force_safety_eval.py unmodified.
# 64 sequential invocations = 2 arms x 4 geometries x 2 directions x 4 seed bases.
# Resume-safe: a run directory that already contains metadata.json is skipped;
# a directory that exists without metadata.json aborts the script for human review.
set -euo pipefail
cd /home/rokey/cacadaca

PY=/home/rokey/isaacsim-6.0.1/python.sh
CKPT=learning/rl/robot/releases/gate_e7_seed1_it211_20260831_201801/model_gate_e7_seed1_it211.pt
EXPECTED_SHA=c733763bfebd7fcead01cf2574bbd1621c4660b9347143cccb35e0e83449d052
PLAN_DIR=learning/rl/robot/results/gate_f9_multi_seed_plan_20260901_072500
RESULTS=learning/rl/robot/results
BATCH_TS=20260901_072500
DIRLIST="$PLAN_DIR/f9_run_dirs.txt"

actual=$(sha256sum "$CKPT" | cut -d' ' -f1)
if [ "$actual" != "$EXPECTED_SHA" ]; then
  echo "FATAL: checkpoint hash mismatch: $actual" >&2
  exit 1
fi

touch "$DIRLIST"
total=0
done_count=0
for base in 40000 41000 42000 43000; do
  for dm in same_xx cross_xy; do
    short=${dm%%_*}
    for kind in flat cylinder sphere freeform; do
      for arm in control static_cap; do
        total=$((total + 1))
        out="$RESULTS/gate_f9_${arm}_${kind}_${short}_seed${base}_${BATCH_TS}"
        if [ -d "$out" ]; then
          if [ -f "$out/metadata.json" ]; then
            echo "[F9 $(date +%H:%M:%S)] skip complete $out"
            done_count=$((done_count + 1))
            continue
          fi
          echo "FATAL: incomplete run dir exists, human review required: $out" >&2
          exit 2
        fi
        echo "[F9 $(date +%F' '%T)] start ($total/64) $arm/$kind/$dm/seed$base -> $out"
        "$PY" learning/rl/gate_f8_force_safety_eval.py \
          --checkpoint "$CKPT" \
          --shield-mode "$arm" \
          --surface-kind "$kind" \
          --direction-mode "$dm" \
          --surface-seed-base "$base" \
          --physics-seed 20260901 \
          --policy-seed 20260831 \
          --envs-per-profile 1 \
          --max-control-steps 8000 \
          --out-dir "$out" \
          --headless
        # python.sh can exit 0 even when the evaluation aborted (e.g. the Isaac
        # asset server is unreachable), so trust the artifact, not the exit code.
        if [ ! -f "$out/metadata.json" ] || [ ! -f "$out/sequences.csv" ]; then
          echo "FATAL: run produced no metadata.json/sequences.csv despite exit 0: $out" >&2
          echo "FATAL: verify network access to the Isaac asset server, then resume." >&2
          exit 3
        fi
        echo "$out" >> "$DIRLIST"
        done_count=$((done_count + 1))
        echo "[F9 $(date +%F' '%T)] done  ($done_count/64) $out"
      done
    done
  done
done
echo "[F9 $(date +%F' '%T)] all $done_count/64 runs complete"
