#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ISAAC_PYTHON="${ISAAC_PYTHON:-$HOME/isaacsim-6.0.1/python.sh}"

if [ ! -x "$ISAAC_PYTHON" ]; then
  echo "Isaac Sim Python을 찾을 수 없습니다: $ISAAC_PYTHON" >&2
  exit 1
fi

RESULT_TAG="normal_z"
PREV_ARG=""
for ARG in "$@"; do
  if [ "$PREV_ARG" = "--tag" ]; then
    RESULT_TAG="$ARG"
    break
  fi
  PREV_ARG="$ARG"
done

# Keep log paths inside the test directory even if an unsafe tag is supplied.
RESULT_TAG="${RESULT_TAG//[^a-zA-Z0-9_.-]/_}"
LOG_DIR="$HERE/results/$RESULT_TAG/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_PATH="$LOG_DIR/gloss_test_$TIMESTAMP.txt"

echo "[Gloss Test] full log: $LOG_PATH"
set +e
"$ISAAC_PYTHON" "$HERE/scripts/run_gloss_sweep.py" "$@" 2>&1 | tee "$LOG_PATH"
RUN_STATUS=${PIPESTATUS[0]}
set -e
STATUS_JSON="$HERE/results/$RESULT_TAG/run_status.json"
if [ ! -f "$STATUS_JSON" ] || ! grep -q '"success": true' "$STATUS_JSON"; then
  RUN_STATUS=1
fi
echo "[Gloss Test] process exit code: $RUN_STATUS" | tee -a "$LOG_PATH"
exit "$RUN_STATUS"
