#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAAC_PYTHON="${ISAAC_PYTHON:-/home/rokey/isaacsim-6.0.1/python.sh}"

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 ASSET.usd [--output-json PATH]" >&2
  exit 2
fi

PYTHONUNBUFFERED=1 "${ISAAC_PYTHON}" \
  "${ROOT_DIR}/scripts/inspect_usd_meshes.py" "$@"
