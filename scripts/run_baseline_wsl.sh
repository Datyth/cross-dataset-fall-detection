#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_root="${FALL_BASELINE_RUNTIME:-${XDG_DATA_HOME:-$HOME/.local/share}/fall-review-baseline}"
python_bin="$runtime_root/venv/bin/python"
port="${1:-8000}"
if [[ ! "$port" =~ ^[0-9]+$ ]] || ((port < 1 || port > 65535)); then
  echo 'Choose a port between 1 and 65535.' >&2; exit 1
fi
if [[ ! -x "$python_bin" ]]; then
  echo 'The isolated baseline runtime is missing. Run scripts/setup_baseline_runtime.sh first.' >&2
  exit 1
fi
unset VIRTUAL_ENV PYTHONHOME PYTHONPATH
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$project_dir"
exec "$python_bin" -X utf8 serve.py --pose-engine mmpose --device cpu --port "$port"
