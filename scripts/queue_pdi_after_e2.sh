#!/usr/bin/env bash
# Wait for the already running E2 shell, then run the PDI pilot only on success.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
e2_report="research/results/e2/REPORT.md"
e2_pid="${1:-}"
if [[ -z "$e2_pid" ]]; then
  echo "Usage: bash scripts/queue_pdi_after_e2.sh E2_SHELL_PID" >&2
  exit 2
fi
if ! [[ "$e2_pid" =~ ^[0-9]+$ ]]; then
  echo "Invalid E2 PID: $e2_pid" >&2
  exit 2
fi
echo "Waiting for E2 shell PID $e2_pid, starting $(date --iso-8601=seconds)"
while kill -0 "$e2_pid" 2>/dev/null; do
  sleep 30
done
if [[ ! -s "$e2_report" || ! -s research/results/e2/metrics.csv ]]; then
  echo "E2 ended without a complete report; PDI will not start." >&2
  exit 1
fi
completed="$(find research/results/e2/runs -mindepth 2 -maxdepth 2 -name provenance.json -type f | wc -l)"
if [[ "$completed" -ne 48 ]]; then
  echo "E2 report exists but only $completed/48 run provenance files; PDI will not start." >&2
  exit 1
fi
echo "E2 complete at $(date --iso-8601=seconds); starting PDI pilot."
bash "$project_root/scripts/run_pdi_pilot.sh"
echo "PDI pilot complete at $(date --iso-8601=seconds)."
