#!/usr/bin/env bash
# Run from any directory. Safe to re-run after interruption; no E1 writes.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate datpt
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
python -u run_e2.py prepare
python -u run_e2.py preflight
mkdir -p research/results/e2/logs
for fold in caucafall gmdcsa24 le2i urfd; do
  for part in 0 1 2 3; do
    for seed in 42 123 2026; do
      log="research/results/e2/logs/${fold}_part${part}_seed${seed}.log"
      python -u run_e2.py train --fold "$fold" --part "$part" --seed "$seed" 2>&1 | tee -a "$log"
    done
  done
done
python -u run_e2.py report 2>&1 | tee -a research/results/e2/logs/report.log
