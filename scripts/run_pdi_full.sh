#!/usr/bin/env bash
# Full source-only P3/P4 comparison, resumable at each epoch.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate datpt
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
mkdir -p research/results/pdi_full/logs
python -u pdi_full.py preflight | tee -a research/results/pdi_full/logs/preflight.log
for fold in caucafall gmdcsa24 le2i urfd; do
  for seed in 42 123 2026; do
    for method in p3_future_aug p4_pdi; do
      echo "Starting $method $fold seed=$seed at $(date --iso-8601=seconds)"
      python -u pdi_full.py train --method "$method" --fold "$fold" --seed "$seed" \
        2>&1 | tee -a "research/results/pdi_full/logs/${method}_${fold}_${seed}.log"
    done
    python -u pdi_full.py report-pair --fold "$fold" --seed "$seed" \
      2>&1 | tee -a "research/results/pdi_full/logs/report_${fold}_${seed}.log"
  done
done
python -u pdi_full.py report 2>&1 | tee -a research/results/pdi_full/logs/report.log
