#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate datpt
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
mkdir -p research/results/pdi/logs
python -u pdi_run.py smoke 2>&1 | tee -a research/results/pdi/logs/smoke.log
python -u pdi_run.py audit 2>&1 | tee -a research/results/pdi/logs/audit.log
for method in p3_future_aug p4_pdi; do
  python -u pdi_run.py train --method "$method" --fold caucafall --seed 42 2>&1 | tee -a "research/results/pdi/logs/${method}_caucafall_42.log"
done
python -u pdi_run.py report --fold caucafall --seed 42 2>&1 | tee -a research/results/pdi/logs/report_caucafall_42.log
