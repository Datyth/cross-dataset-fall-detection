# Cross-Dataset Fall Detection

**Can a fall detector generalize to a dataset it has never seen?**

A reproducible, skeleton-based video classification pipeline built with **PyTorch, RTMPose and CTR-GCN**. It benchmarks Fall / Non-Fall recognition across four datasets using leave-one-dataset-out evaluation, with source-only model selection and an auditable prediction for every held-out recording.

![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9-EE4C2C?logo=pytorch&logoColor=white)
![Protocol](https://img.shields.io/badge/Evaluation-4--fold_LODO-0F4D92)
![Task](https://img.shields.io/badge/Task-Video_classification-42949E)

![Pipeline: RGB recordings, frozen pose extraction, skeleton windows, CTR-GCN and video-level aggregation](docs/figures/pipeline.png)

## Project highlights

- **520 audited recordings** from CAUCAFall, GMDCSA-24, Le2i and URFD, with frozen splits and SHA-256 provenance.
- **End-to-end CLI:** acquisition, dataset audit, pose extraction, visual previews, training and evaluation.
- **Temporal graph learning:** a COCO-17, single-person CTR-GCN adaptation with video-level multiple-instance learning (MIL).
- **Cross-dataset evaluation:** four held-out domains × three fixed seeds; model selection, thresholds and BatchNorm calibration use source data only.
- **ERM and SWAD implementations**, with regression tests for averaging, leakage prevention, deterministic resume and metric aggregation.

## Results

The bundled **ERM baseline** reaches **0.839 macro Fall F1** and **0.924 macro average precision**. These are results from saved real-data experiments, verified against predictions and checkpoint hashes during repository preparation.

![ERM performance on four unseen datasets, with mean and standard deviation across three training seeds](docs/figures/benchmark.png)

| Held-out dataset | Recordings | Fall F1 ↑ | Average precision ↑ |
|---|---:|---:|---:|
| CAUCAFall | 100 | 0.740 ± 0.064 | 0.848 ± 0.056 |
| GMDCSA-24 | 160 | 0.855 ± 0.028 | 0.922 ± 0.018 |
| Le2i | 190 | 0.874 ± 0.017 | 0.942 ± 0.020 |
| URFD | 70 | 0.887 ± 0.044 | 0.983 ± 0.007 |
| **Macro across datasets** | **520** | **0.839 ± 0.014** | **0.924 ± 0.011** |

Values are mean ± sample standard deviation over seeds **42, 123, 2026**. Macro scores give each dataset equal weight within each seed. Average precision is stored as `pr_auc` in the CSVs; it is computed with `sklearn.metrics.average_precision_score`.

Inspect the [per-seed metrics](benchmarks/erm/metrics.csv), [1,560 video predictions](benchmarks/erm/predictions.csv), [provenance](benchmarks/erm/provenance.json) and [evaluation notes](docs/EVALUATION.md). Checkpoints and raw datasets are excluded from this repository.

**SWAD status:** the implementation includes a version-2 averaging correction. At packaging time, 2 of 12 saved SWAD runs had been repaired; the other 10 and the aggregate report still needed updating. SWAD scores are therefore excluded from this snapshot.

## Research idea

We are testing whether learning to predict **future skeleton motion** helps the fall detector work on unseen datasets. The main CTR-GCN still classifies a whole video from its 64-frame windows. During training, a second branch takes the first 48 frames, predicts a representation of the last 16 frames, and compares it with a slowly updated EMA encoder. It also checks whether the prediction stays similar after small pose noise and joint dropout. **P3** uses future prediction; **P4** adds this consistency check. At inference, only the video classifier runs; it needs neither the prediction branch nor the EMA encoder.

![PDI research architecture: video classification above and a training-only future representation branch below](docs/figures/pdi_architecture.png)

This is a research hypothesis, not a reported improvement yet. The first paired P3/P4 experiment is queued after E2; it will test whether the extra consistency loss helps on CAUCAFall with one seed.

## Quick start

### Local review app

On Windows with Ubuntu/WSL installed, place `best_erm.pt` in `results/runs/urfd_2026/` and the original RTMDet/RTMPose `.pth` files in `data/model_weights/`. Model files remain local and are excluded from Git. From the project folder:

```powershell
.\setup-baseline.cmd   # First installation only; already completed on this PC.
.\start-baseline.cmd
```

Open **http://127.0.0.1:8000** for recording analysis, saved history and benchmark review. New analyses use the original **RTMDet-m + RTMPose-m AIC/COCO** weights and the verified URFD/2026 ERM checkpoint with its saved **0.20** threshold. The app checks runtime versions, compiled MMCV operations and pose-weight hashes. It does not fall back to ONNX. Legacy SWAD remains disabled.

Refresh with **Ctrl+F5** after updates. CPU analysis takes minutes per clip; if closing VS Code stops the app, run `start-baseline.cmd` again. See [local baseline setup](docs/BASELINE_LOCAL.md) and [app behavior and limitations](docs/APP.md).

### Research CLI

Use **Python 3.11**. The CPU checks below run without downloading datasets, model weights or the pose stack.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# CPU runtime for tests and the synthetic smoke test:
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt

python run.py --help
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q
python scripts/verify_benchmark.py
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python run.py train --smoke-test
```

The smoke test uses synthetic skeletons to check the model and loss path. The reported benchmark above comes from real recordings.

## Run the full pipeline

Full experiments additionally require dataset access, sufficient disk space, a CUDA-enabled PyTorch installation and the pose dependencies in [requirements-pose.txt](requirements-pose.txt). The recorded environment used CUDA 12.8. **MMCV requires compiled ops compatible with the installed PyTorch/CUDA stack**; a generic pip installation alone may require building those ops. See the [recorded runtime](benchmarks/erm/environment.json).

Run commands from the repository root. Paths and pinned pose-checkpoint URLs are in [config.yaml](config.yaml).

```bash
# Review sources, then download the versions used by this benchmark.
python run.py download --dataset all --source mirror --dry-run
python run.py download --dataset all --source mirror --resume
python run.py inspect

# Check a small pose sample before extracting all recordings.
python run.py extract --dataset all --limit 2 --device cuda --resume
python run.py preview --dataset all --limit 2
python run.py extract --dataset all --device cuda --resume

# Validate the bundled split against the reconstructed manifest.
python run.py split --seed 2026
python run.py inspect --check-splits

# Each ERM trajectory also produces its paired SWAD checkpoint.
python run.py train --method erm --fold all --seeds 42,123,2026
python run.py report --check-only
python run.py report
```

Training uses 15 fps skeletons, 64-frame windows with stride 32, a binary CTR-GCN head and top-two-window logit aggregation. The default batch is 64 videos, forwarded as interleaved chunks of up to 32 windows. Source-training windows alone recalibrate BatchNorm before source validation and final SWAD evaluation.

The bundled manifest and splits are frozen. If a downloaded source changes, split validation rejects a mismatch; preserve the published metadata and use a separate data directory in a copied config for a new benchmark version. Reproduction also depends on the upstream archives remaining available. See [dataset versions and limitations](docs/EVALUATION.md).

For interrupted training, repeat the command with `--resume`. `train --method swad` verifies the paired SWAD output of a completed trajectory. `repair-swad` supports migration of existing legacy runs; fresh training uses corrected averaging directly.

## Repository layout

```text
├── run.py                  # Command-line entry point
├── config.yaml             # Dataset, pose, model and evaluation settings
├── src/
│   ├── data.py             # Downloads, audit, grouping, splits and windows
│   ├── pose.py             # Frozen detection/pose extraction and previews
│   ├── model.py            # COCO-17 CTR-GCN and video MIL
│   ├── train.py            # ERM, SWAD, source-only selection and resume
│   ├── swad_repair.py      # Verified legacy averaging migration
│   └── report.py           # Target inference, metrics and reports
├── tests/                  # Pipeline and SWAD regression tests
├── data/                   # Tracked manifest and frozen splits only
├── benchmarks/erm/         # Lightweight, verified result snapshot
├── scripts/                # Benchmark verification and figure generation
├── docs/                   # Evaluation notes and README figures
└── third_party/            # Upstream license notices
```

Downloaded data, skeleton caches, checkpoints and generated training reports are ignored. The original workspace folders `app/`, `research/` and `figures4papers/` are also excluded; the tracked source runs independently from the repository root.

## Recreate the figures

```bash
python scripts/make_figures.py
python scripts/make_pdi_figure.py
```

The pipeline and benchmark figures use bundled audit metadata and ERM metrics; the PDI figure depicts the research architecture. The scripts export PNG, PDF and SVG using Chen Liu's [Figures for Papers](https://github.com/ChenLiu-1996/figures4papers) conventions: publication typography, semantic palette, clean axes and 300 dpi export. Its source checkout is unnecessary to regenerate the figures.

## References and attribution

- [CTR-GCN](https://github.com/Uason-Chen/CTR-GCN): backbone adapted to COCO-17, one person and a binary video-level task; upstream **CC BY-NC 4.0**.
- [SWAD](https://github.com/khanrc/swad): LossValley adapted to dense per-step averaging and source-domain validation; upstream **MIT**.
- [MMPose](https://github.com/open-mmlab/mmpose): frozen RTMPose-m, paired with RTMDet-m from [MMDetection](https://github.com/open-mmlab/mmdetection).
- Dataset sources and version differences are documented in [evaluation notes](docs/EVALUATION.md).

Upstream terms and modification notices are retained in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [third_party/](third_party/). Dataset access and reuse follow each dataset provider's terms.

GitHub setup commands and a CV description are in [publishing notes](docs/PUBLISH.md).
