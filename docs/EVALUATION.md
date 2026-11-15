# Evaluation and reproducibility

## Task and protocol

`e1-v1` evaluates video-level Fall / Non-Fall classification. For each of four folds, one complete dataset is held out as the target domain. Training and validation use only the other three datasets. Seeds are 42, 123 and 2026; split seed is 2026.

Source train/validation grouping uses subjects when available and suitable, otherwise recording groups. Splits happen before windowing. No recording crosses source train, source validation or target partitions. The exact grouping used by each fold is in [splits.json](../data/splits.json).

Source validation selects the ERM checkpoint, SWAD loss valley and decision threshold. BatchNorm statistics are recalibrated using source-training windows only. Target labels are used for final metrics, never for selection.

## Dataset audit

| Dataset | Audited version | Videos | Fall | Non-Fall |
|---|---|---:|---:|---:|
| CAUCAFall | Zenodo 17170592, reconstructed recordings | 100 | 50 | 50 |
| GMDCSA-24 | Zenodo 13354453, v2.1 | 160 | 79 | 81 |
| Le2i | Zenodo 17170592, reconstructed recordings | 190 | 130 | 60 |
| URFD | Official camera-0 RGB events | 70 | 30 | 40 |
| Total | | 520 | 289 | 231 |

CAUCAFall uses 258 unique processed clips grouped into 100 recordings through annotations and clip timestamps. Duplicate clip bytes are excluded. Le2i uses 967 clips reconstructed into 190 recordings. These versions come from the [processed mirror](https://zenodo.org/records/17170592); they should not be silently equated with the original releases. Membership relative to the PHARL 190-video Le2i subset is unverified.

The [GMDCSA-24 v2.1 archive](https://zenodo.org/records/13354453) contains 79 Fall / 81 ADL videos, reversing the counts stated in its paper. Labels follow the actual source folders and class annotations. [URFD](https://fenix.ur.edu.pl/~mkepski/ds/uf.html) contributes 70 independent events from camera 0; synchronized views are not counted as extra cases. One URFD ADL sequence has 20 trailing frames without timestamps; the preceding cadence is used to extrapolate them.

Every recording's source version, relative source path, grouping and SHA-256 are retained in the [manifest](../data/manifest.csv). No video frames or pose caches are distributed here.

## Model and temporal aggregation

Frozen RTMDet-m + RTMPose-m produce one tracked person's COCO-17 skeleton. Timestamped poses are resampled to 15 fps and normalized in image coordinates. Cached sequences retain joint confidence, missing-joint masks and timestamps. Short sequences repeat the last visible pose for padding while retaining masks.

The classifier receives `[N, 3, 64, 17, 1]` skeleton windows, with three channels for normalized x, normalized y and confidence. Window stride is 32. The average of the highest two window logits (or the single available logit) gives the video logit. Binary cross-entropy supervises this video logit, so a fall recording does not assign a Fall label to every window.

CTR-GCN is trained from scratch using AdamW. Defaults are 150 maximum epochs, learning rate 0.0003, weight decay 0.0001 and 64 videos per batch. The binary head uses initialization standard deviation 0.01. ERM and SWAD share one optimization trajectory per fold/seed.

## What the bundled result snapshot verifies

The snapshot contains 12 ERM metric rows and 1,560 predictions: 520 held-out videos evaluated once per seed. During packaging, each saved ERM checkpoint hash was compared with its run provenance and prediction rows. Fall F1 and average precision were recomputed from the saved probabilities and source-selected thresholds.

[provenance.json](../benchmarks/erm/provenance.json) records checkpoint hashes, original result-file hashes and hashes of the bundled artifacts. [environment.json](../benchmarks/erm/environment.json) records the original Python/PyTorch/CUDA runtime. Checkpoints are not bundled, so verification after cloning checks metadata, hashes, coverage and recomputed metrics; it does not rerun inference or independently inspect the original weights.

```bash
python scripts/verify_benchmark.py
```

F1 uses the per-run threshold chosen from source validation. `f1_at_0_5` provides a fixed-threshold reference. `pr_auc` means average precision, rather than trapezoidal integration. Macro scores average the four domains within each seed; reported standard deviations use `ddof=1`. Per-run confidence intervals in `metrics.csv` use 1,000 recording/group bootstrap resamples and represent a different uncertainty source from training-seed variation.

## SWAD result status

The legacy LossValley implementation counted a convergence segment twice and reported an incorrect averaging start step. The current implementation uses averaging version 2 and enforces one contribution per optimizer step. Regression tests cover a fresh corrected average, legacy recovery, and mismatched or unrepaired artifact rejection.

At packaging time, the workspace contained 2 repaired runs and 10 legacy runs. The previous aggregate SWAD results predated full correction and are excluded from the public snapshot. Fresh runs use version 2. A completed legacy workspace can use `python run.py repair-swad --fold all --seeds 42,123,2026 --resume`, followed by `report --check-only` and `report --force`; this requires its original checkpoints and data caches.

## Interpretation and limits

These results establish a cross-dataset baseline on these particular dataset versions. Small target datasets, pose extraction errors, class imbalance, source-validation selection bias and top-two MIL aggregation affect the estimates. The experiment does not establish real-world deployment performance or a benefit from future-dynamics prediction.

The dataset/evaluation choice was inspired by [PHARL](https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2026.1837781/full). This project uses a skeleton CTR-GCN pipeline and does not reproduce PHARL's RGB model or claim its reported scores.
