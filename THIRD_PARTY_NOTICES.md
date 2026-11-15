# Third-party notices

## CTR-GCN

`src/model.py` adapts the official [CTR-GCN implementation](https://github.com/Uason-Chen/CTR-GCN), by Yuxin Chen and collaborators, distributed under [CC BY-NC 4.0](https://github.com/Uason-Chen/CTR-GCN/blob/main/LICENSE). A copy is retained at [third_party/CC-BY-NC-4.0.txt](third_party/CC-BY-NC-4.0.txt).

Modifications include the COCO-17 graph, single-person input, a binary classification head, accessible temporal features and video-level MIL aggregation. The original attribution and noncommercial terms continue to apply to the adapted material.

## SWAD

LossValley logic in `src/train.py` is adapted from [SWAD](https://github.com/khanrc/swad), Copyright (c) 2021 Junbum Cha, under its [MIT license](https://github.com/khanrc/swad/blob/main/LICENSE). The full notice is retained at [third_party/SWAD-MIT.txt](third_party/SWAD-MIT.txt).

Modifications include dense per-step parameter averages represented by segments, macro source-domain validation, source-training-only BatchNorm calibration and the version-2 duplicate-segment correction. `src/swad_repair.py` provides verified recovery of affected legacy artifacts.

## Figures for Papers

`scripts/make_figures.py` follows the palette, typography, helper conventions and export guidance in Chen Liu's [Figures for Papers](https://github.com/ChenLiu-1996/figures4papers), including its `scientific-figure-making` skill. That project is distributed under [CC BY-NC 4.0](https://github.com/ChenLiu-1996/figures4papers/blob/main/LICENSE); the retained license text is [third_party/CC-BY-NC-4.0.txt](third_party/CC-BY-NC-4.0.txt). Figures and plotting code here are created for this fall-detection project using its actual audit metadata and ERM snapshot; no unrelated example results or artwork are included.

## Datasets and dependencies

Dataset videos, pretrained detector/pose weights and training checkpoints are not distributed here. Dataset sources and versions appear in [docs/EVALUATION.md](docs/EVALUATION.md). Python dependencies retain their own licenses. No blanket MIT license is applied to this repository or its noncommercial upstream adaptations.
