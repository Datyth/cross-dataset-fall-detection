# Domain diagnostics P0–P3

Run from the repository root using Conda `datpt`. These commands use CPU only.

```bash
conda activate datpt
python scripts/analyze_domains.py p0
python scripts/analyze_domains.py p1
python scripts/analyze_domains.py p2
python scripts/analyze_domains.py p3
```

Run each stage after the preceding command succeeds. P0 hashes sources, reconstructed clips and skeleton caches, checks frozen splits and records `ds-v1`. P1 extracts RGB statistics from eight unique representative frame positions at most. P2 extracts eight skeleton features from observed frames. P3 runs 24 MMD permutation tests, balanced sampling sensitivity and grouped C2ST. P1/P2 save after every video and resume completed rows when rerun.

Outputs are local at `research/results/domain_analysis/`. Read `audit_summary.md`, `rgb_summary.csv`, `skeleton_summary.csv`, `mmd_tests.csv`, `c2st.csv` and `REPORT.md`. Plots are written alongside the tables. Inspect the feature summaries before P3, including missing features and the relationship between masks and confidence. Manual review of suspicious poses remains necessary.

The skeleton cache stores image coordinates and a confidence-thresholded validity mask. This analysis normalizes x/W and y/H for diagnostics as specified in the plan. The classifier's original preprocessing remains as recorded by E1. Missing-mask and low-confidence features may coincide because the original pose mask already includes a confidence threshold. Velocity is undefined for caches with fallback timestamps.

MMD uses global feature scaling, median imputation with missing indicators and one fixed bandwidth per family. Group tests average standardized feature vectors per known subject, otherwise per recording group. Holm adjusts all 24 grouped tests. Video-level permutation results are exploratory. Group exchangeability is an assumption, with limited inference from GMDCSA's four subjects and unknown subject identities in Le2i/URFD. C2ST preprocessing is fitted within group-held-out folds. Invalid grouped folds produce `not_estimable`.

The analysis establishes differences on the selected features. It does not identify a causal contribution to classification errors. P0–P3 produce no new fall-classifier training runs.
