"""Blinded target inference, recording-level metrics and generated report."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

from .data import DOMAINS, canonical_hash, digest, load_manifest, load_splits, load_video
from .model import CTRGCN
from .train import (evaluate_source, video_logit, implementation_hash, verify_implementation,
                    verify_erm_implementation, verify_swad_summary, validate_completed_run)


PRED_FIELDS = ("protocol_version", "target_dataset", "fold", "seed", "method", "video_id",
               "y_true", "video_logit", "y_prob", "threshold", "y_pred", "n_windows",
               "checkpoint_hash", "split_hash", "missing_frame_fraction", "empty_skeleton")


def video_metrics(rows):
    y = np.array([int(r["y_true"]) for r in rows])
    p = np.array([float(r["y_prob"]) for r in rows])
    t = np.array([float(r["threshold"]) for r in rows])
    pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"n": len(rows), "fall_prevalence": float(y.mean()),
            "fraction_with_missing_skeletons": float(np.mean([float(r.get("missing_frame_fraction", 0)) > 0 for r in rows])),
            "fraction_empty_skeletons": float(np.mean([int(r.get("empty_skeleton", 0)) for r in rows])),
            "f1": f1_score(y, pred, zero_division=0),
            "f1_at_0_5": f1_score(y, p >= .5, zero_division=0),
            "pr_auc": average_precision_score(y, p) if len(np.unique(y)) == 2 else float("nan"),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "roc_auc": roc_auc_score(y, p) if len(np.unique(y)) == 2 else float("nan"),
            "balanced_accuracy": balanced_accuracy_score(y, pred),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}


def bootstrap_ci(rows, groups, metric, samples=1000, seed=2026):
    rng = np.random.default_rng(seed)
    members = defaultdict(list)
    for row in rows:
        members[groups.get(row["video_id"]) or row["video_id"]].append(row)
    keys = sorted(members)
    values = []
    for _ in range(samples):
        selected = rng.choice(keys, len(keys), replace=True)
        draw = [row for key in selected for row in members[key]]
        value = video_metrics(draw)[metric]
        if np.isfinite(value):
            values.append(value)
    return tuple(np.quantile(values, [.025, .975])) if values else (float("nan"), float("nan"))


def macro_seed_metrics(metrics, method, key, seeds):
    """Give each held-out dataset equal weight within each training seed."""
    values = []
    for seed in seeds:
        per_domain = []
        for domain in DOMAINS:
            found = [float(m[key]) for m in metrics if m["method"] == method and m["seed"] == seed and m["fold"] == domain]
            if len(found) != 1:
                raise ValueError(f"Expected one {method}/{seed}/{domain} metric row; got {len(found)}")
            per_domain.append(found[0])
        values.append(float(np.mean(per_domain)))
    return np.array(values)


def _write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def report_preflight(cfg):
    runs = []
    for fold in DOMAINS:
        for seed in cfg["train"]["seeds"]:
            validate_completed_run(cfg, fold, seed)
            runs.append(f"{fold}_{seed}")
    return {"verified_runs": runs, "count": len(runs), "target_inference": False}


def report(cfg, force=False):
    # Validate EVERY run before performing any target inference or replacing
    # existing predictions. Old valid=True flags alone cannot certify SWAD.
    report_preflight(cfg)
    out = Path(cfg["paths"]["results"])
    out.mkdir(parents=True, exist_ok=True)
    pred_path = out / "predictions.csv"
    if pred_path.exists() and not force:
        raise RuntimeError("predictions.csv exists; use --force to regenerate target inference")
    splits = load_splits(cfg)
    manifest = load_manifest(cfg)
    rows_by_id = {r["video_id"]: r for r in manifest}
    predictions, source_rows, unavailable = [], [], []
    device = cfg["train"]["device"]
    for fold in DOMAINS:
        for seed in cfg["train"]["seeds"]:
            run_dir = out / "runs" / f"{fold}_{seed}"
            for method, filename in (("erm", "best_erm.pt"), ("swad", "swad.pt")):
                path = run_dir / filename
                if not path.exists():
                    if method == "swad" and (run_dir / "provenance.json").exists():
                        provenance = json.loads((run_dir / "provenance.json").read_text())
                        if not provenance["swad"]["valid"]:
                            unavailable.append((fold, seed, provenance["swad"].get("reason", "No valid source-validation loss valley")))
                            continue
                    raise FileNotFoundError(f"Missing {path}; complete real training before report")
                checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                if method == "erm":
                    verify_erm_implementation(checkpoint)
                else:
                    verify_implementation(checkpoint, implementation_hash())
                    verify_swad_summary(checkpoint["extra"])
                if checkpoint["split_hash"] != splits["split_hash"]:
                    raise ValueError("Checkpoint/split mismatch")
                if canonical_hash(checkpoint["config"]) != canonical_hash(cfg):
                    raise ValueError("Checkpoint/config mismatch")
                model = CTRGCN(**checkpoint["config"]["model"]).to(device)
                model.load_state_dict(checkpoint["model"]); model.eval()
                threshold = checkpoint["threshold"]
                ck_hash = digest(path)
                _, source_preds = evaluate_source(model, cfg, rows_by_id, splits["folds"][fold]["val"], device)
                source_domains = sorted({rows_by_id[x[0]]["dataset"] for x in source_preds})
                source_f1 = float(np.mean([
                    f1_score([x[1] for x in source_preds if rows_by_id[x[0]]["dataset"] == domain],
                             [x[2] >= threshold for x in source_preds if rows_by_id[x[0]]["dataset"] == domain],
                             zero_division=0) for domain in source_domains]))
                source_rows.append({"fold": fold, "seed": seed, "method": method,
                                    "source_val_f1": source_f1})
                for vid in splits["folds"][fold]["target"]:
                    row = rows_by_id[vid]
                    windows, _ = load_video(cfg, row)
                    pose_path = Path(cfg["paths"]["data"]) / "skeletons" / row["dataset"] / (vid + ".npz")
                    with np.load(pose_path) as pose:
                        valid = pose["valid_frame"]
                        missing_fraction = float(1 - valid.mean())
                        empty = int(not valid.any())
                    with torch.no_grad():
                        logit = float(video_logit(model, windows, device, cfg["train"]["window_chunk"]))
                    probability = float(torch.sigmoid(torch.tensor(logit)))
                    predictions.append(dict(zip(PRED_FIELDS, (
                        cfg["evaluation"]["protocol_version"], fold, fold, seed, method, vid,
                        int(row["label"]), logit, probability, threshold, int(probability >= threshold),
                        len(windows), ck_hash, splits["split_hash"], missing_fraction, empty))))
    _write_csv(pred_path, predictions, PRED_FIELDS)
    metrics = []
    groups = {r["video_id"]: r["recording_group_id"] for r in manifest}
    for fold in DOMAINS:
        for seed in cfg["train"]["seeds"]:
            for method in ("erm", "swad"):
                subset = [r for r in predictions if r["fold"] == fold and int(r["seed"]) == seed and r["method"] == method]
                if not subset:
                    continue
                result = video_metrics(subset)
                result.update({"fold": fold, "target_dataset": fold, "seed": seed, "method": method})
                result["f1_ci_low"], result["f1_ci_high"] = bootstrap_ci(subset, groups, "f1", cfg["evaluation"]["bootstrap_resamples"], cfg["evaluation"]["bootstrap_seed"])
                result["pr_auc_ci_low"], result["pr_auc_ci_high"] = bootstrap_ci(subset, groups, "pr_auc", cfg["evaluation"]["bootstrap_resamples"], cfg["evaluation"]["bootstrap_seed"])
                metrics.append(result)
    _write_csv(out / "metrics.csv", metrics, list(metrics[0]))
    _write_csv(out / "source_validation.csv", source_rows, list(source_rows[0]))
    _figures(out, metrics, source_rows)
    _markdown(out, cfg, manifest, splits, metrics, source_rows, unavailable)
    return len(predictions)


def _figures(out, metrics, source_rows):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, title in zip(axes, ("f1", "pr_auc"), ("Fall F1", "PR-AUC")):
        x = np.arange(4)
        for j, method in enumerate(("erm", "swad")):
            means = [np.mean([m[key] for m in metrics if m["fold"] == d and m["method"] == method])
                     if any(m["fold"] == d and m["method"] == method for m in metrics) else float("nan") for d in DOMAINS]
            ax.bar(x + (j - .5) * .35, means, .35, label=method.upper())
            for position, value in zip(x + (j - .5) * .35, means):
                if np.isnan(value):
                    ax.text(position, .02, "n/a", ha="center", rotation=90, fontsize=8)
        ax.set_xticks(x, DOMAINS, rotation=25); ax.set_ylim(0, 1); ax.set_title(title)
    axes[0].legend(); fig.tight_layout(); fig.savefig(out / "target_metrics.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    for method, marker in (("erm", "o"), ("swad", "s")):
        xs, ys = [], []
        for s in source_rows:
            if s["method"] != method:
                continue
            m = next(x for x in metrics if x["fold"] == s["fold"] and x["seed"] == s["seed"] and x["method"] == method)
            xs.append(s["source_val_f1"]); ys.append(m["f1"])
        ax.scatter(xs, ys, marker=marker, label=method.upper())
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Source validation F1 (used for model selection)", ylabel="Unseen target F1")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "source_vs_target.png"); plt.close(fig)
    fig, axes = plt.subplots(2, 4, figsize=(12, 5))
    for j, domain in enumerate(DOMAINS):
        for i, method in enumerate(("erm", "swad")):
            subset = [m for m in metrics if m["fold"] == domain and m["method"] == method]
            matrix = np.array([[np.mean([m["tn"] for m in subset]), np.mean([m["fp"] for m in subset])],
                               [np.mean([m["fn"] for m in subset]), np.mean([m["tp"] for m in subset])]]) if subset else np.zeros((2, 2))
            axes[i, j].imshow(matrix, cmap="Blues")
            axes[i, j].set_title(f"{domain} {method.upper()}" + (" (unavailable)" if not subset else ""))
            for a in range(2):
                for b in range(2):
                    axes[i, j].text(b, a, f"{matrix[a,b]:.1f}", ha="center", va="center")
    fig.tight_layout(); fig.savefig(out / "confusion_matrices.png"); plt.close(fig)


def _markdown(out, cfg, manifest, splits, metrics, source_rows, unavailable):
    lines = ["# E1 report", "", "Four-fold leave-one-dataset-out video-level Fall/Non-Fall evaluation.",
             "PHARL-inspired dataset selection; CTR-GCN skeleton classifier, not PHARL RGB model.",
             "URFD uses 70 independent cam0 events when the official archive is complete.", "",
             "## Dataset audit", "", "| Domain | Version | Videos | Fall | ADL | Known subjects | Median duration (s) |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for domain in DOMAINS:
        subset = [r for r in manifest if r["dataset"] == domain]
        lines.append(f"| {domain} | {cfg['datasets'][domain]['version']} | {len(subset)} | {sum(int(r['label']) for r in subset)} | {sum(1-int(r['label']) for r in subset)} | {len({r['subject_id'] for r in subset if r['subject_id']})} | {np.median([float(r['duration_s']) for r in subset]):.1f} |")
    lines += ["", "Source locations: " + "; ".join(f"[{d}]({cfg['datasets'][d]['url']})" for d in DOMAINS) + ".",
              "GMDCSA-24 v2.1 file folders contain 79 Fall and 81 ADL videos, reversing the count stated in the data paper. CAUCAFall and Le2i use reconstructed, mirror-derived recording groups; Le2i membership relative to PHARL is unverified.",
              "URFD uses official camera-0 RGB and frame-timestamp CSVs; one ADL sequence has 20 trailing frames without source timestamps, extrapolated from its preceding cadence."]
    raw = Path(cfg["paths"]["data"]) / "raw"
    lines += ["", "Verified source archives:", ""]
    for domain in ("caucafall", "gmdcsa24", "le2i"):
        for path in sorted((raw / domain).glob("*.zip")) + sorted((raw / domain).glob("*/*.zip")):
            lines.append(f"- `{path}`: SHA-256 `{digest(path)}`, {path.stat().st_size} bytes")
    lines += ["", "Pose checkpoint SHA-256:", ""]
    for sidecar in sorted((Path(cfg["paths"]["data"]) / "model_weights").glob("*.sha256")):
        lines.append(f"- `{sidecar.name[:-7]}`: `{sidecar.read_text().strip()}`")
    lines += ["", f"Manifest SHA-256: `{splits['manifest_sha256']}`; split SHA-256: `{splits['split_hash']}`.",
              "Every target video is evaluated once per seed and method. Source validation is used for model selection.", "",
              "## Preprocessing and training", "", "15 fps; 64-frame windows with stride 32; image-frame normalization; top-two-logit video MIL. Short videos repeat the last observed pose as a padding placeholder and retain a mask.",
              f"Detector: {cfg['pose']['detector_config']} ({cfg['pose']['detector_checkpoint']}); pose: {cfg['pose']['pose_config']} ({cfg['pose']['pose_checkpoint']}).",
              f"CTR-GCN trained from scratch using AdamW. Video batch: {cfg['train']['videos_per_batch']}; gradient accumulation: {cfg['train']['gradient_accumulation']}; window chunk: {cfg['train']['window_chunk']}.",
              f"Binary head initialization std: {cfg['model']['classifier_init_std']}. ERM validation and final SWAD both use deterministic source-train-only BN calibration with shuffled batches of {cfg['train']['bn_calibration_batch_size']} windows (seed {cfg['train']['bn_calibration_seed']}); validation and target videos never contribute BN statistics.",
              "SWAD uses source-validation LossValley on the shared ERM trajectory. Averaging version 2 counts each selected optimizer state exactly once. Legacy SWAD checkpoints are rejected until repaired; repaired checkpoints retain legacy FP32 average rounding error, documented in each run's `swad_repair.json`. ERM checkpoints remain byte-for-byte unchanged by repair.", "", "## LODO results", "",
              "| Target | Method | F1 mean ± SD | PR-AUC mean ± SD | F1@0.5 mean |", "|---|---|---:|---:|---:|"]
    for domain in DOMAINS:
        for method in ("erm", "swad"):
            subset = [m for m in metrics if m["fold"] == domain and m["method"] == method]
            if not subset:
                lines.append(f"| {domain} | {method.upper()} | unavailable | unavailable | unavailable |")
                continue
            f1 = np.array([m["f1"] for m in subset]); ap = np.array([m["pr_auc"] for m in subset])
            sd_f1 = f1.std(ddof=1) if len(f1) > 1 else float("nan")
            sd_ap = ap.std(ddof=1) if len(ap) > 1 else float("nan")
            lines.append(f"| {domain} | {method.upper()} | {f1.mean():.3f} ± {sd_f1:.3f} | {ap.mean():.3f} ± {sd_ap:.3f} | {np.mean([m['f1_at_0_5'] for m in subset]):.3f} |")
    for method in ("erm", "swad"):
        try:
            per_seed = macro_seed_metrics(metrics, method, "f1", cfg["train"]["seeds"])
            per_seed_ap = macro_seed_metrics(metrics, method, "pr_auc", cfg["train"]["seeds"])
            lines.append(f"| Macro domains | {method.upper()} | {np.mean(per_seed):.3f} ± {np.std(per_seed, ddof=1):.3f} | {np.mean(per_seed_ap):.3f} ± {np.std(per_seed_ap, ddof=1):.3f} | — |")
        except ValueError:
            lines.append(f"| Macro domains | {method.upper()} | unavailable (incomplete four-domain grid) | unavailable | — |")
    if unavailable:
        lines += ["", "SWAD unavailable for these source-validation trajectories:"]
        lines += [f"- {fold}, seed {seed}: {reason}" for fold, seed, reason in unavailable]
    lines += ["", "All per-seed results, prevalence, confusion counts, and missing-pose fractions are in [metrics.csv](metrics.csv).",
              "All video-level predictions and source-only thresholds are in [predictions.csv](predictions.csv).", "",
              "### SWAD averaging ranges", "", "| Fold | Seed | Converged step | Averaged steps | States |", "|---|---:|---:|---|---:|"]
    for domain in DOMAINS:
        for seed in cfg["train"]["seeds"]:
            info = json.loads((out / "runs" / f"{domain}_{seed}" / "provenance.json").read_text())["swad"]
            lines.append(f"| {domain} | {seed} | {info['converge_step']} | {info['start_step']}–{info['end_step']} | {info['averaged_states']} |")
    lines += ["", "Intervals in metrics.csv are 1,000 group/video bootstrap resamples within each target and seed; they do not represent training-seed variation.",
              "", "![Target metrics](target_metrics.png)", "![Source validation versus target](source_vs_target.png)",
              "![Confusion matrices](confusion_matrices.png)", "", "## Phase 2 handoff", "",
              "Full timestamped skeleton caches are under `data/skeletons/`; the immutable split is `data/splits.json`.",
              "CTR-GCN `features()` exposes pre-pooling temporal features. Checkpoint and pose hashes are saved with run artifacts.",
              "Phase 2 sufficiency (≥96 valid 15-fps steps) by domain and class:", "", "| Domain | Class | Sufficient / total |", "|---|---|---:|"]
    for domain in DOMAINS:
        for label in (0, 1):
            subset = [r for r in manifest if r["dataset"] == domain and int(r["label"]) == label]
            enough = 0
            for row in subset:
                path = Path(cfg["paths"]["data"]) / "skeletons" / domain / (row["video_id"] + ".npz")
                with np.load(path) as z:
                    enough += int(z["valid_frame"].sum() >= 96)
            lines.append(f"| {domain} | {'Fall' if label else 'ADL'} | {enough}/{len(subset)} |")
    lines += [
              "", "## Limitations", "", "Small target datasets, pose errors, dataset-version variants, source-validation selection bias, and top-two MIL may affect estimates.",
              "", "## References", "", "[PHARL](https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2026.1837781/full), [CAUCAFall](https://data.mendeley.com/datasets/7w7fccy7ky/4), [GMDCSA-24](https://zenodo.org/records/13354453), [Le2i](https://imvia.ube.fr/fall-detection-dataset/), [URFD](https://fenix.ur.edu.pl/~mkepski/ds/uf.html), [CTR-GCN](https://github.com/Uason-Chen/CTR-GCN), [SWAD](https://github.com/khanrc/swad), [MMPose](https://mmpose.readthedocs.io/en/latest/)."]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
