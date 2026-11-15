#!/usr/bin/env python3
"""Verify the portable ERM snapshot without datasets, weights or inference."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True).encode()).hexdigest()


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    base = ROOT / "benchmarks/erm"
    provenance = json.loads((base / "provenance.json").read_text())
    for name, expected in provenance["bundled_artifacts_sha256"].items():
        require(sha256(ROOT / name) == expected, f"Artifact checksum mismatch: {name}")
    splits = json.loads((ROOT / "data/splits.json").read_text())
    require(canonical_hash({k: v for k, v in splits.items() if k != "split_hash"})
            == splits["split_hash"], "Frozen split hash mismatch")
    require(sha256(ROOT / "data/manifest.csv") == splits["manifest_sha256"],
            "Manifest differs from frozen split")
    manifest = read_csv(ROOT / "data/manifest.csv")
    by_id = {r["video_id"]: r for r in manifest}
    require(len(by_id) == len(manifest) == 520, "Unexpected manifest size or duplicate IDs")
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    expected_runs = {(fold, seed) for fold in cfg["datasets"] for seed in cfg["train"]["seeds"]}
    runs = {(r["fold"], r["seed"]): r for r in provenance["runs"]}
    require(set(runs) == expected_runs and len(provenance["runs"]) == 12,
            "Missing or duplicate provenance runs")
    metrics = read_csv(base / "metrics.csv")
    predictions = read_csv(base / "predictions.csv")
    require(len(metrics) == 12 and len(predictions) == 1560, "Unexpected snapshot size")
    seen = set()
    for metric in metrics:
        key = (metric["fold"], int(metric["seed"]))
        require(key in expected_runs and key not in seen and metric["method"] == "erm",
                f"Unexpected metric identity: {key}")
        seen.add(key)
        run = runs[key]
        require(run["config_hash"] == canonical_hash(cfg), f"Config mismatch: {key}")
        require(run["split_hash"] == splits["split_hash"], f"Split mismatch: {key}")
        fold = splits["folds"][key[0]]
        train, val, target = (set(fold[k]) for k in ("train", "val", "target"))
        require(not (train & val or train & target or val & target), f"Split overlap: {key}")
        require(train | val | target == set(by_id), f"Incomplete split coverage: {key}")
        require(target == {r["video_id"] for r in manifest if r["dataset"] == key[0]},
                f"Incorrect held-out domain: {key}")
        rows = [r for r in predictions if (r["fold"], int(r["seed"])) == key]
        require(len(rows) == len(target) and {r["video_id"] for r in rows} == target,
                f"Missing or duplicated target predictions: {key}")
        require(len({r["threshold"] for r in rows}) == 1, f"Inconsistent threshold: {key}")
        for row in rows:
            require(row["method"] == "erm" and row["target_dataset"] == key[0]
                    and row["protocol_version"] == provenance["protocol"]
                    and row["checkpoint_hash"] == run["checkpoint_sha256"]
                    and row["split_hash"] == splits["split_hash"], f"Prediction lineage mismatch: {key}")
            require(row["y_true"] == by_id[row["video_id"]]["label"], "Prediction label mismatch")
        y = np.array([int(r["y_true"]) for r in rows])
        prob = np.array([float(r["y_prob"]) for r in rows])
        threshold = np.array([float(r["threshold"]) for r in rows])
        pred = (prob >= threshold).astype(int)
        require(np.isfinite(prob).all() and ((prob >= 0) & (prob <= 1)).all(), "Invalid probabilities")
        require(np.array_equal(pred, [int(r["y_pred"]) for r in rows]), "Stored decision mismatch")
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        actual = {
            "n": len(rows), "f1": f1_score(y, pred, zero_division=0),
            "f1_at_0_5": f1_score(y, prob >= .5, zero_division=0),
            "pr_auc": average_precision_score(y, prob),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "roc_auc": roc_auc_score(y, prob),
            "balanced_accuracy": balanced_accuracy_score(y, pred),
            "fall_prevalence": y.mean(), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        }
        for name, value in actual.items():
            require(np.isclose(value, float(metric[name]), rtol=0, atol=1e-12),
                    f"Recomputed metric mismatch: {key}/{name}")
    require(seen == expected_runs, "Missing metric rows")
    for name in ("f1", "pr_auc"):
        means = [np.mean([float(r[name]) for r in metrics if int(r["seed"]) == seed])
                 for seed in cfg["train"]["seeds"]]
        print(f"Macro {name}: {np.mean(means):.3f} ± {np.std(means, ddof=1):.3f}")
    print("Verified: artifact hashes, frozen splits, 12 ERM runs, 1,560 predictions and recomputed metrics.")
    print("Original checkpoint hashes are recorded; weights and inference are not bundled.")


if __name__ == "__main__":
    main()
