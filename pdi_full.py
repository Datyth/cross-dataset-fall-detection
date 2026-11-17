#!/usr/bin/env python3
"""Run the full P3/P4 comparison at each saved E1 run's optimizer budget."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from src import pdi_train
from src.data import DOMAINS, digest, load_manifest, load_splits
from src.train import _domain_balanced_order

ROOT = Path(__file__).resolve().parent
SEEDS = (42, 123, 2026)
METHODS = pdi_train.METHODS
OUT = ROOT / "research/results/pdi_full"


def config_for(fold, seed):
    base, pdi = pdi_train.configs()
    e1_dir = ROOT / "research/results/runs" / f"{fold}_{seed}"
    provenance = json.loads((e1_dir / "provenance.json").read_text())
    trace = json.loads((e1_dir / "trace.json").read_text())
    checkpoint = e1_dir / "best_erm.pt"
    split = load_splits(base)
    if (provenance["fold"] != fold or provenance["seed"] != seed
            or provenance["split_hash"] != split["split_hash"]
            or digest(checkpoint) != provenance["erm_checkpoint_hash"]):
        raise ValueError(f"E1 lineage mismatch: {fold} seed={seed}")
    if not trace or [r["epoch"] for r in trace] != list(range(1, len(trace) + 1)):
        raise ValueError(f"Invalid E1 epoch trace: {fold} seed={seed}")
    rows = {r["video_id"]: r for r in load_manifest(base)}
    train_ids = split["folds"][fold]["train"]
    steps = sum(math.ceil(sum(1 for _ in _domain_balanced_order(rows, train_ids, seed + epoch)) /
                          base["train"]["videos_per_batch"])
                for epoch in range(len(trace)))
    if steps != trace[-1]["step"]:
        raise ValueError(f"Optimizer budget mismatch: {fold} seed={seed}")
    pdi.update(protocol="pdi-full-e1-budget-v1", results_dir="research/results/pdi_full",
               max_epochs=len(trace), e1_optimizer_steps=steps,
               e1_checkpoint_sha256=provenance["erm_checkpoint_hash"])
    return base, pdi


def preflight():
    budgets = {}
    for fold in DOMAINS:
        for seed in SEEDS:
            _, pdi = config_for(fold, seed)
            budgets[f"{fold}/{seed}"] = {"epochs": pdi["max_epochs"],
                                         "optimizer_steps": pdi["e1_optimizer_steps"]}
    return budgets


def train(fold, seed, method, stop_after_epoch=None):
    base, pdi = config_for(fold, seed)
    result = pdi_train.train(base, pdi, method, fold, seed, stop_after_epoch)
    if not result.get("incomplete"):
        trace = json.loads((pdi_train.run_directory(pdi, method, fold, seed) / "trace.json").read_text())
        if len(trace) != pdi["max_epochs"]:
            raise ValueError("PDI epoch budget mismatch")
    return result


def report_pair(fold, seed):
    base, pdi = config_for(fold, seed)
    result = pdi_train.report(base, pdi, fold, seed)
    result["status"] = "full run; compare with E1 using the same fold, seed and optimizer budget"
    result["e1_epochs"] = pdi["max_epochs"]
    result["e1_optimizer_steps"] = pdi["e1_optimizer_steps"]
    directory = OUT / "paired" / fold / str(seed)
    (directory / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [f"# PDI full: {fold}, seed {seed}", "",
             f"P3 and P4 each ran {pdi['max_epochs']} epochs and {pdi['e1_optimizer_steps']} optimizer steps, matching E1.",
             "Checkpoints and thresholds were selected using source validation only.", "",
             "| Method | Fall F1 | AP | Precision | Recall | FP | FN |",
             "|---|---:|---:|---:|---:|---:|---:|",]
    for method, key in ((METHODS[0], "p3"), (METHODS[1], "p4")):
        m = result[key]
        lines.append(f"| {method} | {m['f1']:.3f} | {m['pr_auc']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | {m['fp']} | {m['fn']} |")
    lines.append("")
    (directory / "REPORT.md").write_text("\n".join(lines))
    return result


def aggregate():
    e1 = {}
    with (ROOT / "benchmarks/erm/metrics.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            e1[row["fold"], int(row["seed"])] = row
    rows = []
    for fold in DOMAINS:
        for seed in SEEDS:
            base, pdi = config_for(fold, seed)
            for method in METHODS:
                pdi_train.verify_run(base, pdi, method, fold, seed)
            pair = OUT / "paired" / fold / str(seed)
            result = json.loads((pair / "metrics.json").read_text())
            predictions = list(csv.DictReader((pair / "predictions.csv").open(newline="")))
            n = len(load_splits(base)["folds"][fold]["target"])
            if len(predictions) != 2 * n:
                raise ValueError(f"Incomplete predictions: {fold} seed={seed}")
            old = e1[fold, seed]
            rows.append({"fold": fold, "seed": seed, "n": n,
                         "e1_f1": float(old["f1"]), "p3_f1": result["p3"]["f1"],
                         "p4_f1": result["p4"]["f1"],
                         "p4_minus_e1_f1": result["p4"]["f1"] - float(old["f1"]),
                         "p4_minus_p3_f1": result["delta_f1"],
                         "e1_ap": float(old["pr_auc"]), "p3_ap": result["p3"]["pr_auc"],
                         "p4_ap": result["p4"]["pr_auc"],
                         "p4_minus_e1_ap": result["p4"]["pr_auc"] - float(old["pr_auc"]),
                         "p4_minus_p3_ap": result["delta_ap"]})
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    lines = ["# Full PDI comparison with E1", "",
             "Four held-out domains and three paired seeds. P3/P4 match each E1 run's epoch and optimizer-step budget.",
             "Model selection and thresholds use source validation only. CAUCAFall seed 42 was explored in the pilot.", "",
             "| Target | E1 F1 | P3 F1 | P4 F1 | P4−E1 | P4−P3 |",
             "|---|---:|---:|---:|---:|---:|"]
    for fold in (*DOMAINS, "Macro"):
        subset = rows if fold == "Macro" else [r for r in rows if r["fold"] == fold]
        values = [np.mean([r[k] for r in subset]) for k in
                  ("e1_f1", "p3_f1", "p4_f1", "p4_minus_e1_f1", "p4_minus_p3_f1")]
        lines.append("| " + fold + " | " + " | ".join(f"{v:.3f}" for v in values) + " |")
    lines += ["", "AP and all per-seed scores are in metrics.csv. Target-domain labels were used only for final scoring.", ""]
    (OUT / "REPORT.md").write_text("\n".join(lines))
    return {"runs": len(rows) * 2, "report": str(OUT / "REPORT.md")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    t = sub.add_parser("train")
    t.add_argument("--fold", required=True, choices=DOMAINS)
    t.add_argument("--seed", required=True, type=int, choices=SEEDS)
    t.add_argument("--method", required=True, choices=METHODS)
    t.add_argument("--stop-after-epoch", type=int)
    r = sub.add_parser("report-pair")
    r.add_argument("--fold", required=True, choices=DOMAINS)
    r.add_argument("--seed", required=True, type=int, choices=SEEDS)
    sub.add_parser("report")
    args = parser.parse_args()
    if args.command == "preflight": result = preflight()
    elif args.command == "train": result = train(args.fold, args.seed, args.method, args.stop_after_epoch)
    elif args.command == "report-pair": result = report_pair(args.fold, args.seed)
    else: result = aggregate()
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
