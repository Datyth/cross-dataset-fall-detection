"""Paired ERM domain-exposure experiment using the frozen E1 pose caches."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from .data import DOMAINS, canonical_hash, digest, load_manifest, load_splits, load_video
from .model import CTRGCN
from .report import video_metrics
from .train import (evaluate_source, recalibrate_bn, restore_rng, rng_state,
                    select_threshold, set_seed, video_batch_logits, video_logit)

ROOT = Path(__file__).resolve().parents[1]
PARTS = (0, 1, 2, 3)
SEEDS = (42, 123, 2026)
PROTOCOL = "e2-domain-exposure-v1"


def paths():
    return ROOT / "research/data", ROOT / "research/results/e2"


def load_config():
    import yaml
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    data, results = paths()
    cfg["paths"] = {"data": str(data.resolve()), "results": str(results.resolve())}
    cfg["evaluation"]["protocol_version"] = PROTOCOL
    return cfg


def code_hash():
    return canonical_hash({name: digest(ROOT / name) for name in
                           ("src/e2.py", "src/data.py", "src/model.py", "src/train.py")})


def read_csv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def _partitions(rows, domain):
    target = [r for r in rows if r["dataset"] == domain]
    parts = [[] for _ in PARTS]
    if domain == "gmdcsa24":
        for i, subject in enumerate(sorted({r["subject_id"] for r in target})):
            parts[i] = sorted(r["video_id"] for r in target if r["subject_id"] == subject)
    elif domain == "caucafall":
        for i, subject in enumerate(sorted({r["subject_id"] for r in target})):
            parts[i % 4].extend(r["video_id"] for r in target if r["subject_id"] == subject)
    elif domain == "le2i":
        scenes = ({"Coffee_room_01", "Coffee_room_02"}, {"Home_01", "Home_02"},
                  {"Lecture_room"}, {"Office"})
        for i, scene_group in enumerate(scenes):
            parts[i] = sorted(r["video_id"] for r in target if r["scene_id"] in scene_group)
        if set().union(*scenes) != {r["scene_id"] for r in target}:
            raise ValueError("Le2i scene inventory changed")
    else:
        rng = np.random.default_rng(2026)
        for label in (0, 1):
            ids = sorted(r["video_id"] for r in target if int(r["label"]) == label)
            rng.shuffle(ids)
            for i, vid in enumerate(ids):
                parts[i % 4].append(vid)
    by_id = {r["video_id"]: r for r in target}
    if sorted(vid for part in parts for vid in part) != sorted(by_id):
        raise ValueError(f"{domain}: target parts do not cover each recording exactly once")
    for part in parts:
        if {int(by_id[vid]["label"]) for vid in part} != {0, 1}:
            raise ValueError(f"{domain}: test part lacks a class")
    key = "subject_id" if domain in ("caucafall", "gmdcsa24") else (
        "scene_id" if domain == "le2i" else "recording_group_id")
    groups = [{by_id[vid][key] for vid in part} for part in parts]
    if any(groups[i] & groups[j] for i in PARTS for j in PARTS if i < j):
        raise ValueError(f"{domain}: group leakage across target parts")
    return [sorted(part) for part in parts]


def make_split(cfg):
    rows = load_manifest(cfg)
    e1_cfg = copy.deepcopy(cfg)
    e1_cfg["paths"]["data"] = str(ROOT / "data")
    e1 = load_splits(e1_cfg)
    folds = {}
    for domain in DOMAINS:
        target = _partitions(rows, domain)
        original = e1["folds"][domain]
        folds[domain] = []
        for i in PARTS:
            held_out = set(target[i])
            target_train = sorted(set(original["target"]) - held_out)
            train = sorted(original["train"] + target_train)
            val = original["val"]
            if set(train) & (set(val) | held_out) or set(val) & held_out:
                raise ValueError("E2 split leakage")
            folds[domain].append({"train": train, "source_train": original["train"],
                                  "target_train": target_train, "val": val,
                                  "test": target[i]})
    artifact = {"protocol": PROTOCOL, "seed": 2026,
                "manifest_sha256": digest(ROOT / "data/manifest.csv"),
                "e1_split_hash": e1["split_hash"], "folds": folds}
    artifact["split_hash"] = canonical_hash(artifact)
    return artifact


def split(cfg, write=False):
    expected = make_split(cfg)
    path = paths()[1] / "splits.json"
    if path.exists():
        if json.loads(path.read_text()) != expected:
            raise ValueError("E2 split changed; refusing to replace frozen split")
    elif write:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(expected, indent=2) + "\n")
    else:
        raise FileNotFoundError("Prepare the E2 split first")
    return expected


def e1_reference(cfg):
    e1_cfg = copy.deepcopy(cfg)
    e1_cfg["paths"]["data"] = str(ROOT / "data")
    e1 = load_splits(e1_cfg)
    e1_rows = read_csv(ROOT / "benchmarks/erm/predictions.csv")
    e1_metrics = {(r["fold"], int(r["seed"])): r for r in
                  read_csv(ROOT / "benchmarks/erm/metrics.csv")}
    if len(e1_rows) != 1560:
        raise ValueError("E1 predictions incomplete")
    for fold in DOMAINS:
        for seed in SEEDS:
            selected = [r for r in e1_rows if r["fold"] == fold and int(r["seed"]) == seed]
            if sorted(r["video_id"] for r in selected) != e1["folds"][fold]["target"]:
                raise ValueError(f"E1 coverage changed: {fold}/{seed}")
            actual = video_metrics(selected)
            saved = e1_metrics[fold, seed]
            for key in ("f1", "pr_auc", "precision", "recall", "tp", "fp", "tn", "fn"):
                if not math.isclose(actual[key], float(saved[key]), abs_tol=1e-12):
                    raise ValueError(f"E1 metric mismatch: {fold}/{seed}/{key}")
    return e1_rows


def horizon(fold, seed):
    root = ROOT / "research/results/runs" / f"{fold}_{seed}"
    provenance = json.loads((root / "provenance.json").read_text())
    if digest(root / "best_erm.pt") != provenance["erm_checkpoint_hash"]:
        raise ValueError(f"E1 checkpoint hash mismatch: {fold}/{seed}")
    trace = json.loads((root / "trace.json").read_text())
    if not trace or trace[-1]["epoch"] != len(trace):
        raise ValueError(f"Invalid E1 trace: {fold}/{seed}")
    return len(trace), int(trace[-1]["step"])


def preflight(cfg, check_cuda=True):
    import scripts.verify_benchmark as verifier
    verifier.main()
    artifact = split(cfg)
    rows = load_manifest(cfg)
    if len(rows) != 520 or len(e1_reference(cfg)) != 1560:
        raise ValueError("Unexpected E1 dataset size")
    pose_hash = hashlib.sha256(json.dumps(cfg["pose"], sort_keys=True).encode()).hexdigest()
    for row in rows:
        path = Path(cfg["paths"]["data"]) / "skeletons" / row["dataset"] / (row["video_id"] + ".npz")
        with np.load(path) as z:
            if (z["source_sha256"].item() != row["source_sha256"] or
                    z["pose_config_hash"].item() != pose_hash or z["xy"].shape[1:] != (17, 2) or
                    not np.all(np.diff(z["timestamp"]) > 0)):
                raise ValueError(f"Pose cache provenance invalid: {path}")
    for fold in DOMAINS:
        for seed in SEEDS:
            epochs, steps = horizon(fold, seed)
            source = artifact["folds"][fold][0]["source_train"]
            by_domain = defaultdict(int)
            lookup = {r["video_id"]: r for r in rows}
            for vid in source:
                by_domain[lookup[vid]["dataset"]] += 1
            expected = math.ceil(3 * max(by_domain.values()) / cfg["train"]["videos_per_batch"])
            if steps != epochs * expected:
                raise ValueError(f"E1 steps do not match expected budget: {fold}/{seed}")
    if check_cuda and (not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 1536 * 1024**2):
        raise RuntimeError("CUDA unavailable or less than 1536 MiB free")
    return {"protocol": PROTOCOL, "split_hash": artifact["split_hash"],
            "pose_caches": len(rows), "e1_predictions": 1560, "e2_runs": 48,
            "cuda": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def epoch_order(rows, partition, seed, epoch):
    """Exactly E1's 3*M video exposures and optimizer steps per epoch."""
    groups = defaultdict(list)
    for vid in partition["train"]:
        groups[rows[vid]["dataset"]].append(vid)
    source_domains = sorted({rows[vid]["dataset"] for vid in partition["source_train"]})
    if len(groups) != 4 or len(source_domains) != 3:
        raise ValueError("Expected three source domains and one target domain")
    total = 3 * max(len(groups[d]) for d in source_domains)
    rng = np.random.default_rng(seed + epoch)
    domains = sorted(groups)
    for domain in domains:
        groups[domain].sort(); rng.shuffle(groups[domain])
    positions = defaultdict(int)
    order = []
    for i in range(total):
        domain = domains[i % 4]
        pool = groups[domain]
        if positions[domain] and positions[domain] % len(pool) == 0:
            rng.shuffle(pool)
        order.append(pool[positions[domain] % len(pool)])
        positions[domain] += 1
    return order


def _train_epoch(model, optimizer, cfg, rows, order):
    model.train(); optimizer.zero_grad(set_to_none=True)
    size = cfg["train"]["videos_per_batch"]
    chunk = cfg["train"]["window_chunk"]
    device = cfg["train"]["device"]
    losses_by_domain = defaultdict(list)
    for start in range(0, len(order), size):
        ids = order[start:start + size]
        windows = [load_video(cfg, rows[vid])[0] for vid in ids]
        logits = video_batch_logits(model, windows, device, chunk)
        labels = torch.tensor([float(rows[vid]["label"]) for vid in ids], device=device)
        losses = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        if not torch.isfinite(losses).all():
            raise FloatingPointError("Nonfinite E2 training loss")
        for vid, loss in zip(ids, losses.detach().cpu().tolist()):
            losses_by_domain[rows[vid]["dataset"]].append(loss)
        losses.mean().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"], error_if_nonfinite=True)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
    return {d: float(np.mean(v)) for d, v in sorted(losses_by_domain.items())}


def run_dir(fold, part, seed):
    return paths()[1] / "runs" / f"{fold}_part{part}_seed{seed}"


def _identity(cfg, artifact, fold, part, seed):
    return {"protocol": PROTOCOL, "split_hash": artifact["split_hash"],
            "config_hash": canonical_hash(cfg), "code_hash": code_hash(),
            "fold": fold, "part": part, "seed": seed}


def verify_run(cfg, artifact, fold, part, seed):
    directory = run_dir(fold, part, seed)
    info = json.loads((directory / "provenance.json").read_text())
    identity = _identity(cfg, artifact, fold, part, seed)
    if any(info.get(k) != v for k, v in identity.items()):
        raise ValueError(f"E2 run identity changed: {directory}")
    if digest(directory / "best_erm.pt") != info["checkpoint_sha256"]:
        raise ValueError(f"E2 checkpoint changed: {directory}")
    ck = torch.load(directory / "best_erm.pt", map_location="cpu", weights_only=False)
    if any(ck.get(k) != v for k, v in identity.items()):
        raise ValueError(f"E2 checkpoint identity changed: {directory}")
    return info


def train_run(cfg, artifact, fold, part, seed, stop_after_epoch=None):
    partition = artifact["folds"][fold][part]
    rows = {r["video_id"]: r for r in load_manifest(cfg)}
    epochs, e1_steps = horizon(fold, seed)
    identity = _identity(cfg, artifact, fold, part, seed)
    directory = run_dir(fold, part, seed)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "provenance.json").exists():
        return verify_run(cfg, artifact, fold, part, seed)
    device = cfg["train"]["device"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("E2 requires CUDA; preflight first")
    set_seed(seed)
    model = CTRGCN(**cfg["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["learning_rate"],
                                  weight_decay=cfg["train"]["weight_decay"])
    last = directory / "last.pt"
    best, best_epoch, best_state, trace, first = float("inf"), 0, None, [], 0
    if last.exists():
        state = torch.load(last, map_location="cpu", weights_only=False)
        if any(state.get(k) != v for k, v in identity.items()) or state["epochs"] != epochs:
            raise ValueError(f"E2 resume identity changed: {directory}")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        best, best_epoch, best_state = state["best"], state["best_epoch"], state["best_state"]
        trace, first = state["trace"], state["epoch"]
        restore_rng(state["rng"])
    limit = min(epochs, stop_after_epoch) if stop_after_epoch else epochs
    for epoch in range(first, limit):
        started = time.monotonic()
        order = epoch_order(rows, partition, seed, epoch)
        train_losses = _train_epoch(model, optimizer, cfg, rows, order)
        recalibrate_bn(model, cfg, rows, partition["train"], device,
                       forbidden_ids=partition["val"] + partition["test"])
        val_loss, _ = evaluate_source(model, cfg, rows, partition["val"], device)
        if val_loss < best:
            best, best_epoch, best_state = val_loss, epoch + 1, copy.deepcopy(model.state_dict())
        counts = {d: sum(rows[vid]["dataset"] == d for vid in order) for d in DOMAINS}
        trace.append({"epoch": epoch + 1, "step": (epoch + 1) * math.ceil(len(order) / cfg["train"]["videos_per_batch"]),
                      "source_val_macro_bce": val_loss, "train_bce_by_domain": train_losses,
                      "samples_by_domain": counts, "seconds": time.monotonic() - started})
        state = {**identity, "epochs": epochs, "epoch": epoch + 1,
                 "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "best": best, "best_epoch": best_epoch, "best_state": best_state,
                 "trace": trace, "rng": rng_state()}
        temporary = directory / "last.tmp.pt"
        torch.save(state, temporary); temporary.replace(last)
        (directory / "trace.json").write_text(json.dumps(trace, indent=2) + "\n")
        print(f"{fold} part={part} seed={seed} epoch={epoch + 1}/{epochs} val_bce={val_loss:.5f} best={best:.5f}", flush=True)
    if limit < epochs:
        return {**identity, "incomplete": True, "epoch": limit}
    if not best_state:
        raise ValueError("E2 has no selected checkpoint")
    if trace[-1]["step"] != e1_steps:
        raise ValueError(f"E2 compute budget mismatch: {fold}/{seed}")
    model.load_state_dict(best_state)
    val_loss, val_preds = evaluate_source(model, cfg, rows, partition["val"], device)
    domains = {vid: rows[vid]["dataset"] for vid in partition["val"]}
    threshold = select_threshold(val_preds, cfg["evaluation"]["threshold_grid"], domains)
    ck = {**identity, "model": model.state_dict(), "threshold": threshold,
          "val_loss": val_loss, "best_epoch": best_epoch, "e1_epochs": epochs,
          "e1_steps": e1_steps, "config": cfg}
    temporary = directory / "best_erm.tmp.pt"
    torch.save(ck, temporary); temporary.replace(directory / "best_erm.pt")
    info = {**identity, "checkpoint_sha256": digest(directory / "best_erm.pt"),
            "best_epoch": best_epoch, "epochs": epochs, "steps": trace[-1]["step"],
            "threshold": threshold, "source_val_macro_bce": val_loss}
    (directory / "provenance.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def _paired_interval(a, b, groups, metric, seed, samples=1000):
    rng = np.random.default_rng(seed)
    members = defaultdict(list)
    for old, new in zip(a, b):
        members[groups[old["video_id"]]].append((old, new))
    keys = sorted(members)
    values = []
    for _ in range(samples):
        draw = [pair for key in rng.choice(keys, len(keys), replace=True) for pair in members[key]]
        old_rows, new_rows = zip(*draw)
        if metric == "pr_auc" and (len({r["y_true"] for r in old_rows}) < 2):
            continue
        values.append(video_metrics(new_rows)[metric] - video_metrics(old_rows)[metric])
    return (float(np.quantile(values, .025)), float(np.quantile(values, .975))) if values else (float("nan"), float("nan"))


def report(cfg, artifact):
    old_rows = e1_reference(cfg)
    old_map = {(r["fold"], int(r["seed"]), r["video_id"]): r for r in old_rows}
    rows = {r["video_id"]: r for r in load_manifest(cfg)}
    device = cfg["train"]["device"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable for E2 inference")
    # Validate all 48 completed runs before writing any report artifact.
    for fold in DOMAINS:
        for part in PARTS:
            for seed in SEEDS:
                verify_run(cfg, artifact, fold, part, seed)
    predictions = []
    for fold in DOMAINS:
        for part in PARTS:
            test_ids = artifact["folds"][fold][part]["test"]
            for seed in SEEDS:
                directory = run_dir(fold, part, seed)
                ck = torch.load(directory / "best_erm.pt", map_location="cpu", weights_only=False)
                checkpoint_sha256 = digest(directory / "best_erm.pt")
                model = CTRGCN(**cfg["model"]).to(device)
                model.load_state_dict(ck["model"]); model.eval()
                for vid in test_ids:
                    windows, _ = load_video(cfg, rows[vid])
                    with torch.no_grad():
                        logit = float(video_logit(model, windows, device, cfg["train"]["window_chunk"]))
                    probability = float(torch.sigmoid(torch.tensor(logit)))
                    key = (fold, seed, vid)
                    if key not in old_map or int(old_map[key]["y_true"]) != int(rows[vid]["label"]):
                        raise ValueError(f"E1/E2 pair mismatch: {key}")
                    predictions.append({"protocol": PROTOCOL, "fold": fold, "part": part,
                                        "seed": seed, "video_id": vid, "y_true": int(rows[vid]["label"]),
                                        "y_prob": probability, "threshold": ck["threshold"],
                                        "checkpoint_sha256": checkpoint_sha256,
                                        "split_hash": artifact["split_hash"]})
    new_map = {(r["fold"], r["seed"], r["video_id"]): r for r in predictions}
    if len(predictions) != 1560 or set(new_map) != set(old_map):
        raise ValueError("E2 does not cover every E1 prediction exactly once")
    metrics = []
    grouping = {vid: (row["subject_id"] if row["dataset"] in ("caucafall", "gmdcsa24")
                      else row["scene_id"] if row["dataset"] == "le2i"
                      else row["recording_group_id"]) for vid, row in rows.items()}
    for fold in DOMAINS:
        for seed in SEEDS:
            old = sorted((r for r in old_rows if r["fold"] == fold and int(r["seed"]) == seed), key=lambda r: r["video_id"])
            new = sorted((r for r in predictions if r["fold"] == fold and r["seed"] == seed), key=lambda r: r["video_id"])
            a, b = video_metrics(old), video_metrics(new)
            result = {"fold": fold, "seed": seed, "n": len(new)}
            for metric in ("f1", "pr_auc", "precision", "recall", "fp", "fn"):
                result.update({f"e1_{metric}": a[metric], f"e2_{metric}": b[metric],
                               f"delta_{metric}": b[metric] - a[metric]})
            for metric in ("f1", "pr_auc"):
                lo, hi = _paired_interval(old, new, grouping, metric, 2026 + seed)
                result[f"delta_{metric}_ci_low"] = lo
                result[f"delta_{metric}_ci_high"] = hi
            metrics.append(result)
    out = paths()[1]
    out.mkdir(parents=True, exist_ok=True)
    for name, values in (("predictions.csv", predictions), ("metrics.csv", metrics)):
        temporary = out / (name + ".tmp")
        with temporary.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)
        temporary.replace(out / name)
    lines = ["# E2 paired domain-exposure report", "", "E1 and E2 use the same 520 videos and three seeds. E2 predictions are out of fold.",
             "Training uses frozen E1 skeleton caches, source validation and an equal per-run optimizer budget.", "",
             "| Target | E1 Fall F1 | E2 Fall F1 | ΔF1 | E1 AP | E2 AP | ΔAP |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    keys = ("e1_f1", "e2_f1", "delta_f1", "e1_pr_auc", "e2_pr_auc", "delta_pr_auc")
    for fold in DOMAINS:
        subset = [r for r in metrics if r["fold"] == fold]
        values = [np.array([r[k] for r in subset]) for k in keys]
        lines.append(f"| {fold} | " + " | ".join(f"{v.mean():.3f} ± {v.std(ddof=1):.3f}" for v in values) + " |")
    per_seed = [{k: float(np.mean([r[k] for r in metrics if r["seed"] == seed])) for k in keys}
                for seed in SEEDS]
    values = [np.array([r[k] for r in per_seed]) for k in keys]
    lines.append("| Macro | " + " | ".join(f"{v.mean():.3f} ± {v.std(ddof=1):.3f}" for v in values) + " |")
    lines += ["", "Δ = E2 − E1. AP is average precision. Each domain has three paired seeds; per-seed metrics and grouped paired intervals are in metrics.csv.",
              "E2 uses four models per target domain and seed. Le2i scene groups and GMDCSA's four subjects make uncertainty estimates coarse.",
              "A gain measures the effect of exposure to other target-domain recordings under this protocol; it does not identify a specific cause of domain shift.", ""]
    (out / "REPORT.md").write_text("\n".join(lines))
    return {"predictions": len(predictions), "runs": 48, "report": str(out / "REPORT.md")}
