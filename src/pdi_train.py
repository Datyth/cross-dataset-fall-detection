"""Source-only P3/P4 training, resume and one-fold paired evaluation."""
from __future__ import annotations

import copy
import csv
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from .data import canonical_hash, digest, load_manifest, load_splits, load_video
from .model import CTRGCN
from .pdi import Predictor, augment_past, losses, make_target, pair_candidates, update_ema
from .report import video_metrics
from .train import (evaluate_source, recalibrate_bn, restore_rng, rng_state, select_threshold,
                    set_seed, video_batch_logits, video_logit, _domain_balanced_order)

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("p3_future_aug", "p4_pdi")


def configs():
    base = yaml.safe_load((ROOT / "config.yaml").read_text())
    pdi = yaml.safe_load((ROOT / "configs/pdi.yaml").read_text())
    base["paths"]["data"] = str((ROOT / pdi["data_dir"]).resolve())
    base["train"]["device"] = pdi["device"]
    return base, pdi


def output_root(pdi):
    return ROOT / pdi["results_dir"]


def identity(base, pdi, fold, seed, method):
    source_hash = canonical_hash({name: digest(ROOT / name) for name in
                                  ("src/pdi.py", "src/pdi_train.py", "src/model.py", "src/data.py", "src/train.py")})
    return {"protocol": pdi["protocol"], "method": method, "fold": fold, "seed": seed,
            "source_hash": source_hash, "config_hash": canonical_hash({"base": base, "pdi": pdi}),
            "split_hash": load_splits(base)["split_hash"]}


def run_directory(pdi, method, fold, seed):
    return output_root(pdi) / method / fold / str(seed)


def audit(base, pdi):
    rows = load_manifest(base)
    counts = defaultdict(Counter)
    for row in rows:
        windows, masks = load_video(base, row)
        candidates = pair_candidates(windows, masks, pdi["min_past_valid_frames"],
                                     pdi["min_future_valid_frames"])
        key = (row["dataset"], "Fall" if int(row["label"]) else "Non-Fall")
        counts[key]["videos"] += 1
        counts[key]["windows"] += len(windows)
        counts[key]["valid_windows"] += len(candidates)
        counts[key]["valid_videos"] += bool(candidates)
    if sum(x["videos"] for x in counts.values()) != 520:
        raise ValueError("Expected all 520 audited videos")
    return {f"{domain}/{label}": dict(values) for (domain, label), values in sorted(counts.items())}


def _auxiliary(online, target, predictor, base, pdi, rows, ids, rng, method):
    """Backpropagate in small chunks after classification backward frees its graph."""
    prepared = []
    for vid in ids:
        windows, masks = load_video(base, rows[vid])
        eligible = pair_candidates(windows, masks, pdi["min_past_valid_frames"],
                                   pdi["min_future_valid_frames"])
        if eligible:
            x = windows[int(rng.choice(eligible))]
            prepared.append((x[:, :48], augment_past(x[:, :48], rng,
                              pdi["coordinate_noise_std"], pdi["joint_drop_probability"]), x[:, 48:64]))
    if not prepared:
        return 0, 0., 0., 0.
    device = base["train"]["device"]
    pred_values, pdi_values, embeddings = [], [], []
    for start in range(0, len(prepared), pdi["aux_batch_size"]):
        group = prepared[start:start + pdi["aux_batch_size"]]
        past, augmented, future = (torch.from_numpy(np.stack([item[j] for item in group])).to(device)
                                   for j in range(3))
        prediction, consistency, y_p = losses(online, target, predictor, past, augmented, future,
                                              method == "p4_pdi")
        weight = len(group) / len(prepared)
        (weight * (pdi["lambda_pred"] * prediction + pdi["lambda_pdi"] * consistency)).backward()
        pred_values.append((float(prediction.detach()), len(group)))
        pdi_values.append((float(consistency.detach()), len(group)))
        embeddings.append(y_p.cpu())
    all_features = torch.cat(embeddings)
    feature_std = float(all_features.std(dim=0).mean()) if len(all_features) > 1 else 0.
    return len(prepared), sum(v * n for v, n in pred_values) / len(prepared), sum(v * n for v, n in pdi_values) / len(prepared), feature_std


def verify_run(base, pdi, method, fold, seed):
    directory = run_directory(pdi, method, fold, seed)
    info = json.loads((directory / "provenance.json").read_text())
    wanted = identity(base, pdi, fold, seed, method)
    if any(info.get(k) != v for k, v in wanted.items()):
        raise ValueError(f"PDI run identity changed: {directory}")
    if digest(directory / "best.pt") != info["checkpoint_sha256"]:
        raise ValueError(f"PDI checkpoint changed: {directory}")
    return info


def train(base, pdi, method, fold, seed, stop_after_epoch=None):
    if method not in METHODS:
        raise ValueError(method)
    if pdi["aux_batch_size"] < 1 or pdi["max_epochs"] < 1:
        raise ValueError("Positive auxiliary batch size and epoch count required")
    splits = load_splits(base)
    partition = splits["folds"][fold]
    rows = {row["video_id"]: row for row in load_manifest(base)}
    if any(rows[vid]["dataset"] == fold for vid in partition["train"] + partition["val"]):
        raise AssertionError("PDI target-domain leakage")
    directory = run_directory(pdi, method, fold, seed)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "provenance.json").exists():
        return verify_run(base, pdi, method, fold, seed)
    device = base["train"]["device"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    set_seed(seed)
    rng = np.random.default_rng(seed)
    online = CTRGCN(**base["model"]).to(device)
    target = make_target(online)
    predictor = Predictor(base["model"]["base_channels"] * 4).to(device)
    optimizer = torch.optim.AdamW([*online.parameters(), *predictor.parameters()],
                                  lr=base["train"]["learning_rate"], weight_decay=base["train"]["weight_decay"])
    wanted = identity(base, pdi, fold, seed, method)
    last = directory / "last.pt"
    first, best, best_epoch, best_state, trace = 0, float("inf"), 0, None, []
    if last.exists():
        state = torch.load(last, map_location="cpu", weights_only=False)
        if any(state.get(k) != v for k, v in wanted.items()):
            raise ValueError(f"PDI resume mismatch: {directory}")
        online.load_state_dict(state["online"])
        target.load_state_dict(state["target"])
        predictor.load_state_dict(state["predictor"])
        optimizer.load_state_dict(state["optimizer"])
        rng.bit_generator.state = state["aux_rng"]
        restore_rng(state["rng"])
        first, best, best_epoch, best_state, trace = (state[k] for k in
                                                      ("epoch", "best", "best_epoch", "best_state", "trace"))
    limit = min(pdi["max_epochs"], stop_after_epoch) if stop_after_epoch else pdi["max_epochs"]
    train_ids = partition["train"]
    val_ids = partition["val"]
    for epoch in range(first, limit):
        began = time.monotonic()
        online.train(); predictor.train(); target.eval()
        order = list(_domain_balanced_order(rows, train_ids, seed + epoch))
        batch_size = base["train"]["videos_per_batch"]
        totals = defaultdict(list)
        optimizer.zero_grad(set_to_none=True)
        for start in range(0, len(order), batch_size):
            ids = order[start:start + batch_size]
            windows = [load_video(base, rows[vid])[0] for vid in ids]
            logits = video_batch_logits(online, windows, device, base["train"]["window_chunk"])
            labels = torch.tensor([float(rows[vid]["label"]) for vid in ids], device=device)
            classification = F.binary_cross_entropy_with_logits(logits, labels)
            if not torch.isfinite(classification):
                raise FloatingPointError("Nonfinite PDI classification loss")
            classification.backward()
            n, pred, consistency, feature_std = _auxiliary(online, target, predictor, base, pdi,
                                                            rows, ids, rng, method)
            torch.nn.utils.clip_grad_norm_([*online.parameters(), *predictor.parameters()],
                                            base["train"]["grad_clip"], error_if_nonfinite=True)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            update_ema(target, online, pdi["ema_decay"])
            for key, value in (("classification", float(classification.detach())), ("prediction", pred),
                               ("consistency", consistency), ("feature_std", feature_std),
                               ("aux_videos", n)):
                totals[key].append(value)
        recalibrate_bn(online, base, rows, train_ids, device, forbidden_ids=val_ids + partition["target"])
        val_loss, _ = evaluate_source(online, base, rows, val_ids, device)
        if val_loss < best:
            best, best_epoch, best_state = val_loss, epoch + 1, copy.deepcopy(online.state_dict())
        trace.append({"epoch": epoch + 1, "source_val_macro_bce": val_loss,
                      **{k: float(np.mean(v)) for k, v in totals.items()},
                      "seconds": time.monotonic() - began})
        state = {**wanted, "epoch": epoch + 1, "online": online.state_dict(),
                 "target": target.state_dict(), "predictor": predictor.state_dict(),
                 "optimizer": optimizer.state_dict(), "aux_rng": rng.bit_generator.state,
                 "rng": rng_state(), "best": best, "best_epoch": best_epoch,
                 "best_state": best_state, "trace": trace}
        temporary = directory / "last.tmp.pt"
        torch.save(state, temporary); temporary.replace(last)
        (directory / "trace.json").write_text(json.dumps(trace, indent=2) + "\n")
        print(f"{method} {fold} seed={seed} epoch={epoch+1}/{pdi['max_epochs']} val_bce={val_loss:.5f} aux_videos={sum(totals['aux_videos'])}", flush=True)
    if limit < pdi["max_epochs"]:
        return {**wanted, "incomplete": True, "epoch": limit}
    online.load_state_dict(best_state)
    val_loss, val_preds = evaluate_source(online, base, rows, val_ids, device)
    domains = {vid: rows[vid]["dataset"] for vid in val_ids}
    threshold = select_threshold(val_preds, base["evaluation"]["threshold_grid"], domains)
    ck = {**wanted, "model": online.state_dict(), "threshold": threshold,
          "best_epoch": best_epoch, "val_loss": val_loss, "base_config": base, "pdi_config": pdi}
    temporary = directory / "best.tmp.pt"
    torch.save(ck, temporary); temporary.replace(directory / "best.pt")
    info = {**wanted, "checkpoint_sha256": digest(directory / "best.pt"),
            "best_epoch": best_epoch, "epochs": len(trace), "threshold": threshold,
            "source_val_macro_bce": val_loss}
    (directory / "provenance.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def report(base, pdi, fold, seed):
    for method in METHODS:
        verify_run(base, pdi, method, fold, seed)
    rows = {row["video_id"]: row for row in load_manifest(base)}
    test_ids = load_splits(base)["folds"][fold]["target"]
    device = base["train"]["device"]
    all_predictions, summary = [], {}
    for method in METHODS:
        directory = run_directory(pdi, method, fold, seed)
        ck = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
        checkpoint_sha256 = digest(directory / "best.pt")
        model = CTRGCN(**base["model"]).to(device)
        model.load_state_dict(ck["model"]); model.eval()
        predictions = []
        for vid in test_ids:
            windows, _ = load_video(base, rows[vid])
            with torch.no_grad():
                logit = float(video_logit(model, windows, device, base["train"]["window_chunk"]))
            probability = float(torch.sigmoid(torch.tensor(logit)))
            predictions.append({"method": method, "fold": fold, "seed": seed, "video_id": vid,
                                "y_true": int(rows[vid]["label"]), "y_prob": probability,
                                "threshold": ck["threshold"], "checkpoint_sha256": checkpoint_sha256})
        summary[method] = video_metrics(predictions)
        all_predictions.extend(predictions)
    output = output_root(pdi) / "paired" / fold / str(seed)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "predictions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_predictions[0])); writer.writeheader(); writer.writerows(all_predictions)
    result = {"protocol": pdi["protocol"], "fold": fold, "seed": seed,
              "p3": summary[METHODS[0]], "p4": summary[METHODS[1]],
              "delta_f1": summary[METHODS[1]]["f1"] - summary[METHODS[0]]["f1"],
              "delta_ap": summary[METHODS[1]]["pr_auc"] - summary[METHODS[0]]["pr_auc"],
              "status": "exploratory single-fold single-seed; not a final claim"}
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# PDI pilot: CAUCAFall, seed 42" if fold == "caucafall" and seed == 42 else f"# PDI pilot: {fold}, seed {seed}",
             "", "Source-only LODO training; both methods use the same 30-epoch budget and source validation.",
             "", "| Method | Fall F1 | Average Precision | Precision | Recall | FP | FN |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        m = summary[method]
        lines.append(f"| {method} | {m['f1']:.3f} | {m['pr_auc']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | {m['fp']} | {m['fn']} |")
    lines += ["", f"P4−P3: ΔF1 = {result['delta_f1']:+.3f}; ΔAP = {result['delta_ap']:+.3f}.",
              "", "This one-fold, one-seed result is exploratory. It cannot establish a general PDI benefit.",
              "The auxiliary loss predicts a future segment representation during training; this is not an early-fall prediction evaluation.", ""]
    (output / "REPORT.md").write_text("\n".join(lines))
    return result
