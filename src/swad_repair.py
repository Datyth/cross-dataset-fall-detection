"""Recover legacy SWAD without changing ERM or replaying the whole trajectory.

The legacy average is a linear combination of dense optimizer-state segments.
Only the converge segment was counted twice. Replay its short prefix, subtract
the surplus contribution in float64, then recalibrate BN/threshold on sources.
This retains the original FP32 average's rounding error; it is not claimed to
be bit-identical to a fresh chronological average. Replay mismatch fails closed.
"""
from __future__ import annotations

import copy
import importlib.metadata
import json
import shutil
from collections import Counter, deque
from pathlib import Path

import numpy as np
import torch

from .data import canonical_hash, digest, load_manifest, load_splits
from .train import (LEGACY_IMPLEMENTATION, SWAD_VERSION, LossValley, StateAverage,
                    _load_average, _model, _save_checkpoint, evaluate_source,
                    implementation_hash, recalibrate_bn, select_threshold,
                    set_seed, train_epoch, validate_completed_run,
                    verify_erm_implementation, verify_swad_summary)


def legacy_membership(trace, settings):
    """Replay the old selection/counting logic using epoch indices, no weights."""
    converge = deque(maxlen=settings["n_converge"])
    smooth = deque(maxlen=settings["n_tolerance"])
    selected, first, threshold, dead = [], None, None, False
    for i, row in enumerate(trace):
        if dead:
            break
        converge.append(i); smooth.append(i)
        if first is None:
            if len(converge) < converge.maxlen:
                continue
            losses = [trace[k]["source_val_macro_bce"] for k in converge]
            if int(np.argmin(losses)) != 0:
                continue
            first = converge[0]
            selected = [first]
            threshold = float(np.mean(losses) * (1 + settings["tolerance_ratio"]))
            if smooth.maxlen > converge.maxlen:
                earlier = list(smooth)[:smooth.maxlen - converge.maxlen + 1]
                start = 0
                for j in reversed(range(len(earlier))):
                    if trace[earlier[j]]["source_val_macro_bce"] > threshold:
                        start = j + 1
                        break
                selected.extend(earlier[start + 1:])
            continue
        if trace[smooth[0]]["step"] < trace[first]["step"]:
            continue
        if min(trace[k]["source_val_macro_bce"] for k in smooth) > threshold:
            dead = True
            break
        if trace[smooth[0]]["step"] > trace[selected[-1]]["step"]:
            selected.append(smooth[0])
    if first is not None and not dead:
        pending = list(smooth)[1:]
        for j, k in enumerate(pending):
            if min(trace[x]["source_val_macro_bce"] for x in pending[j:]) > threshold:
                break
            if trace[k]["step"] > trace[selected[-1]]["step"]:
                selected.append(k)
    return selected, first, threshold, dead


def recovery_plan(trace, settings):
    """Cross-check old membership against corrected production LossValley."""
    selected, first, threshold, dead = legacy_membership(trace, settings)
    if first is None:
        raise ValueError("Legacy run has no converged SWAD valley")
    counts = [r["step"] - (trace[i - 1]["step"] if i else 0) for i, r in enumerate(trace)]
    old = Counter(selected)
    valley = LossValley(**{k: settings[k] for k in ("n_converge", "n_tolerance", "tolerance_ratio")})
    for i, row in enumerate(trace):
        segment = StateAverage()
        basis = torch.zeros(len(trace), dtype=torch.float64); basis[i] = 1
        segment.add({"basis": basis}, counts[i], row["step"] - counts[i] + 1, row["step"])
        valley.update(segment, row["source_val_macro_bce"])
    corrected = valley.finish()
    summary = valley.summary(); verify_swad_summary(summary)
    contributions = corrected.state["basis"] * corrected.count
    expected = torch.tensor([counts[i] if i in old else 0 for i in range(len(trace))], dtype=torch.float64)
    if not torch.allclose(contributions, expected, rtol=0, atol=1e-8):
        raise ValueError("Correction changes valley membership; prefix-only repair is unsupported")
    removed = {i + 1: n - 1 for i, n in old.items() if n > 1}
    if any(n < 1 for n in counts) or not removed.keys() <= {first + 1}:
        raise ValueError("Unexpected legacy averaging pattern")
    legacy = {"valid": True, "converge_step": trace[first]["step"], "threshold": threshold,
              "dead_valley": dead, "start_step": trace[first]["step"] - counts[first] + 1,
              "end_step": trace[selected[-1]]["step"],
              "averaged_states": sum(counts[i] * n for i, n in old.items())}
    return {"legacy": legacy, "corrected": summary, "removed_epochs": removed,
            "epoch_counts": counts, "replay_until_epoch": max(removed, default=0)}


def correct_average(parameters, old_count, new_count, removed):
    """Subtract (segment mean, surplus dense-state count), retaining dtype."""
    if old_count - sum(count for _state, count in removed) != new_count or new_count < 1:
        raise ValueError("Invalid correction state counts")
    fixed = {}
    for name, value in parameters.items():
        if not removed:
            fixed[name] = value.detach().cpu().clone()
            continue
        total = value.detach().cpu().double() * old_count
        for state, count in removed:
            total.sub_(state[name].double(), alpha=count)
        fixed[name] = (total / new_count).to(value.dtype)
        if not torch.isfinite(fixed[name]).all():
            raise FloatingPointError(f"Non-finite repaired average: {name}")
    return fixed


def replay_prefix(cfg, fold, seed, trace, needed, environment):
    """Reproduce and verify source-only training through the duplicated epoch."""
    part = load_splits(cfg)["folds"][fold]
    rows = {r["video_id"]: r for r in load_manifest(cfg)}
    train_ids, val_ids, target_ids = part["train"], part["val"], part["target"]
    if (set(train_ids) & set(val_ids) or (set(train_ids) | set(val_ids)) & set(target_ids)
            or any(rows[v]["dataset"] == fold for v in train_ids + val_ids)):
        raise AssertionError("Target/source leakage in repair")
    device = cfg["train"]["device"]
    if importlib.metadata.version("torch") != environment["torch"]:
        raise ValueError("Prefix replay requires the original PyTorch version")
    if device.startswith("cuda"):
        if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != environment["gpu"]:
            raise ValueError("Prefix replay requires the original GPU type")
        if torch.version.cuda != environment["cuda_runtime"]:
            raise ValueError("Prefix replay requires the original CUDA runtime")
    torch.set_num_threads(environment["torch_num_threads"])
    set_seed(seed)
    model = _model(cfg, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["learning_rate"],
                                  weight_decay=cfg["train"]["weight_decay"])
    segments, checks, step = {}, [], 0
    for epoch in range(max(needed, default=0)):
        segment, step, train_bce = train_epoch(model, optimizer, cfg, rows, train_ids, seed, epoch, step)
        raw_bce, _ = evaluate_source(model, cfg, rows, val_ids, device)
        recalibrate_bn(model, cfg, rows, train_ids, device, forbidden_ids=val_ids + target_ids)
        val_bce, _ = evaluate_source(model, cfg, rows, val_ids, device)
        expected = trace[epoch]
        measured = (train_bce, raw_bce, val_bce)
        saved = (expected["source_train_macro_bce"], expected["source_val_bce_before_bn_calibration"],
                 expected["source_val_macro_bce"])
        difference = max(abs(x - y) for x, y in zip(measured, saved))
        if step != expected["step"] or not np.allclose(measured, saved, rtol=1e-5, atol=2e-6):
            raise RuntimeError(f"Prefix replay mismatch at {fold}/{seed}/epoch {epoch + 1}: {measured} != {saved}; no checkpoint replaced")
        checks.append({"epoch": epoch + 1, "step": step, "max_loss_difference": difference})
        if epoch + 1 in needed:
            segments[epoch + 1] = segment
        print(f"repair {fold} seed={seed} prefix_epoch={epoch + 1}/{max(needed)} replay_error={difference:.3g}", flush=True)
    return segments, checks


def _atomic_json(path, obj):
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(obj, indent=2) + "\n"); temporary.replace(path)


def repair_swad(cfg, fold, seed, resume=False, dry_run=False):
    directory = Path(cfg["paths"]["results"]) / "runs" / f"{fold}_{seed}"
    current = json.loads((directory / "provenance.json").read_text())
    if current["swad"].get("averaging_version") == SWAD_VERSION:
        result = validate_completed_run(cfg, fold, seed)
        if not resume and not dry_run:
            raise RuntimeError("SWAD already corrected; use --resume to verify and skip")
        return {"run": directory.name, "status": "verified-corrected", "swad": result["swad"]}
    backup = Path(cfg["paths"]["results"]) / "diagnostics" / "swad_v2" / directory.name
    original_info = backup / "provenance.json"
    info = json.loads(original_info.read_text()) if original_info.exists() else current
    if info.get("implementation_hash") != LEGACY_IMPLEMENTATION:
        raise ValueError("Only the fingerprinted legacy SWAD implementation is supported")
    verify_erm_implementation(info)
    if (info["config_hash"] != canonical_hash(cfg) or info["split_hash"] != load_splits(cfg)["split_hash"]
            or info["fold"] != fold or info["seed"] != seed):
        raise ValueError("Repair config/split/identity mismatch")
    erm_path, last_path = directory / "best_erm.pt", directory / "last.pt"
    erm_hash, last_hash = digest(erm_path), digest(last_path)
    if erm_hash != info["erm_checkpoint_hash"]:
        raise ValueError("Original ERM checksum mismatch")
    erm = torch.load(erm_path, map_location="cpu", weights_only=False)
    verify_erm_implementation(erm)
    if (erm["fold"] != fold or erm["seed"] != seed or erm["method"] != "erm"
            or erm["split_hash"] != info["split_hash"] or canonical_hash(erm["config"]) != canonical_hash(cfg)):
        raise ValueError("Original ERM lineage mismatch")
    trace = json.loads((directory / "trace.json").read_text())
    plan = recovery_plan(trace, cfg["swad"])
    for key, expected in plan["legacy"].items():
        if info["swad"].get(key) != expected:
            raise ValueError(f"Legacy trace/provenance mismatch: {key}")
    old_path = backup / "swad.pt" if original_info.exists() else directory / "swad.pt"
    if digest(old_path) != info["swad"]["checkpoint_hash"]:
        raise ValueError("Original SWAD checksum mismatch")
    old = torch.load(old_path, map_location="cpu", weights_only=False)
    if (old["fold"] != fold or old["seed"] != seed or old["method"] != "swad"
            or old["implementation_hash"] != LEGACY_IMPLEMENTATION
            or old["split_hash"] != info["split_hash"] or canonical_hash(old["config"]) != canonical_hash(cfg)):
        raise ValueError("Original SWAD lineage mismatch")
    if dry_run:
        return {"run": directory.name, "status": "repair-required",
                "plan": {k: v for k, v in plan.items() if k != "epoch_counts"}}
    backup.mkdir(parents=True, exist_ok=True)
    if not original_info.exists():
        shutil.copy2(directory / "swad.pt", backup / "swad.pt")
        shutil.copy2(directory / "provenance.json", original_info)
    cache = backup / "prefix_segments.pt"
    environment = json.loads((directory / "environment.json").read_text())
    # A cold resumed process must use the same calibration flags even when
    # its verified prefix is loaded from cache rather than replayed again.
    torch.set_num_threads(environment["torch_num_threads"])
    set_seed(seed)
    cache_key = canonical_hash({"implementation": implementation_hash(), "config": canonical_hash(cfg),
                                "fold": fold, "seed": seed, "trace_hash": digest(directory / "trace.json")})
    if cache.exists():
        replay = torch.load(cache, map_location="cpu", weights_only=False)
        if replay.get("cache_key") != cache_key:
            raise ValueError("Repair prefix cache changed; archive it before retrying")
        segments, checks = replay["segments"], replay["checks"]
    else:
        segments, checks = replay_prefix(cfg, fold, seed, trace, plan["removed_epochs"], environment)
        temporary = cache.with_suffix(".tmp.pt")
        torch.save({"cache_key": cache_key, "segments": segments, "checks": checks}, temporary)
        temporary.replace(cache)
    device = cfg["train"]["device"]
    model = _model(cfg, device)
    names = dict(model.named_parameters())
    removed = [(segments[e].state, copies * segments[e].count) for e, copies in plan["removed_epochs"].items()]
    params = correct_average({name: old["model"][name] for name in names},
                             plan["legacy"]["averaged_states"], plan["corrected"]["averaged_states"], removed)
    average = StateAverage(); average.state = params
    _load_average(model, average)
    rows = {r["video_id"]: r for r in load_manifest(cfg)}
    part = load_splits(cfg)["folds"][fold]
    summary = copy.deepcopy(plan["corrected"])
    summary["bn_recalibration_chunks"] = recalibrate_bn(model, cfg, rows, part["train"], device,
                                                        forbidden_ids=part["val"] + part["target"])
    loss, predictions = evaluate_source(model, cfg, rows, part["val"], device)
    threshold = select_threshold(predictions, cfg["evaluation"]["threshold_grid"],
                                 {v: rows[v]["dataset"] for v in part["val"]})
    temporary = directory / "swad.repair.tmp.pt"
    ck_hash = _save_checkpoint(temporary, model, cfg, info["split_hash"], seed, fold, "swad", threshold, loss, summary)
    if digest(erm_path) != erm_hash or digest(last_path) != last_hash:
        raise RuntimeError("ERM/last checkpoint changed during repair; no SWAD replacement")
    repair = {"averaging_version": SWAD_VERSION, "method": "float64_subtraction_of_replayed_duplicate_segments",
              "rounding": "retains legacy FP32 summation error; not bit-identical to a fresh average",
              "original_implementation_hash": LEGACY_IMPLEMENTATION,
              "implementation_hash": implementation_hash(), "config_hash": canonical_hash(cfg),
              "original_swad_hash": info["swad"]["checkpoint_hash"], "repaired_swad_hash": ck_hash,
              "erm_hash_unchanged": erm_hash, "last_hash_unchanged": last_hash,
              "replayed_epochs": len(checks), "prefix_checks": checks, "plan": plan,
              "source_val_bce": loss, "source_selected_threshold": threshold}
    summary["checkpoint_hash"] = ck_hash
    updated = copy.deepcopy(info); updated["swad"] = summary
    updated["swad_implementation_hash"] = implementation_hash()
    updated["swad_repair"] = {"record": str(directory / "swad_repair.json"),
                               "original_swad_hash": repair["original_swad_hash"]}
    temporary.replace(directory / "swad.pt")
    _atomic_json(directory / "swad_repair.json", repair)
    _atomic_json(directory / "provenance.json", updated)
    validate_completed_run(cfg, fold, seed)
    return {"run": directory.name, "status": "repaired", "replayed_epochs": len(checks),
            "source_val_bce": loss, "threshold": threshold, "swad": summary}
