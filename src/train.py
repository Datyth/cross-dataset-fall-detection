"""Video-level ERM training and source-validation LossValley SWAD.

LossValley logic adapted from Junbum Cha et al., SWAD (MIT license):
https://github.com/khanrc/swad/blob/main/domainbed/swad.py
Changes: segment snapshots represent dense per-step parameter averages; validation
is macro source-domain BCE; no target samples enter selection or BN recalibration.
"""
from __future__ import annotations

import copy
import csv
import importlib.metadata
import json
import platform
import random
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, average_precision_score

from .data import digest, load_manifest, load_splits, load_video, canonical_hash
from .model import CTRGCN, mil_logit

SWAD_VERSION = 2
LEGACY_IMPLEMENTATION = "ec05d92c6223f1e8d802797a1909c2916fc00c7c528b88b474d929a545cd7ed5"
LEGACY_COMPONENTS = {
    "src/model.py": "6fe5b88912cc08a1bcdc5cd91b3b7d4e8503f87153df50f0f96ec1d4500fe779",
    "src/data.py": "5da5c5109a3db4fb0c982a040b9d1b7a4d1195706e1a11486c500ae24fb9ca2c",
}


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def implementation_hash():
    root = Path(__file__).resolve().parents[1]
    return canonical_hash({name: digest(root / name) for name in
                           ("src/model.py", "src/train.py", "src/data.py", "src/swad_repair.py", "run.py")})


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def verify_implementation(saved, expected):
    if saved.get("implementation_hash") != expected:
        raise ValueError("Training implementation changed or is unversioned; archive the old run and restart")


def verify_erm_implementation(saved):
    """The known old trajectory remains valid; only its SWAD averaging was wrong."""
    if saved.get("implementation_hash") == implementation_hash():
        return
    root = Path(__file__).resolve().parents[1]
    if (saved.get("implementation_hash") == LEGACY_IMPLEMENTATION
            and all(digest(root / name) == value for name, value in LEGACY_COMPONENTS.items())):
        return
    raise ValueError("Unknown or incompatible ERM implementation")


def verify_swad_summary(summary):
    if summary.get("averaging_version") != SWAD_VERSION:
        raise ValueError("Legacy SWAD averaging is uncorrected; run: python run.py repair-swad --resume")
    if summary["valid"]:
        span = summary["end_step"] - summary["start_step"] + 1
        if summary["averaged_states"] != span or span < 1:
            raise ValueError("SWAD averaging range/count mismatch")


def validate_completed_run(cfg, fold, seed):
    """Validate lineage, hashes and averaging before resume or target inference."""
    directory = Path(cfg["paths"]["results"]) / "runs" / f"{fold}_{seed}"
    info = json.loads((directory / "provenance.json").read_text())
    split_hash = load_splits(cfg)["split_hash"]
    if (info["config_hash"] != canonical_hash(cfg) or info["split_hash"] != split_hash
            or info["fold"] != fold or info["seed"] != seed):
        raise ValueError(f"Completed run provenance changed: {directory}")
    verify_erm_implementation(info)
    for method, filename, expected in (("erm", "best_erm.pt", info["erm_checkpoint_hash"]),
                                        ("swad", "swad.pt", info["swad"].get("checkpoint_hash"))):
        if method == "swad" and not info["swad"]["valid"]:
            continue
        path = directory / filename
        if digest(path) != expected:
            raise ValueError(f"Checkpoint hash mismatch: {path}")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if (ck["method"] != method or ck["fold"] != fold or ck["seed"] != seed
                or ck["split_hash"] != split_hash or canonical_hash(ck["config"]) != canonical_hash(cfg)):
            raise ValueError(f"Checkpoint lineage mismatch: {path}")
        if method == "erm":
            verify_erm_implementation(ck)
        else:
            verify_swad_summary(info["swad"])
            verify_swad_summary(ck["extra"])
            verify_implementation(ck, implementation_hash())
    return info


def source_metrics(predictions, rows):
    by_domain = defaultdict(list)
    for vid, label, probability in predictions:
        by_domain[rows[vid]["dataset"]].append((label, probability))
    result = {}
    for domain, values in sorted(by_domain.items()):
        y = np.array([x[0] for x in values])
        p = np.array([x[1] for x in values])
        majority = int(y.mean() >= .5)
        result[domain] = {"n": len(values), "f1_at_0_5": float(f1_score(y, p >= .5, zero_division=0)),
                          "pr_auc": float(average_precision_score(y, p)),
                          "majority_f1": float(f1_score(y, np.full_like(y, majority), zero_division=0)),
                          "accuracy_at_0_5": float(np.mean(y == (p >= .5))),
                          "majority_accuracy": float(np.mean(y == majority)),
                          "probability_min": float(p.min()), "probability_max": float(p.max())}
    return result


def _parameter_state(model):
    return {k: p.detach().float().cpu().clone() for k, p in model.named_parameters()}


class StateAverage:
    def __init__(self):
        self.state = None
        self.count = 0
        self.start_step = None
        self.end_step = None

    def add(self, state, count=1, start_step=None, end_step=None):
        if count < 1:
            raise ValueError("Averaging count must be positive")
        if start_step is not None and end_step is not None and end_step - start_step + 1 != count:
            raise ValueError("Dense averaging count does not match optimizer-step range")
        if self.end_step is not None and start_step is not None and start_step != self.end_step + 1:
            raise ValueError("Overlapping or noncontiguous averaging segments")
        state = state.state if isinstance(state, StateAverage) else state
        if self.state is None:
            self.state = {k: v.clone() for k, v in state.items()}
            self.count = count
            self.start_step = start_step
            self.end_step = end_step
            return
        new_count = self.count + count
        for k in self.state:
            self.state[k].add_((state[k] - self.state[k]) * (count / new_count))
        self.count = new_count
        self.end_step = end_step


class LossValley:
    """Official converge/smooth queues, using dense optimizer-step segments."""
    def __init__(self, n_converge=3, n_tolerance=6, tolerance_ratio=0.3):
        if not 1 <= n_converge <= n_tolerance or tolerance_ratio < 0:
            raise ValueError("Require 1 <= n_converge <= n_tolerance and nonnegative tolerance_ratio")
        self.n_converge = n_converge
        self.n_tolerance = n_tolerance
        self.tolerance_ratio = tolerance_ratio
        self.converge_q = deque(maxlen=n_converge)
        self.smooth_q = deque(maxlen=n_tolerance)
        self.converge_step = None
        self.threshold = None
        self.final = None
        self.dead = False

    def update(self, segment, loss):
        if not np.isfinite(loss):
            raise FloatingPointError("Non-finite SWAD validation loss")
        if self.dead:
            return
        item = (copy.deepcopy(segment), float(loss))
        self.converge_q.append(item)
        self.smooth_q.append(item)
        if self.converge_step is None:
            if len(self.converge_q) < self.n_converge:
                return
            losses = [x[1] for x in self.converge_q]
            if int(np.argmin(losses)) != 0:
                return
            first = self.converge_q[0][0]
            self.converge_step = first.end_step
            candidates = [first]
            self.threshold = float(np.mean(losses) * (1 + self.tolerance_ratio))
            if self.n_tolerance > self.n_converge:
                earlier = list(self.smooth_q)[:self.n_tolerance - self.n_converge + 1]
                start = 0
                for i in reversed(range(len(earlier))):
                    if earlier[i][1] > self.threshold:
                        start = i + 1
                        break
                for extra, _ in earlier[start + 1:]:
                    candidates.append(extra)
            # The converge segment is also in `earlier`. Count it only once,
            # and include eligible preceding segments in chronological order.
            unique = {(s.start_step, s.end_step): s for s in candidates}
            self.final = StateAverage()
            for key in sorted(unique):
                s = unique[key]
                self.final.add(s, s.count, s.start_step, s.end_step)
            return
        if self.smooth_q[0][0].end_step < self.converge_step:
            return
        if min(x[1] for x in self.smooth_q) > self.threshold:
            self.dead = True
            return
        old, _ = self.smooth_q[0]
        if old.end_step > self.final.end_step:
            self.final.add(old, old.count, old.start_step, old.end_step)

    def finish(self):
        if self.converge_step is None:
            return None
        if not self.dead:
            pending = list(self.smooth_q)[1:]
            for i, (segment, _) in enumerate(pending):
                if min(x[1] for x in pending[i:]) > self.threshold:
                    break
                if segment.end_step > self.final.end_step:
                    self.final.add(segment, segment.count, segment.start_step, segment.end_step)
        return self.final

    def summary(self):
        return {"averaging_version": SWAD_VERSION,
                "valid": self.converge_step is not None, "converge_step": self.converge_step,
                "threshold": self.threshold, "dead_valley": self.dead,
                "start_step": self.final.start_step if self.final else None,
                "end_step": self.final.end_step if self.final else None,
                "averaged_states": self.final.count if self.final else 0}


def _model(cfg, device):
    return CTRGCN(**cfg["model"]).to(device)


def video_logit(model, windows, device, chunk=2):
    parts = []
    for i in range(0, len(windows), chunk):
        x = torch.from_numpy(windows[i:i + chunk]).to(device)
        parts.append(model(x))
    return mil_logit(torch.cat(parts))


def video_batch_logits(model, videos, device, chunk):
    """Mix recordings within forward batches, then apply MIL per video.

    Interleave their windows so small forward chunks contain different videos.
    Every window is forwarded exactly once; labels are applied only after each
    video's top-two aggregation, never to individual windows.
    """
    if not videos or chunk < 1:
        raise ValueError("Nonempty videos and positive window chunk required")
    positions = [(v, w) for w in range(max(map(len, videos)))
                 for v in range(len(videos)) if w < len(videos[v])]
    parts = [[] for _ in videos]
    for start in range(0, len(positions), chunk):
        selected = positions[start:start + chunk]
        x = np.stack([videos[v][w] for v, w in selected])
        logits = model(torch.from_numpy(x).to(device))
        for i, (v, _w) in enumerate(selected):
            parts[v].append(logits[i])
    return torch.stack([mil_logit(torch.stack(p)) for p in parts])


@torch.no_grad()
def evaluate_source(model, cfg, rows, ids, device):
    model.eval()
    by_domain = defaultdict(list)
    predictions = []
    for vid in ids:
        row = rows[vid]
        windows, _ = load_video(cfg, row)
        logit = video_logit(model, windows, device, cfg["train"]["window_chunk"])
        if not torch.isfinite(logit):
            raise FloatingPointError(f"Non-finite source-validation logit for {vid}")
        prob = float(torch.sigmoid(logit))
        label = int(row["label"])
        loss = float(F.binary_cross_entropy_with_logits(logit, torch.tensor(float(label), device=device)))
        by_domain[row["dataset"]].append(loss)
        predictions.append((vid, label, prob))
    macro_loss = float(np.mean([np.mean(x) for x in by_domain.values()]))
    return macro_loss, predictions


def select_threshold(predictions, grid, domain_by_video=None):
    def score(threshold):
        if domain_by_video is None:
            return f1_score([x[1] for x in predictions],
                            [x[2] >= threshold for x in predictions], zero_division=0)
        domains = sorted({domain_by_video[x[0]] for x in predictions})
        return float(np.mean([f1_score([x[1] for x in predictions if domain_by_video[x[0]] == domain],
                                       [x[2] >= threshold for x in predictions if domain_by_video[x[0]] == domain],
                                       zero_division=0) for domain in domains]))
    return min(grid, key=lambda t: (-score(t), abs(t - .5), t))


def _domain_balanced_order(rows, ids, seed):
    rng = np.random.default_rng(seed)
    by_domain = defaultdict(list)
    for vid in ids:
        by_domain[rows[vid]["dataset"]].append(vid)
    domains = sorted(by_domain)
    for domain in domains:
        rng.shuffle(by_domain[domain])
    steps = max(len(x) for x in by_domain.values())
    for i in range(steps):
        for domain in domains:
            yield by_domain[domain][i % len(by_domain[domain])]


@torch.no_grad()
def recalibrate_bn(model, cfg, rows, train_ids, device, forbidden_ids=()):
    """Recompute BN from shuffled, mixed-video SOURCE TRAIN windows only.

    One-video microbatches underestimate between-video variance. A larger,
    deterministic no-gradient calibration batch fixes inference statistics
    without changing the video-level training objective or the MIL windows.
    Dropout remains disabled. Apply identically before ERM validation and to
    the final SWAD weights, never to validation/target videos.
    """
    if set(train_ids) & set(forbidden_ids):
        raise AssertionError("Validation/target IDs supplied to BN recalibration")
    batch = cfg["train"]["bn_calibration_batch_size"]
    if batch < 2:
        raise ValueError("bn_calibration_batch_size must be at least 2")
    windows = np.concatenate([load_video(cfg, rows[vid])[0] for vid in sorted(train_ids)])
    order = np.random.default_rng(cfg["train"]["bn_calibration_seed"]).permutation(len(windows))
    bn = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    old = [m.momentum for m in bn]
    model.eval()
    for m in bn:
        m.reset_running_stats(); m.momentum = None
        m.train()
    count = 0
    try:
        for i in range(0, len(order), batch):
            model(torch.from_numpy(windows[order[i:i + batch]]).to(device))
            count += 1
    finally:
        for m, momentum in zip(bn, old):
            m.momentum = momentum
        model.eval()
    return count


def _load_average(model, average):
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(average.state[name].to(parameter.device, dtype=parameter.dtype))


def _save_checkpoint(path, model, cfg, split_hash, seed, fold, method, threshold, val_loss, extra):
    torch.save({"model": model.state_dict(), "config": cfg, "split_hash": split_hash,
                "implementation_hash": implementation_hash(),
                "seed": seed, "fold": fold, "method": method, "threshold": threshold,
                "val_loss": val_loss, "extra": extra}, path)
    return digest(path)


def train_epoch(model, optimizer, cfg, rows, train_ids, seed, epoch, step):
    """Shared, unchanged optimizer trajectory for normal training and prefix replay."""
    model.train(); optimizer.zero_grad(set_to_none=True)
    segment = StateAverage()
    order = list(_domain_balanced_order(rows, train_ids, seed + epoch))
    size = cfg["train"]["videos_per_batch"]
    batches = [order[i:i + size] for i in range(0, len(order), size)]
    accum = cfg["train"]["gradient_accumulation"]
    train_losses = defaultdict(list)
    device = cfg["train"]["device"]
    for i, ids in enumerate(batches):
        windows = [load_video(cfg, rows[vid])[0] for vid in ids]
        logits = video_batch_logits(model, windows, device, cfg["train"]["window_chunk"])
        labels = torch.tensor([float(rows[vid]["label"]) for vid in ids], device=device)
        losses = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        if not torch.isfinite(losses).all():
            raise FloatingPointError(f"Non-finite training loss: epoch={epoch + 1}, videos={ids}")
        for vid, loss in zip(ids, losses.detach().cpu().tolist()):
            train_losses[rows[vid]["dataset"]].append(loss)
        group_start = (i // accum) * accum
        group_videos = sum(len(b) for b in batches[group_start:group_start + accum])
        (losses.sum() / group_videos).backward()
        if (i + 1) % accum == 0 or i + 1 == len(batches):
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"], error_if_nonfinite=True)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            step += 1
            segment.add(_parameter_state(model), 1, step, step)
    bce = float(np.mean([np.mean(x) for x in train_losses.values()]))
    return segment, step, bce


def train_fold(cfg, fold, seed, smoke_test=False, resume=False, stop_after_epoch=None):
    if cfg["swad"]["validation_interval_epochs"] != 1:
        raise ValueError("This epoch-segment implementation requires validation_interval_epochs=1")
    if cfg["train"]["gradient_accumulation"] < 1:
        raise ValueError("gradient_accumulation must be positive")
    if stop_after_epoch is not None and stop_after_epoch < 1:
        raise ValueError("stop_after_epoch must be positive")
    splits = load_splits(cfg)
    partition = splits["folds"][fold]
    all_rows = {r["video_id"]: r for r in load_manifest(cfg)}
    train_ids, val_ids = partition["train"], partition["val"]
    videos_per_batch = cfg["train"]["videos_per_batch"]
    if videos_per_batch < 1:
        raise ValueError("videos_per_batch must be positive")
    if (set(train_ids) & set(val_ids) or set(train_ids) & set(partition["target"])
            or set(val_ids) & set(partition["target"])):
        raise AssertionError("Target leakage")
    if any(all_rows[vid]["dataset"] == fold for vid in train_ids + val_ids):
        raise AssertionError("Target domain supplied to source training/validation")
    for vid in train_ids + val_ids:
        if not (Path(cfg["paths"]["data"]) / "skeletons" / all_rows[vid]["dataset"] / (vid + ".npz")).exists():
            raise FileNotFoundError(f"Missing pose cache for {vid}")
    set_seed(seed)
    device = cfg["train"]["device"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.startswith("cuda"):
        free_mib = torch.cuda.mem_get_info()[0] // (1024 * 1024)
        if free_mib < 1536:
            raise RuntimeError(f"Only {free_mib} MiB GPU memory free; need at least 1536 MiB to start a reproducible training trajectory")
    code_hash = implementation_hash()
    model = _model(cfg, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["learning_rate"], weight_decay=cfg["train"]["weight_decay"])
    valley = LossValley(**{k: cfg["swad"][k] for k in ("n_converge", "n_tolerance", "tolerance_ratio")})
    output = Path(cfg["paths"]["results"]) / "runs" / f"{fold}_{seed}"
    config_hash = canonical_hash(cfg)
    start_epoch = 0
    last_path = output / "last.pt"
    resume_state = None
    if not smoke_test:
        output.mkdir(parents=True, exist_ok=True)
        if (output / "provenance.json").exists():
            if resume:
                return validate_completed_run(cfg, fold, seed)
            raise RuntimeError(f"Completed run already exists: {output}; use --resume to verify and skip")
        if (output / "best_erm.pt").exists() and not (resume and (output / "last.pt").exists()):
            raise RuntimeError(f"Partial completed checkpoint exists: {output}; inspect before rerun")
        if last_path.exists():
            if not resume:
                raise RuntimeError(f"Partial training run exists: {output}; use --resume")
            resume_state = torch.load(last_path, map_location="cpu", weights_only=False)
            verify_implementation(resume_state, code_hash)
            if (resume_state["config_hash"] != config_hash or resume_state["split_hash"] != splits["split_hash"]
                    or resume_state["seed"] != seed or resume_state["fold"] != fold):
                raise ValueError("Resume checkpoint provenance mismatch")
        (output / "config.yaml").write_text(__import__("yaml").safe_dump(cfg, sort_keys=False))
        versions = {}
        for name in ("torch", "numpy", "pandas", "scikit-learn", "scipy", "mmpose", "mmdet", "mmengine", "mmcv"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        versions.update({"python": platform.python_version(), "cuda_runtime": torch.version.cuda,
                         "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                         "cudnn_deterministic": True,
                         "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                         "torch_num_threads": torch.get_num_threads(),
                         "implementation_hash": code_hash})
        (output / "environment.json").write_text(json.dumps(versions, indent=2) + "\n")
    best, best_epoch, best_state, trace, step = float("inf"), -1, None, [], 0
    if resume_state is not None:
        state = resume_state
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        valley = state["valley"]
        best, best_epoch, best_state = state["best"], state["best_epoch"], state["best_state"]
        trace, step, start_epoch = state["trace"], state["step"], state["epoch"]
        restore_rng(state["rng_state"])
        print("Resuming", fold, seed, "at epoch", start_epoch, flush=True)
    max_epochs = 2 if smoke_test else cfg["train"]["max_epochs"]
    epoch_limit = min(max_epochs, stop_after_epoch) if stop_after_epoch is not None else max_epochs
    early_stopped = False
    accum = cfg["train"]["gradient_accumulation"]
    for epoch in range(start_epoch, epoch_limit):
        epoch_started = time.monotonic()
        segment, step, train_bce = train_epoch(model, optimizer, cfg, all_rows, train_ids, seed, epoch, step)
        raw_val_loss, _ = evaluate_source(model, cfg, all_rows, val_ids, device)
        calibration_chunks = recalibrate_bn(model, cfg, all_rows, train_ids, device,
                                             forbidden_ids=val_ids + partition["target"])
        val_loss, val_preds = evaluate_source(model, cfg, all_rows, val_ids, device)
        valley.update(segment, val_loss)
        record = {"epoch": epoch + 1, "step": step, "source_val_macro_bce": val_loss,
                  "source_train_macro_bce": train_bce, "source_val_bce_before_bn_calibration": raw_val_loss,
                  "source_val_by_domain": source_metrics(val_preds, all_rows),
                  "bn_calibration_chunks": calibration_chunks, "seconds": time.monotonic() - epoch_started,
                  "swad": valley.summary()}
        trace.append(record)
        print(f"{fold} seed={seed} epoch={epoch + 1} step={step} train_bce={train_bce:.5f} source_val_bce={val_loss:.5f} raw_val_bce={raw_val_loss:.5f} seconds={record['seconds']:.1f} swad={valley.summary()}", flush=True)
        if val_loss < best:
            best, best_epoch = val_loss, epoch + 1
            best_state = copy.deepcopy(model.state_dict())
        if not smoke_test:
            temporary = output / "last.tmp.pt"
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "valley": valley, "best": best, "best_epoch": best_epoch,
                        "best_state": best_state, "trace": trace, "step": step,
                        "epoch": epoch + 1, "seed": seed, "fold": fold,
                        "implementation_hash": code_hash, "rng_state": rng_state(),
                        "split_hash": splits["split_hash"], "config_hash": config_hash}, temporary)
            temporary.replace(last_path)
            (output / "trace.json").write_text(json.dumps(trace, indent=2) + "\n")
        if valley.dead and epoch + 1 - best_epoch >= cfg["train"]["patience_after_swad_convergence"]:
            early_stopped = True
            break
    if smoke_test:
        return {"smoke_test": True, "epochs": len(trace), "best_source_val_bce": best}
    if epoch_limit < max_epochs and not early_stopped:
        return {"incomplete": True, "checkpoint": str(last_path), "epochs": len(trace),
                "best_source_val_bce": best, "resume_to_continue": True}
    (output / "trace.json").write_text(json.dumps(trace, indent=2) + "\n")
    model.load_state_dict(best_state)
    erm_loss, erm_preds = evaluate_source(model, cfg, all_rows, val_ids, device)
    source_domains = {vid: row["dataset"] for vid, row in all_rows.items() if vid in val_ids}
    erm_threshold = select_threshold(erm_preds, cfg["evaluation"]["threshold_grid"], source_domains)
    erm_path = output / "best_erm.pt"
    erm_hash = _save_checkpoint(erm_path, model, cfg, splits["split_hash"], seed, fold, "erm", erm_threshold, erm_loss, {"best_epoch": best_epoch})
    average = valley.finish()
    summary = valley.summary()
    if average is not None:
        swad_model = _model(cfg, device)
        _load_average(swad_model, average)
        summary["bn_recalibration_chunks"] = recalibrate_bn(swad_model, cfg, all_rows, train_ids, device,
                                                            forbidden_ids=val_ids + partition["target"])
        swad_loss, swad_preds = evaluate_source(swad_model, cfg, all_rows, val_ids, device)
        swad_threshold = select_threshold(swad_preds, cfg["evaluation"]["threshold_grid"], source_domains)
        swad_path = output / "swad.pt"
        summary["checkpoint_hash"] = _save_checkpoint(swad_path, swad_model, cfg, splits["split_hash"], seed, fold, "swad", swad_threshold, swad_loss, summary)
    else:
        summary["reason"] = "No source-validation loss valley converged"
    provenance = {"fold": fold, "seed": seed, "split_hash": splits["split_hash"], "config_hash": config_hash,
                  "implementation_hash": code_hash,
                  "effective_video_batch": videos_per_batch * accum, "window_chunk": cfg["train"]["window_chunk"],
                  "bn_calibration": {"source": "source_train_only", "every_validation": True,
                                     "batch_size": cfg["train"]["bn_calibration_batch_size"],
                                     "seed": cfg["train"]["bn_calibration_seed"]},
                  "erm_checkpoint_hash": erm_hash, "swad": summary}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def synthetic_smoke(cfg, device="cpu"):
    """No data artifact or scientific result is produced by this check."""
    set_seed(42)
    model = _model(cfg, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    model.train()
    x = torch.randn(2, 3, 64, 17, 1, device=device)
    logits = model(x)
    loss = F.binary_cross_entropy_with_logits(mil_logit(logits), torch.tensor(1., device=device))
    loss.backward(); optimizer.step()
    model.eval()
    with torch.no_grad():
        score = float(torch.sigmoid(mil_logit(model(x))))
    return {"synthetic_smoke": True, "finite": bool(np.isfinite(score))}
