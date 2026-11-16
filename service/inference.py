"""Verified local checkpoints and recording-level inference."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from src.data import canonical_hash, load_splits, make_windows, window_starts
from src.model import CTRGCN, mil_logit
from src.train import verify_erm_implementation, verify_implementation, implementation_hash, verify_swad_summary
from .pose_runtime import PORTABLE_ID, portable_config, readiness

ROOT = Path(__file__).resolve().parents[1]
MAX_SECONDS = 600


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


class Cancelled(Exception):
    pass


class ModelRegistry:
    def __init__(self, root=ROOT, device="cpu"):
        self.root, self.device = Path(root), device
        self.entries, self.checkpoints, self.models = {}, {}, {}
        base = self.root / "results/runs/urfd_2026"
        snapshot = json.loads((self.root / "benchmarks/erm/provenance.json").read_text(encoding="utf-8"))
        expected = next(r for r in snapshot["runs"] if r["fold"] == "urfd" and r["seed"] == 2026)
        for method, filename in (("erm", "best_erm.pt"), ("swad", "swad.pt")):
            ident = f"urfd_2026_{method}"
            path = base / filename
            entry = dict(id=ident, name=f"{method.upper()} / URFD / 2026", method=method,
                         fold="urfd", seed=2026, path=str(path.relative_to(self.root)),
                         available=False, reason="Checkpoint file is missing.")
            try:
                if not path.is_file():
                    raise ValueError(entry["reason"])
                checkpoint = torch.load(path, map_location="cpu", weights_only=True)
                fingerprint = sha256(path)
                entry.update(sha256=fingerprint, threshold=float(checkpoint["threshold"]))
                if (checkpoint["fold"], checkpoint["seed"], checkpoint["method"]) != ("urfd", 2026, method):
                    raise ValueError("Checkpoint identity does not match its registered run.")
                if not math.isfinite(entry["threshold"]) or not 0 < entry["threshold"] < 1:
                    raise ValueError("Invalid saved threshold.")
                cfg = checkpoint["config"]
                split_cfg = {**cfg, "paths": {"data": str(self.root / "data")}}
                if checkpoint["split_hash"] != load_splits(split_cfg)["split_hash"]:
                    raise ValueError("Checkpoint and frozen split differ.")
                if method == "erm":
                    verify_erm_implementation(checkpoint)
                    if fingerprint != expected["checkpoint_sha256"]:
                        raise ValueError("ERM hash differs from the bundled URFD/2026 benchmark.")
                    if canonical_hash(cfg) != expected["config_hash"]:
                        raise ValueError("ERM configuration differs from benchmark provenance.")
                else:
                    verify_swad_summary(checkpoint["extra"])
                    verify_implementation(checkpoint, implementation_hash())
                entry.update(available=True, reason="Verified checkpoint", split_hash=checkpoint["split_hash"])
                self.checkpoints[ident] = checkpoint
            except Exception as exc:
                entry["reason"] = str(exc)
            self.entries[ident] = entry

    def public(self):
        return list(self.entries.values())

    def get(self, ident):
        if ident not in self.checkpoints:
            raise ValueError(self.entries.get(ident, {}).get("reason", "Unknown model."))
        return self.checkpoints[ident]

    def model(self, ident):
        if ident not in self.models:
            ck = self.get(ident)
            model = CTRGCN(**ck["config"]["model"])
            model.load_state_dict(ck["model"], strict=True)
            self.models[ident] = model.to(self.device).eval()
        return self.models[ident]


def pose_readiness(engine="auto"):
    return readiness(engine)


def validate_pose(arrays, cfg):
    required = ("xy", "score", "valid_joint", "image_size", "timestamp")
    if any(k not in arrays for k in required):
        raise ValueError("Pose cache needs xy, score, valid_joint, image_size and timestamp arrays.")
    xy = np.asarray(arrays["xy"], dtype=np.float32)
    score = np.asarray(arrays["score"], dtype=np.float32)
    valid = np.asarray(arrays["valid_joint"])
    timestamp = np.asarray(arrays["timestamp"], dtype=np.float64)
    size = np.asarray(arrays["image_size"])
    n = len(xy)
    if not 1 <= n <= MAX_SECONDS * cfg["preprocess"]["fps"] + 1:
        raise ValueError("Pose cache is empty or exceeds the 10-minute limit.")
    if xy.shape != (n, 17, 2) or score.shape != (n, 17) or valid.shape != (n, 17):
        raise ValueError("Expected COCO-17 arrays: xy [T,17,2], score/valid_joint [T,17].")
    if (size.shape != (2,) or not np.isfinite(size).all() or not (size > 0).all()
            or not (size <= 32768).all() or not np.equal(size, np.floor(size)).all()):
        raise ValueError("image_size must contain positive integer width and height, at most 32768.")
    if timestamp.shape != (n,) or not np.isfinite(timestamp).all() or not np.all(np.diff(timestamp) > 0):
        raise ValueError("Pose timestamps must be finite and strictly increasing.")
    if not np.isfinite(xy).all():
        raise ValueError("Invalid coordinates: pose coordinates contain NaN or infinity.")
    # SimCC keypoint scores are response peaks, not calibrated probabilities.
    # The frozen research loader preserves them as-is (including values > 1)
    # and zeroes invalid joints using the mask. Clipping would change inputs.
    if not np.isfinite(score).all():
        raise ValueError("Invalid joint response scores: pose scores contain NaN or infinity.")
    if not np.isin(valid, (False, True)).all():
        raise ValueError("valid_joint must be a boolean mask.")
    valid = valid.astype(bool)
    period = 1 / cfg["preprocess"]["fps"]
    if n > 1 and not np.allclose(np.diff(timestamp), period, rtol=0, atol=1e-4):
        raise ValueError("Pose cache must already be resampled to the model's 15 fps.")
    frames = valid.any(axis=1)
    if "valid_frame" in arrays and not np.array_equal(arrays["valid_frame"], frames):
        raise ValueError("valid_frame disagrees with valid_joint.")
    if "pose_config_hash" in arrays:
        backend = str(np.asarray(arrays.get("pose_backend", "mmpose")).item())
        if backend not in ("mmpose", PORTABLE_ID):
            raise ValueError("Unknown skeleton extraction backend.")
        settings = portable_config(cfg) if backend == PORTABLE_ID else cfg["pose"]
        expected = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
        if str(np.asarray(arrays["pose_config_hash"]).item()) != expected:
            raise ValueError("Pose cache was produced with a different extraction configuration.")
    return dict(xy=xy, score=score, valid_joint=valid, valid_frame=frames,
                image_size=size.astype(int), timestamp=timestamp)


def classify_pose(registry, model_id, arrays, cancelled=lambda: False, progress=lambda *_: None):
    ck = registry.get(model_id)
    cfg = ck["config"]
    pose = validate_pose(arrays, cfg)
    valid = pose["valid_joint"]
    coverage = float(pose["valid_frame"].mean())
    warnings = []
    backend = str(np.asarray(arrays.get("pose_backend", "mmpose" if "pose_config_hash" in arrays else "unverified-cache")).item())
    if backend == PORTABLE_ID:
        warnings.append("Portable ONNX pose engine: YOLOX-tiny + RTMPose-m/body7 differs from the benchmark extractor. Benchmark accuracy does not establish performance for these poses.")
    if "pose_config_hash" not in arrays:
        warnings.append("Imported skeleton cache has no extraction-config hash; its extractor identity is unverified.")
    if coverage < 1:
        warnings.append(f"No usable pose in {(1-coverage)*100:.1f}% of sampled frames. Review the skeleton overlay.")
    ambiguous = int(np.asarray(arrays.get("ambiguous_frames", 0)).item())
    if ambiguous:
        warnings.append(f"Multiple person candidates in {ambiguous} frames; only one track was classified.")
    if bool(np.asarray(arrays.get("timestamp_fallback", False)).item()):
        warnings.append("Container timestamps were non-monotonic; frame-rate timing was used.")
    base = dict(pose_coverage=coverage, joint_coverage=float(valid.mean()),
                observed_frames=len(valid), ambiguous_frames=ambiguous, warnings=warnings,
                model=registry.entries[model_id], preprocess=cfg["preprocess"], windows=[],
                pose_backend=backend)
    if "extractor_provenance" in arrays:
        base["extractor_provenance"] = json.loads(str(np.asarray(arrays["extractor_provenance"]).item()))
    if not pose["valid_frame"].any():
        return {**base, "decision": "Unable to analyze", "score": None,
                "threshold": ck["threshold"], "reason": "No usable person pose was found."}, pose
    width, height = pose["image_size"]
    xy = pose["xy"]
    values = np.stack(((xy[..., 0]-width/2)/max(width,height),
                       (xy[..., 1]-height/2)/max(width,height), pose["score"]), axis=-1).astype(np.float32)
    values[~valid] = 0
    size, stride = cfg["preprocess"]["window_frames"], cfg["preprocess"]["stride_frames"]
    windows, masks = make_windows(values, size, stride)
    model, logits = registry.model(model_id), []
    chunk = cfg["train"]["window_chunk"]
    with torch.inference_mode():
        for start in range(0, len(windows), chunk):
            if cancelled():
                raise Cancelled()
            output = model(torch.from_numpy(windows[start:start+chunk]).to(registry.device)).cpu()
            if not torch.isfinite(output).all():
                raise ValueError("The classifier returned a non-finite score.")
            logits.append(output)
            progress(min(start+chunk, len(windows)), len(windows))
        logits = torch.cat(logits)
        score = float(torch.sigmoid(mil_logit(logits)))
    selected = set(torch.topk(logits, min(2, len(logits))).indices.tolist())
    start_indices = window_starts(len(values), size, stride)
    records = []
    for i, start in enumerate(start_indices):
        end = min(start+size, len(values))-1
        records.append(dict(index=i, start=float(pose["timestamp"][start]-pose["timestamp"][0]),
                            end=float(pose["timestamp"][end]-pose["timestamp"][0]), logit=float(logits[i]),
                            score=float(torch.sigmoid(logits[i])), contributes=i in selected,
                            observed_frames=int(masks[i].sum())))
    if len(values) < size:
        warnings.append(f"Short recording: {size-len(values)} padding frames repeat the last visible pose.")
    return {**base, "decision": "Fall" if score >= ck["threshold"] else "Non-Fall",
            "score": score, "threshold": ck["threshold"], "windows": records}, pose


def extract_video(path, cfg, device, cancelled, progress, inferencer):
    """Wrap the frozen decoder/inferencer to add limits and cooperative cancellation."""
    import cv2
    from src.pose import extract_one
    cap = cv2.VideoCapture(str(path))
    try:
        ok, first = cap.read()
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not ok or not math.isfinite(fps) or fps <= 0:
            raise ValueError("Video cannot be decoded or has no usable frame rate.")
        if count > 0 and count/fps > MAX_SECONDS:
            raise ValueError("Videos must be 10 minutes or shorter.")
        height, width = first.shape[:2]
        expected = max(1, round(count/fps*cfg["preprocess"]["fps"]))
    finally:
        cap.release()

    # extract_one calls its module's decoder. Instead of mutating it globally,
    # make a function with an isolated globals mapping and the guarded decoder.
    import types
    original = extract_one.__globals__["_source_frames"]
    def frames(row):
        for i, (timestamp, frame) in enumerate(original(row)):
            if cancelled():
                raise Cancelled()
            if i > fps*MAX_SECONDS or timestamp > MAX_SECONDS:
                raise ValueError("Decoded recording exceeds the 10-minute limit.")
            if frame.shape[:2] != (height, width):
                raise ValueError("Variable-resolution videos are not supported.")
            yield timestamp, frame
    completed = 0
    def wrapped(*args, **kwargs):
        nonlocal completed
        for prediction in inferencer(*args, **kwargs):
            if cancelled():
                raise Cancelled()
            completed += 1
            progress(completed, expected)
            yield prediction
    function = types.FunctionType(extract_one.__code__, {**extract_one.__globals__, "_source_frames": frames})
    arrays = function(cfg, {"source_path": str(path), "original_fps": fps}, wrapped)
    arrays["pose_config_hash"] = np.array(hashlib.sha256(json.dumps(cfg["pose"], sort_keys=True).encode()).hexdigest())
    return arrays, dict(duration=count/fps if count > 0 else len(arrays["timestamp"])/15,
                        width=width, height=height, fps=fps)
