"""Explicit pose-engine selection; portable exports never impersonate benchmark poses."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import requests

PORTABLE_ID = "portable-onnx-v1"
MODELS = {
    "detector": {
        "name": "YOLOX-tiny / HumanArt", "input_size": [416, 416],
        "url": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_tiny_8xb8-300e_humanart-6f3252f9.zip",
    },
    "pose": {
        "name": "RTMPose-m / body7 / COCO-17", "input_size": [192, 256],
        "url": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip",
    },
}


def portable_config(cfg):
    return {"engine": PORTABLE_ID, "benchmark_settings": cfg["pose"], "exports": MODELS}


def readiness(engine="auto"):
    if engine == "auto":
        engine = os.environ.get("FALL_POSE_ENGINE", "auto")
    from .baseline_runtime import runtime_status, weight_status
    original = runtime_status()
    weights = weight_status(Path(__file__).resolve().parents[1] / "data/model_weights")
    missing = original["missing"] + weights["missing"]
    portable_missing = [n for n in ("rtmlib", "onnxruntime") if importlib.util.find_spec(n) is None]
    options = [
        {"id": "mmpose", "name": "Original MMPose pipeline", "available": not missing, "missing": missing},
        {"id": "onnx", "name": "Portable CPU (ONNX)", "available": not portable_missing, "missing": portable_missing},
    ]
    if engine == "auto":
        engine = "mmpose"
    selected = next((o for o in options if o["id"] == engine), None)
    if selected is None:
        return {"available": False, "engine": engine, "missing": [], "options": options,
                "message": "Unknown pose engine. Choose mmpose or onnx."}
    available = selected["available"]
    if engine == "onnx":
        message = ("Portable CPU pose extraction is available. It uses YOLOX-tiny + RTMPose-m/body7; this differs from the benchmark extractor. Missing export weights download on first use."
                   if available else "Portable video analysis needs rtmlib and onnxruntime. Install requirements-portable.txt in this Python environment.")
    else:
        message = ("Original runtime, compiled NMS and copied baseline weights verified. RTMDet-m + RTMPose-m AIC/COCO."
                   if available else "Start start-baseline.cmd to use the isolated local baseline runtime. Setup needed: " + "; ".join(missing))
    return {"available": available, "engine": engine, "missing": selected["missing"],
            "options": options, "message": message}


def ensure_export(directory, key):
    """Download from the pinned official URL and retain a local byte fingerprint."""
    directory = Path(directory) / key
    directory.mkdir(parents=True, exist_ok=True)
    target, sidecar = directory / "model.onnx", directory / "model.sha256"
    if target.exists():
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if not sidecar.exists() or sidecar.read_text(encoding="ascii").strip() != actual:
            raise ValueError(f"Portable {key} fingerprint is missing or changed: {target}")
        return target, actual
    part = directory / "download.part"
    url = MODELS[key]["url"]
    print(f"Downloading portable {key} weights from {url}", flush=True)
    with requests.get(url, stream=True, timeout=(15, 60)) as response:
        response.raise_for_status()
        size = 0
        with part.open("wb") as out:
            for chunk in response.iter_content(1024**2):
                size += len(chunk)
                if size > 300 * 1024**2:
                    raise ValueError("Pose export archive exceeds its size limit.")
                out.write(chunk)
    # Read a single model entry; do not extract arbitrary archive paths.
    temporary = directory / "model.tmp"
    with zipfile.ZipFile(part) as archive:
        files = [f for f in archive.infolist() if f.filename.endswith("end2end.onnx")]
        if len(files) != 1 or files[0].file_size > 300*1024**2:
            raise ValueError("Expected one bounded end2end.onnx export in the model archive.")
        with archive.open(files[0]) as source, temporary.open("wb") as out:
            while chunk := source.read(1024**2):
                out.write(chunk)
    fingerprint = hashlib.sha256(temporary.read_bytes()).hexdigest()
    temporary.replace(target)
    sidecar.write_text(fingerprint + "\n", encoding="ascii")
    part.unlink()
    return target, fingerprint


class PortableInferencer:
    def __init__(self, cfg):
        from rtmlib import RTMPose, YOLOX
        directory = Path(cfg["paths"]["data"]) / "model_weights/portable"
        det, det_hash = ensure_export(directory, "detector")
        pose, pose_hash = ensure_export(directory, "pose")
        self.detector = YOLOX(str(det), model_input_size=tuple(MODELS["detector"]["input_size"]),
                              backend="onnxruntime", device="cpu")
        self.pose = RTMPose(str(pose), model_input_size=tuple(MODELS["pose"]["input_size"]),
                            backend="onnxruntime", device="cpu", to_openpose=False)
        self.provenance = {"engine": PORTABLE_ID, "benchmark_equivalent": False,
                           "models": {key: {**MODELS[key], "sha256": value} for key, value in
                                      (("detector", det_hash), ("pose", pose_hash))}}

    def __call__(self, frames, **kwargs):
        for frame in frames:
            boxes = self.detector(frame)
            instances = []
            # RTMPose's no-bbox API falls back to full-image inference; abstain
            # instead when the person detector sees nobody.
            if len(boxes):
                points, scores = self.pose(frame, bboxes=boxes)
                for box, xy, confidence in zip(boxes, points, scores):
                    if xy.shape != (17, 2):
                        raise ValueError("Portable export did not return COCO-17.")
                    instances.append({"bbox": np.asarray(box).tolist(), "bbox_score": 1.0,
                                      "keypoints": xy.tolist(), "keypoint_scores": confidence.tolist()})
            yield {"predictions": [instances]}


def make_inferencer(cfg, device, engine):
    status = readiness(engine)
    if not status["available"]:
        raise RuntimeError(status["message"])
    if status["engine"] == "onnx":
        return PortableInferencer(cfg)
    from .baseline_runtime import BaselineInferencer
    return BaselineInferencer(cfg, device)
