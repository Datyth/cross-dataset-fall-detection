"""Local checks for the frozen RTMDet/RTMPose environment and copied weights."""
from __future__ import annotations

import hashlib
import importlib.metadata
import platform
from functools import lru_cache
from pathlib import Path

VERSIONS = {"torch": "2.9.1", "torchvision": "0.24.1", "numpy": "1.26.4",
            "pandas": "2.3.3", "scikit-learn": "1.7.2", "scipy": "1.16.3",
            "opencv-python-headless": "4.10.0.84",
            "mmpose": "1.3.2", "mmdet": "3.3.0", "mmengine": "0.10.7", "mmcv": "2.1.0"}
WEIGHTS = {
    "rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth":
        "229f527ca88498e8894a778a62a878a322b4a3ea2cae09ea537d34b7e907792b",
    "rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth":
        "5e55be2a03f6e5dcd14d088afc4ae5afe94a4f9de93c22e5deb725ad0eee899d",
}


@lru_cache(maxsize=1)
def runtime_status():
    versions, errors = {}, []
    for package, expected in VERSIONS.items():
        try:
            actual = importlib.metadata.version(package)
            versions[package] = actual
            if actual.split("+")[0] != expected:
                errors.append(f"{package} {expected} required (found {actual})")
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"{package} {expected} missing")
    if platform.python_version_tuple()[:2] != ("3", "11"):
        errors.append(f"Python 3.11 required (found {platform.python_version()})")
    if not errors:
        try:
            import torch
            from mmcv.ops import nms
            from mmpose.apis import MMPoseInferencer  # noqa: F401
            _, kept = nms(torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.]]),
                          torch.tensor([.9, .8]), .5)
            if kept.tolist() != [0]:
                raise RuntimeError("Unexpected CPU NMS result")
        except Exception as exc:
            errors.append(f"Compiled MMCV/MMPose check failed: {exc}")
    return {"available": not errors, "missing": errors, "versions": versions,
            "python": platform.python_version(), "device_note":
            "CPU uses the original pipeline; floating-point results may differ from the training GPU."}


@lru_cache(maxsize=4)
def _weight_hash(path, size, mtime_ns):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def weight_status(directory):
    verified, errors = {}, []
    for name, expected in WEIGHTS.items():
        path = Path(directory) / name
        try:
            stat = path.stat()
            actual = _weight_hash(str(path.resolve()), stat.st_size, stat.st_mtime_ns)
            if actual != expected:
                raise ValueError("SHA-256 differs from imported baseline weights")
            verified[name] = actual
        except (OSError, ValueError) as exc:
            errors.append(f"{name}: {exc}")
    return {"available": not errors, "missing": errors, "weights_sha256": verified}


class BaselineInferencer:
    def __init__(self, cfg, device):
        status = runtime_status()
        weights = weight_status(Path(cfg["paths"]["data"]) / "model_weights")
        if not status["available"] or not weights["available"]:
            raise RuntimeError("; ".join(status["missing"] + weights["missing"]))
        from src.pose import _inferencer
        self.inferencer = _inferencer(cfg, device)
        self.provenance = {"engine": "mmpose", "pipeline_matches_baseline": True,
                           "device": device, "python": status["python"],
                           "versions": status["versions"], "weights_sha256": weights["weights_sha256"],
                           "numerical_note": status["device_note"]}

    def __call__(self, *args, **kwargs):
        return self.inferencer(*args, **kwargs)
