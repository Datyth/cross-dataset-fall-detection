"""One-video RGB inference with the frozen pose pipeline and CTR-GCN."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .data import digest, make_windows
from .model import COCO_EDGES, CTRGCN
from .pose import _inferencer, extract_one
from .train import video_logit


def _video_metadata(path: Path):
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot decode video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if not np.isfinite(fps) or fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
        raise ValueError(
            f"Invalid video metadata: fps={fps}, frames={frame_count}, size={width}x{height}"
        )
    return fps, frame_count, width, height


def _pose_windows(arrays, cfg):
    xy = arrays["xy"]
    score = arrays["score"]
    valid = arrays["valid_joint"]
    width, height = arrays["image_size"]
    scale = float(max(width, height))
    x = (xy[..., 0] - width / 2) / scale
    y = (xy[..., 1] - height / 2) / scale
    values = np.stack([x, y, score], axis=-1).astype(np.float32)
    values[~valid] = 0
    return make_windows(
        values,
        cfg["preprocess"]["window_frames"],
        cfg["preprocess"]["stride_frames"],
    )


def _render_overlay(video_path, output_path, arrays, result):
    import cv2

    output_path = Path(output_path).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot reopen video for rendering: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create annotated video: {output_path}")

    source_indices = np.asarray(arrays["source_indices"])
    available = np.flatnonzero(source_indices >= 0)
    available_source = source_indices[available]
    is_fall = result["prediction"] == "Fall"
    color = (30, 30, 235) if is_fall else (30, 190, 30)
    title = f"{'FALL' if is_fall else 'NON-FALL'}  {result['fall_probability'] * 100:.1f}%"
    subtitle = f"threshold {result['threshold']:.2f} | {result['method'].upper()} {result['fold']} seed {result['seed']}"

    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            pose_i = None
            if len(available):
                insertion = int(np.searchsorted(available_source, frame_index))
                candidates = []
                if insertion < len(available):
                    candidates.append(available[insertion])
                if insertion > 0:
                    candidates.append(available[insertion - 1])
                pose_i = min(candidates, key=lambda i: abs(int(source_indices[i]) - frame_index))

            if pose_i is not None:
                points = arrays["xy"][pose_i]
                visible = arrays["valid_joint"][pose_i]
                visible_points = points[visible]
                if len(visible_points) >= 2:
                    x0, y0 = visible_points.min(axis=0)
                    x1, y1 = visible_points.max(axis=0)
                    pad_x = max(20.0, (x1 - x0) * 0.18)
                    pad_y = max(20.0, (y1 - y0) * 0.12)
                    left = max(0, int(x0 - pad_x))
                    top = max(0, int(y0 - pad_y))
                    right = min(width - 1, int(x1 + pad_x))
                    bottom = min(height - 1, int(y1 + pad_y))
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 4, cv2.LINE_AA)
                    label_top = max(0, top - 44)
                    cv2.rectangle(frame, (left, label_top), (min(width - 1, left + 310), top), color, -1)
                    cv2.putText(frame, title, (left + 10, max(30, top - 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
                for a, b in COCO_EDGES:
                    if visible[a] and visible[b]:
                        cv2.line(frame, tuple(points[a].astype(int)), tuple(points[b].astype(int)),
                                 (0, 230, 255), 3, cv2.LINE_AA)
                for point, shown in zip(points, visible):
                    if shown:
                        cv2.circle(frame, tuple(point.astype(int)), 5, (255, 255, 255), -1, cv2.LINE_AA)
                        cv2.circle(frame, tuple(point.astype(int)), 5, color, 2, cv2.LINE_AA)

            cv2.rectangle(frame, (0, 0), (width, 72), (20, 20, 20), -1)
            cv2.putText(frame, title, (18, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                        color, 2, cv2.LINE_AA)
            cv2.putText(frame, subtitle, (18, 59), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (235, 235, 235), 1, cv2.LINE_AA)
            writer.write(frame)
            frame_index += 1
    finally:
        cap.release()
        writer.release()
    if frame_index == 0:
        output_path.unlink(missing_ok=True)
        raise ValueError("No frames were rendered")
    return output_path


def predict_video(cfg, video, checkpoint, device="cuda", output=None):
    """Return a JSON-serializable Fall/Non-Fall prediction for one RGB video."""
    video_path = Path(video).expanduser().resolve()
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video does not exist: {video_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but no CUDA device is available; use --device cpu")

    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required = {"model", "config", "threshold", "fold", "seed", "method"}
    missing = sorted(required - set(saved))
    if missing:
        raise ValueError(f"Checkpoint is missing required fields: {missing}")

    # The checkpoint configuration is authoritative for preprocessing and the
    # classifier. Keep only the current data path so pose weights can be reused.
    effective_cfg = saved["config"]
    effective_cfg["paths"]["data"] = cfg["paths"]["data"]
    fps, frame_count, width, height = _video_metadata(video_path)
    row = {
        "video_id": video_path.stem,
        "source_path": str(video_path),
        "source_sha256": digest(video_path),
        "original_fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
    }

    inferencer = _inferencer(effective_cfg, device)
    arrays = extract_one(effective_cfg, row, inferencer)
    windows, _ = _pose_windows(arrays, effective_cfg)

    model = CTRGCN(**effective_cfg["model"]).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    with torch.no_grad():
        logit = video_logit(
            model, windows, device, effective_cfg["train"]["window_chunk"]
        )
        probability = float(torch.sigmoid(logit))
    if not np.isfinite(probability):
        raise FloatingPointError("Model returned a non-finite probability")

    threshold = float(saved["threshold"])
    valid_fraction = float(np.mean(arrays["valid_frame"]))
    result = {
        "video": str(video_path),
        "checkpoint": str(checkpoint_path),
        "fold": saved["fold"],
        "seed": int(saved["seed"]),
        "method": saved["method"],
        "fall_probability": probability,
        "threshold": threshold,
        "prediction": "Fall" if probability >= threshold else "Non-Fall",
        "source_frames": frame_count,
        "sampled_pose_frames": int(len(arrays["timestamp"])),
        "valid_pose_fraction": valid_fraction,
        "windows": int(len(windows)),
    }
    if output is not None:
        result["annotated_video"] = str(
            _render_overlay(video_path, output, arrays, result)
        )
    print(json.dumps(result, indent=2))
    return result
