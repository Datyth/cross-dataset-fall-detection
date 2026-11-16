"""Frozen RTMDet + RTMPose extraction; COCO-17, original pixel coordinates."""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import zipfile
import csv
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from .data import digest, download_file, load_manifest, resample_indices, urfd_timestamps


def _source_frames(row):
    import cv2
    path = Path(row["source_path"])
    if path.suffix.lower() == ".json":
        playlist = json.loads(path.read_text())
        last_t = -float("inf")
        for segment in playlist["segments"]:
            clip = path.parent / segment["path"]
            cap = cv2.VideoCapture(str(clip))
            if not cap.isOpened():
                raise ValueError(f"Undecodable playlist clip {clip}")
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            i = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                t = float(segment["start"]) + i / fps
                if t > last_t + .001:
                    yield t, frame
                    last_t = t
                i += 1
            cap.release()
    elif path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.lower().endswith(".png")]
            names.sort(key=lambda n: tuple(int(x) for x in re.findall(r"\d+", Path(n).stem)))
            timestamps = urfd_timestamps(path, len(names))
            for i, name in enumerate(names):
                frame = cv2.imdecode(np.frombuffer(z.read(name), np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError(f"Undecodable PNG {name}")
                yield float(timestamps[i]), frame
    else:
        cap = cv2.VideoCapture(str(path))
        fps = float(row["original_fps"])
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            t = ms / 1000 if ms > 0 else i / fps
            yield t, frame
            i += 1
        cap.release()


def _choose_track(instances, previous, max_gap, gap):
    """Deterministic confidence/overlap/center policy with retained gaps."""
    if not instances:
        return None
    candidates = []
    for idx, inst in enumerate(instances):
        box = np.asarray(inst["bbox"], dtype=float).reshape(-1)[:4]
        confidence = float(np.asarray(inst.get("bbox_score", 0)).reshape(-1)[0])
        if previous is None or gap > max_gap:
            rank = (confidence, -idx)
        else:
            x0 = max(box[0], previous[0]); y0 = max(box[1], previous[1])
            x1 = min(box[2], previous[2]); y1 = min(box[3], previous[3])
            inter = max(0, x1 - x0) * max(0, y1 - y0)
            area = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
            prev_area = max(0, previous[2] - previous[0]) * max(0, previous[3] - previous[1])
            iou = inter / max(area + prev_area - inter, 1)
            center = np.linalg.norm((box[:2] + box[2:]) / 2 - (previous[:2] + previous[2:]) / 2)
            scale = max(np.linalg.norm(previous[2:] - previous[:2]), 1)
            rank = (iou - 0.1 * center / scale, confidence, -idx)
        candidates.append((rank, idx, box))
    _, idx, box = max(candidates, key=lambda x: x[0])
    return instances[idx], box


def _inferencer(cfg, device):
    # MMEngine 0.10 calls torch.load without weights_only. PyTorch 2.9 now
    # defaults to True, which rejects NumPy metadata in these official files.
    # The two URLs are pinned above and their downloaded hashes are recorded.
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    try:
        from mmpose.apis import MMPoseInferencer
    except ImportError as e:
        raise RuntimeError("Pose extraction needs mmpose, mmdet, mmengine, mmcv in datpt; see README") from e
    def checkpoint(url):
        dest = Path(cfg["paths"]["data"]) / "model_weights" / Path(urlparse(url).path).name
        download_file(url, dest, resume=True)
        fingerprint = digest(dest)
        recorded = dest.with_suffix(dest.suffix + ".sha256")
        if recorded.exists() and recorded.read_text().strip() != fingerprint:
            raise ValueError(f"Pretrained checkpoint hash changed: {dest}")
        recorded.write_text(fingerprint + "\n")
        print("model checkpoint", dest, fingerprint)
        return str(dest)
    return MMPoseInferencer(
        pose2d=cfg["pose"]["pose_config"], pose2d_weights=checkpoint(cfg["pose"]["pose_checkpoint"]),
        det_model=cfg["pose"]["detector_config"], det_weights=checkpoint(cfg["pose"]["detector_checkpoint"]),
        det_cat_ids=0, device=device, show_progress=False)


def extract_one(cfg, row, inferencer, progress=None):
    fps = cfg["preprocess"]["fps"]
    tolerance = cfg["preprocess"]["timestamp_tolerance_s"]
    path = Path(row["source_path"])
    if progress:
        progress(0.03, "Đang đọc thông tin video")
    if path.suffix.lower() == ".zip":
        timestamps = urfd_timestamps(path, int(row["frame_count"]))
    else:
        # A metadata pass preserves actual container timestamps without keeping
        # decoded RGB frames in memory. The second pass performs inference.
        timestamps = np.array([t for t, _ in _source_frames(row)], dtype=np.float64)
    if len(timestamps) == 0:
        raise ValueError("No decodable frames")
    # Some containers report repeated/zero POS_MSEC. Use verified container FPS and log it.
    fallback = False
    if not np.all(np.diff(timestamps) > 0):
        timestamps = np.arange(len(timestamps), dtype=np.float64) / float(row["original_fps"])
        fallback = True
    grid, indices = resample_indices(timestamps, fps, tolerance)
    if progress:
        progress(0.08, "Đang lấy mẫu video theo baseline")
    n = len(grid)
    xy = np.zeros((n, 17, 2), np.float32)
    score = np.zeros((n, 17), np.float32)
    valid_joint = np.zeros((n, 17), bool)
    valid_frame = np.zeros(n, bool)
    previous, gap, ambiguous = None, 0, 0
    targets_by_source = {}
    for target_i, source_i in enumerate(indices):
        if source_i >= 0:
            targets_by_source.setdefault(int(source_i), []).append(target_i)
    image_size = None
    last_target = -1
    batch_frames, batch_positions = [], []
    processed = 0
    total_samples = int(np.count_nonzero(indices >= 0))

    def flush():
        nonlocal previous, gap, ambiguous, last_target, processed
        if not batch_frames:
            return
        outputs = list(inferencer(batch_frames, batch_size=cfg["pose"]["inference_batch"],
                                  return_vis=False, bbox_thr=cfg["pose"]["detector_score_threshold"]))
        if len(outputs) != len(batch_frames):
            raise ValueError(f"MMPose returned {len(outputs)} predictions for {len(batch_frames)} frames")
        for prediction, positions in zip(outputs, batch_positions):
            gap += positions[0] - last_target - 1
            last_target = positions[-1]
            instances = prediction.get("predictions", [[]])[0]
            if len(instances) > 1:
                ambiguous += len(positions)
            chosen = _choose_track(instances, previous, cfg["pose"]["max_track_gap"], gap)
            if chosen is None:
                gap += len(positions)
                continue
            inst, box = chosen
            points = np.asarray(inst["keypoints"], dtype=np.float32).reshape(-1, 2)
            confidence = np.asarray(inst["keypoint_scores"], dtype=np.float32).reshape(-1)
            if points.shape != (17, 2) or confidence.shape != (17,):
                raise ValueError(f"Expected COCO-17, got {points.shape}, {confidence.shape}")
            good = np.isfinite(points).all(-1) & np.isfinite(confidence) & (confidence >= cfg["pose"]["score_threshold"])
            for i in positions:
                xy[i] = np.nan_to_num(points)
                score[i] = np.nan_to_num(confidence)
                valid_joint[i] = good
                valid_frame[i] = bool(good.any())
            previous, gap = box, 0
        processed += len(batch_frames)
        if progress:
            progress(0.08 + 0.76 * processed / max(total_samples, 1),
                     f"Đang phân tích pose: {processed}/{total_samples} khung")
        batch_frames.clear(); batch_positions.clear()

    for source_i, (_, frame) in enumerate(_source_frames(row)):
        if image_size is None:
            image_size = np.array([frame.shape[1], frame.shape[0]], np.int32)
        positions = targets_by_source.get(source_i)
        if positions:
            batch_frames.append(frame)
            batch_positions.append(positions)
            if len(batch_frames) >= cfg["pose"]["inference_batch"]:
                flush()
    flush()
    if image_size is None:
        raise ValueError("No decoded image")
    if progress:
        progress(0.86, "Đang tạo các giai đoạn đánh giá")
    return dict(xy=xy, score=score, valid_joint=valid_joint, valid_frame=valid_frame,
                timestamp=grid, image_size=image_size,
                source_indices=indices, timestamp_fallback=np.array(fallback), ambiguous_frames=np.array(ambiguous))


def extract(cfg, dataset="all", device="cuda", limit=None, resume=False, force=False, video_id=None):
    if device.startswith("cuda"):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA extraction requested, but no CUDA device is available")
        free_mib = torch.cuda.mem_get_info()[0] // (1024 * 1024)
        if free_mib < 1536:
            raise RuntimeError(f"Only {free_mib} MiB GPU memory free; need at least 1536 MiB before loading frozen detector and pose models. Retry when the shared GPU is available, or use --device cpu.")
    rows = [r for r in load_manifest(cfg) if dataset == "all" or r["dataset"] == dataset]
    if video_id is not None:
        rows = [r for r in rows if r["video_id"] == video_id]
        if len(rows) != 1:
            raise ValueError(f"Expected one manifest row for --video-id {video_id}; found {len(rows)}")
    elif limit is not None:
        # Choose short, real recordings from both classes for a cheap smoke run.
        selected = []
        for domain in sorted({r["dataset"] for r in rows}):
            for cls in (1, 0):
                candidates = [r for r in rows if r["dataset"] == domain and int(r["label"]) == cls]
                match = min(candidates, key=lambda r: (float(r["duration_s"]), r["video_id"])) if candidates else None
                if match and match not in selected:
                    selected.append(match)
        rows = selected[:limit] if dataset != "all" else selected[:limit * len({r["dataset"] for r in rows})]
    inferencer = _inferencer(cfg, device)
    results, failures = [], []
    pose_hash = hashlib.sha256(json.dumps(cfg["pose"], sort_keys=True).encode()).hexdigest()
    for row in rows:
        dest = Path(cfg["paths"]["data"]) / "skeletons" / row["dataset"] / (row["video_id"] + ".npz")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and resume and not force:
            with np.load(dest) as z:
                if (z["xy"].shape[1:] == (17, 2) and np.all(np.diff(z["timestamp"]) > 0)
                    and z["source_sha256"].item() == row["source_sha256"]
                    and z["pose_config_hash"].item() == pose_hash):
                    print("verified-existing", dest)
                    continue
            raise RuntimeError(f"Invalid existing pose cache {dest}; refusing overwrite")
        if dest.exists() and not force:
            raise RuntimeError(f"Pose cache exists: {dest}; use --resume to verify and skip or --force to overwrite")
        try:
            arrays = extract_one(cfg, row, inferencer)
            arrays["source_sha256"] = np.array(row["source_sha256"])
            arrays["pose_config_hash"] = np.array(pose_hash)
            temp = dest.with_suffix(".tmp.npz")
            np.savez_compressed(temp, **arrays)
            temp.rename(dest)
            results.append((row["video_id"], len(arrays["timestamp"]), float(1 - arrays["valid_frame"].mean())))
            print("saved", dest, "frames", results[-1][1], "missing_fraction", results[-1][2])
        except Exception as e:
            print("FAILED", row["video_id"], e)
            results.append((row["video_id"], 0, 1.0))
            failures.append({"video_id": row["video_id"], "dataset": row["dataset"], "error": str(e)})
    if failures:
        path = Path(cfg["paths"]["results"]) / "extraction_failures.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=("video_id", "dataset", "error"))
            writer.writeheader(); writer.writerows(failures)
        raise RuntimeError(f"{len(failures)} pose extractions failed; see {path}")
    return results


def preview(cfg, dataset="all", limit=5):
    """Save deterministic three-frame pose overlays for manual coordinate checks."""
    import cv2
    from .model import COCO_EDGES
    roots = (sorted(cfg["datasets"]) if dataset == "all" else [dataset])
    saved = []
    for domain in roots:
        rows = [r for r in load_manifest(cfg) if r["dataset"] == domain]
        available = [r for r in rows if (Path(cfg["paths"]["data"]) / "skeletons" / domain / (r["video_id"] + ".npz")).exists()]
        if not available:
            continue
        # Evenly spaced IDs make repeated previews independent of file ordering.
        picks = np.unique(np.linspace(0, len(available) - 1, min(limit, len(available)), dtype=int))
        for pick in picks:
            row = available[int(pick)]
            cache = Path(cfg["paths"]["data"]) / "skeletons" / domain / (row["video_id"] + ".npz")
            with np.load(cache) as z:
                targets = np.unique(np.linspace(0, len(z["timestamp"]) - 1, 3, dtype=int))
                src = z["source_indices"][targets]
                xy = z["xy"][targets]
                good = z["valid_joint"][targets]
            source_to_sample = {int(s): i for i, s in enumerate(src) if s >= 0}
            frames = {}
            for j, (_, frame) in enumerate(_source_frames(row)):
                if j in source_to_sample:
                    frames[source_to_sample[j]] = frame
                if j >= max(source_to_sample, default=-1):
                    break
            tiles = []
            for i in range(len(targets)):
                frame = frames.get(i)
                if frame is None:
                    frame = np.zeros((240, 320, 3), np.uint8)
                else:
                    frame = frame.copy()
                    for a, b in COCO_EDGES:
                        if good[i, a] and good[i, b]:
                            cv2.line(frame, tuple(xy[i, a].astype(int)), tuple(xy[i, b].astype(int)), (0, 255, 255), 2)
                    for point, visible in zip(xy[i], good[i]):
                        if visible:
                            cv2.circle(frame, tuple(point.astype(int)), 3, (0, 0, 255), -1)
                cv2.putText(frame, f"{domain} {row['label']} frame {int(targets[i])}", (5, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
                tiles.append(cv2.resize(frame, (320, 240)))
            dest = Path(cfg["paths"]["results"]) / "previews" / domain / (row["video_id"] + ".png")
            dest.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dest), np.concatenate(tiles, axis=1))
            saved.append(str(dest))
    return saved
