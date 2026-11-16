"""Local browser webcam app for real-time fall inference."""
from __future__ import annotations

import argparse
import os
import json
import tempfile
import threading
import uuid
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch

from .model import CTRGCN, mil_logit

ROOT = Path(__file__).resolve().parents[1]

PAGE = Path(__file__).with_name('live_page.html').read_text(encoding='utf-8')


class LiveEngine:
    def __init__(self, checkpoint: Path, device: str):
        from .pose import _choose_track, _inferencer
        self.device = torch.device(device)
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.cfg = ck["config"]
        self.threshold = float(ck.get("threshold", .5))
        self.model = CTRGCN(**self.cfg["model"])
        self.model.load_state_dict(ck["model"])
        self.model.to(self.device).eval()
        # Reuse the benchmark's RTMDet and RTMPose models and preprocessing.
        self.pose = _inferencer(self.cfg, device)
        self.choose_track = _choose_track
        self.buffers: dict[str, deque] = {}
        self.tracks: dict[str, tuple] = {}
        self.lock = threading.Lock()

    def analyze_video(self, path: Path, progress):
        import cv2
        from .data import make_windows, window_starts
        from .pose import extract_one

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError("Không đọc được video; hãy thử MP4, MOV, AVI hoặc WebM.")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if not np.isfinite(fps) or fps <= 0:
            fps = 30.0
        duration = frame_count / fps if frame_count > 0 else 0.0
        if duration > 900:
            raise ValueError("Video tối đa 15 phút để phân tích trên máy này.")
        row = {"source_path": str(path), "original_fps": fps, "frame_count": frame_count}
        with self.lock:
            pose = extract_one(self.cfg, row, self.pose, progress=progress)

        xy, score, valid = pose["xy"], pose["score"], pose["valid_joint"]
        width, height = pose["image_size"]
        scale = float(max(width, height))
        values = np.stack([(xy[..., 0] - width / 2) / scale,
                           (xy[..., 1] - height / 2) / scale, score], axis=-1).astype(np.float32)
        values[~valid] = 0
        windows, _ = make_windows(values, self.cfg["preprocess"]["window_frames"],
                                  self.cfg["preprocess"]["stride_frames"])
        progress(0.88, "Đang chấm điểm từng giai đoạn")
        logits = []
        chunk_size = int(self.cfg["train"].get("window_chunk", 16))
        with torch.inference_mode():
            for offset in range(0, len(windows), chunk_size):
                x = torch.from_numpy(windows[offset:offset + chunk_size]).to(self.device)
                logits.append(self.model(x).detach().cpu())
                progress(0.88 + 0.1 * min(offset + len(x), len(windows)) / len(windows),
                         f"Đang chấm điểm: {min(offset + len(x), len(windows))}/{len(windows)} giai đoạn")
        logits = torch.cat(logits)
        probabilities = torch.sigmoid(logits).tolist()
        starts = window_starts(len(values), self.cfg["preprocess"]["window_frames"],
                               self.cfg["preprocess"]["stride_frames"])
        stages = []
        for index, (start, probability) in enumerate(zip(starts, probabilities), 1):
            end = min(start + self.cfg["preprocess"]["window_frames"], len(values))
            center = min(start + self.cfg["preprocess"]["window_frames"] // 2, len(values) - 1)
            points = np.column_stack((xy[center], score[center])).astype(float)
            points[~valid[center]] = 0
            stages.append({
                "index": index,
                "start_sec": round(float(pose["timestamp"][start] - pose["timestamp"][0]), 2),
                "end_sec": round(float(pose["timestamp"][end - 1] - pose["timestamp"][0]), 2),
                "center_sec": round(float(pose["timestamp"][center] - pose["timestamp"][0]), 2),
                "probability": float(probability),
                "keypoints": points.tolist(),
            })
        overall = float(torch.sigmoid(mil_logit(logits)).item())
        progress(1.0, "Phân tích hoàn tất")
        return {"probability": overall, "threshold": self.threshold,
                "duration_sec": round(float(pose["timestamp"][-1] - pose["timestamp"][0]), 2),
                "sampled_frames": len(values), "stage_count": len(stages), "stages": stages}

    def infer(self, sid: str, encoded: bytes):
        import cv2
        image = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Cannot decode webcam frame")
        height, width = image.shape[:2]
        with self.lock:
            predictions = list(self.pose([image], return_vis=False,
                                         bbox_thr=self.cfg["pose"]["detector_score_threshold"]))
            instances = predictions[0].get("predictions", [[]])[0] if predictions else []
            previous, gap = self.tracks.get(sid, (None, 0))
            chosen = self.choose_track(instances, previous, self.cfg["pose"]["max_track_gap"], gap)
            keypoints = None
            if chosen:
                inst, box = chosen
                self.tracks[sid] = (box, 0)
                xy = np.asarray(inst["keypoints"], np.float32).reshape(17, 2)
                score = np.asarray(inst["keypoint_scores"], np.float32).reshape(17)
                keypoints = np.column_stack((xy, score)).tolist()
                good = np.isfinite(xy).all(-1) & np.isfinite(score) & (score >= self.cfg["pose"]["score_threshold"])
                scale = float(max(width, height))
                value = np.stack([(xy[:, 0] - width / 2) / scale,
                                  (xy[:, 1] - height / 2) / scale, score], axis=-1)
                value[~good] = 0
            else:
                self.tracks[sid] = (previous, gap + 1)
                value = np.zeros((17, 3), np.float32)
            buf = self.buffers.setdefault(sid, deque(maxlen=64))
            buf.append(value)
            probability = None
            if len(buf) == 64:
                clip = np.stack(buf).transpose(2, 0, 1)[None, ..., None].copy()
                with torch.inference_mode():
                    x = torch.from_numpy(clip).to(self.device)
                    probability = float(torch.sigmoid(mil_logit(self.model(x))).item())
            return {"frames": len(buf), "probability": probability,
                    "threshold": self.threshold, "keypoints": keypoints}

def serve(checkpoint: Path, host: str, port: int, device: str):
    engine = LiveEngine(checkpoint, device)
    jobs = {}
    jobs_lock = threading.Lock()

    def run_video_job(job_id, path):
        def update(value, message):
            with jobs_lock:
                jobs[job_id].update(progress=float(value), message=message)
        try:
            with jobs_lock:
                jobs[job_id].update(status="running", message="Đang chuẩn bị video", progress=0.01)
            result = engine.analyze_video(path, update)
            with jobs_lock:
                jobs[job_id].update(status="done", progress=1.0, message="Phân tích hoàn tất", result=result)
        except Exception as exc:
            with jobs_lock:
                jobs[job_id].update(status="error", message=str(exc))
        finally:
            path.unlink(missing_ok=True)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def send(self, status, payload, content="application/json"):
            data = payload.encode() if isinstance(payload, str) else json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", content + "; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/":
                self.send(200, PAGE, "text/html")
            elif self.path == "/status":
                self.send(200, {"threshold": engine.threshold})
            elif self.path.startswith("/analysis?"):
                from urllib.parse import parse_qs, urlparse
                job_id = parse_qs(urlparse(self.path).query).get("id", [""])[0]
                with jobs_lock:
                    job = jobs.get(job_id)
                    payload = dict(job) if job else None
                self.send(200 if payload else 404, payload or {"error": "Không tìm thấy tác vụ"})
            else:
                self.send(404, {"error": "Not found"})

        def do_POST(self):
            if self.path.startswith("/analyze?"):
                from urllib.parse import parse_qs, urlparse
                query = parse_qs(urlparse(self.path).query)
                suffix = Path(query.get("name", [""])[0]).suffix.lower()
                if suffix not in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpeg", ".mpg"}:
                    return self.send(400, {"error": "Định dạng video chưa được hỗ trợ."})
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                    if size <= 0 or size > 256 * 1024 * 1024:
                        raise ValueError("Video phải nhỏ hơn 256 MB.")
                    upload_dir = ROOT / ".uploads"
                    upload_dir.mkdir(exist_ok=True)
                    path = upload_dir / f"{uuid.uuid4().hex}{suffix}"
                    remaining = size
                    with path.open("wb") as target:
                        while remaining:
                            chunk = self.rfile.read(min(1024 * 1024, remaining))
                            if not chunk:
                                raise ValueError("Tệp video tải lên chưa hoàn tất.")
                            target.write(chunk)
                            remaining -= len(chunk)
                    job_id = uuid.uuid4().hex
                    with jobs_lock:
                        jobs[job_id] = {"status": "queued", "progress": 0.0, "message": "Đang chờ phân tích"}
                    threading.Thread(target=run_video_job, args=(job_id, path), daemon=True).start()
                    return self.send(202, {"id": job_id})
                except Exception as exc:
                    if "path" in locals():
                        path.unlink(missing_ok=True)
                    return self.send(400, {"error": str(exc)})
            if not self.path.startswith("/infer?"):
                return self.send(404, {"error": "Not found"})
            from urllib.parse import parse_qs, urlparse
            sid = parse_qs(urlparse(self.path).query).get("sid", ["default"])[0][:80]
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if size <= 0 or size > 3_000_000:
                    raise ValueError("Kích thước hình ảnh không hợp lệ")
                result = engine.infer(sid, self.rfile.read(size))
                self.send(200, result)
            except Exception as exc:
                self.send(500, {"error": str(exc)})

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Fallwatch running: http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nFallwatch stopped")
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description="Local real-time webcam fall detector")
    parser.add_argument("--checkpoint", default=str(ROOT / "best_erm.pt"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    serve(Path(args.checkpoint), args.host, args.port, args.device)


if __name__ == "__main__":
    main()
