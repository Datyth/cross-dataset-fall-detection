"""Local API, persistent jobs and static UI. Run with serve.py."""
from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import os
import shutil
import sqlite3
import threading
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .inference import (ROOT, Cancelled, ModelRegistry, classify_pose, extract_video,
                        pose_readiness, sha256)
from .pose_runtime import PORTABLE_ID, make_inferencer, portable_config

ACTIVE = {"queued", "validating", "extracting_pose", "classifying", "preparing_result", "cancelling"}
EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


class JobStore:
    def __init__(self, path):
        self.path, self.lock = path, threading.RLock()
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, body TEXT NOT NULL)")
        for job in self.list():
            if job["status"] in ACTIVE:
                self.update(job["id"], status="failed", error="Analysis was interrupted by an app restart. Upload the recording again to retry.")

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def get(self, ident):
        with self.lock, self.connect() as db:
            row = db.execute("SELECT body FROM jobs WHERE id=?", (ident,)).fetchone()
            if row is None:
                raise KeyError(ident)
            return json.loads(row[0])

    def update(self, ident, **changes):
        with self.lock, self.connect() as db:
            row = db.execute("SELECT body FROM jobs WHERE id=?", (ident,)).fetchone()
            body = json.loads(row[0]) if row else {"id": ident}
            body.update(changes, updated_at=datetime.now(timezone.utc).isoformat())
            db.execute("INSERT OR REPLACE INTO jobs VALUES (?,?)", (ident, json.dumps(body, allow_nan=False)))
            return body

    def list(self):
        with self.lock, self.connect() as db:
            return [json.loads(row[0]) for row in db.execute("SELECT body FROM jobs ORDER BY rowid DESC")]

    def delete(self, ident):
        with self.lock, self.connect() as db:
            db.execute("DELETE FROM jobs WHERE id=?", (ident,))


class Manager:
    def __init__(self, root, storage, registry, pose_factory=None, baseline_only=True):
        self.root, self.storage, self.registry = Path(root), Path(storage).resolve(), registry
        self.storage.mkdir(parents=True, exist_ok=True)
        self.store = JobStore(self.storage / "jobs.sqlite3")
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fall-inference")
        self.cancel_events = {}
        self.pose_factory, self.inferencers = pose_factory, {}
        self.baseline_only = baseline_only

    def folder(self, ident):
        # IDs are server generated. Also enforce scope before removal/access.
        if str(uuid.UUID(ident)) != ident:
            raise ValueError("Invalid analysis ID.")
        folder = (self.storage / ident).resolve()
        if folder.parent != self.storage:
            raise ValueError("Invalid analysis path.")
        return folder

    def queue(self, ident):
        self.cancel_events[ident] = threading.Event()
        self.executor.submit(self.run, ident)

    def run(self, ident):
        started = time.monotonic()
        event = self.cancel_events[ident]
        def check():
            if event.is_set():
                raise Cancelled()
        def update(stage, done=None, total=None):
            check()
            self.store.update(ident, status=stage, progress={"done": done, "total": total})
        try:
            job = self.store.get(ident)
            folder = self.folder(ident)
            update("validating")
            cfg = copy.deepcopy(self.registry.get(job["model_id"])["config"])
            cfg["paths"]["data"] = str(self.root / "data")
            source = folder / job["source_file"]
            source_hash = sha256(source)
            metadata = {}
            if job["kind"] == "cache":
                with zipfile.ZipFile(source) as archive:
                    if sum(f.file_size for f in archive.infolist()) > 256 * 1024**2:
                        raise ValueError("Expanded skeleton cache exceeds the 256 MB limit.")
                with np.load(source, allow_pickle=False) as z:
                    arrays = {name: z[name] for name in z.files}
                if self.baseline_only:
                    if "pose_config_hash" not in arrays or str(np.asarray(arrays.get("pose_backend", "mmpose")).item()) != "mmpose":
                        raise ValueError("Baseline analysis requires an original MMPose cache with its extraction-config hash. Reanalyze the source video with the baseline runtime.")
            else:
                update("extracting_pose")
                engine = job.get("pose_engine", "mmpose")
                key = (job["model_id"], engine)
                if key not in self.inferencers:
                    if self.pose_factory:
                        self.inferencers[key] = self.pose_factory(cfg, self.registry.device)
                    else:
                        self.inferencers[key] = make_inferencer(cfg, self.registry.device, engine)
                arrays, metadata = extract_video(source, cfg, self.registry.device, event.is_set,
                    lambda d, t: update("extracting_pose", d, t), self.inferencers[key])
                if not self.pose_factory and engine == "onnx":
                    arrays["pose_backend"] = np.array(PORTABLE_ID)
                    arrays["pose_config_hash"] = np.array(hashlib.sha256(json.dumps(portable_config(cfg), sort_keys=True).encode()).hexdigest())
                    arrays["extractor_provenance"] = np.array(json.dumps(self.inferencers[key].provenance))
                elif not self.pose_factory:
                    arrays["pose_backend"] = np.array("mmpose")
                    arrays["extractor_provenance"] = np.array(json.dumps(self.inferencers[key].provenance))
                arrays["source_sha256"] = np.array(source_hash)
                np.savez_compressed(folder / "pose.npz", **arrays)
            update("classifying")
            result, pose = classify_pose(self.registry, job["model_id"], arrays, event.is_set,
                                        lambda d, t: update("classifying", d, t))
            update("preparing_result")
            result.update(source_sha256=source_hash, input_kind=job["kind"],
                          processing_seconds=round(time.monotonic()-started, 3))
            skeleton = dict(timestamp=(pose["timestamp"]-pose["timestamp"][0]).tolist(),
                            xy=pose["xy"].tolist(), valid=pose["valid_joint"].tolist(),
                            image_size=pose["image_size"].tolist())
            (folder / "skeleton.json").write_text(json.dumps(skeleton), encoding="utf-8")
            duration = float(pose["timestamp"][-1]-pose["timestamp"][0]) + 1/cfg["preprocess"]["fps"]
            metadata = {"duration": duration, **metadata}
            if job["kind"] == "video":
                # Preserve original timing. Native video controls expose codec errors.
                result["warnings"].append("If this video's codec cannot play in your browser, convert a copy to H.264 MP4 and analyze it again.")
            check()
            self.store.update(ident, status="completed", result=result, metadata=metadata, progress=None)
        except Cancelled:
            self.store.update(ident, status="cancelled", progress=None)
        except Exception as exc:
            self.store.update(ident, status="failed", error=str(exc), progress=None)

    def close(self):
        for event in self.cancel_events.values():
            event.set()
        self.executor.shutdown(wait=True, cancel_futures=True)


def create_app(root=ROOT, storage=None, registry=None, pose_factory=None, baseline_only=True):
    root = Path(root)
    storage = Path(storage or os.environ.get("FALL_STORAGE", root / "local_app"))
    device = os.environ.get("FALL_DEVICE", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Start with --device cpu.")
    torch.set_num_threads(4 if baseline_only else 2)
    registry = registry or ModelRegistry(root, device)
    manager = Manager(root, storage, registry, pose_factory, baseline_only)

    @asynccontextmanager
    async def lifespan(app):
        yield
        manager.close()

    app = FastAPI(title="Fall Review", lifespan=lifespan)
    app.state.manager = manager

    def get_job(ident):
        try:
            manager.folder(ident)
            return manager.store.get(ident)
        except (KeyError, ValueError):
            raise HTTPException(404, "Analysis not found.")

    @app.get("/api/readiness")
    def readiness():
        pose = pose_readiness("mmpose" if baseline_only else "auto")
        if baseline_only:
            pose["options"] = [o for o in pose["options"] if o["id"] == "mmpose"]
        models = registry.public()
        ready = any(m["available"] for m in models)
        return dict(models=models, device=device, classifier_ready=ready,
                    pose=pose, video_ready=ready and pose["available"], cache_ready=ready,
                    limits={"video_mb": 512, "cache_mb": 32, "duration_seconds": 600})

    @app.get("/api/models")
    def models():
        return registry.public()

    @app.get("/api/analyses")
    def analyses():
        return manager.store.list()

    @app.get("/api/analyses/{ident}")
    def analysis(ident: str):
        return get_job(ident)

    @app.post("/api/analyses", status_code=202)
    async def upload(file: UploadFile = File(...), model_id: str = Form("urfd_2026_erm"),
                     kind: str = Form("video"), pose_engine: str = Form("auto")):
        if kind not in ("video", "cache"):
            raise HTTPException(400, "Choose video or cache analysis.")
        try:
            registry.get(model_id)
        except ValueError as exc:
            raise HTTPException(409, str(exc))
        if kind == "video":
            if baseline_only and pose_engine not in ("auto", "mmpose"):
                raise HTTPException(409, "This app uses the original baseline MMPose pipeline. Portable ONNX analysis is disabled.")
            if baseline_only:
                pose_engine = "mmpose"
            status = pose_readiness(pose_engine)
            if not pose_factory and not status["available"]:
                raise HTTPException(409, status["message"])
            pose_engine = "mmpose" if pose_factory else status["engine"]
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in (EXTENSIONS if kind == "video" else {".npz"}):
            raise HTTPException(400, "Choose a supported video file." if kind == "video" else "Choose a .npz skeleton cache.")
        ident = str(uuid.uuid4())
        folder = manager.folder(ident)
        folder.mkdir()
        filename = "source" + suffix
        limit = (512 if kind == "video" else 32) * 1024**2
        try:
            size = 0
            with (folder / filename).open("wb") as handle:
                while chunk := await file.read(1024**2):
                    size += len(chunk)
                    if size > limit:
                        raise HTTPException(413, f"File exceeds the {limit//1024**2} MB limit.")
                    handle.write(chunk)
            if not size:
                raise HTTPException(400, "The selected file is empty.")
            job = manager.store.update(ident, filename=Path((file.filename or "recording").replace("\\", "/")).name,
                        source_file=filename, bytes=size, kind=kind, model_id=model_id,
                        pose_engine=pose_engine if kind == "video" else None,
                        status="queued", created_at=datetime.now(timezone.utc).isoformat(),
                        progress=None, result=None, error=None)
            manager.queue(ident)
            return job
        except Exception:
            shutil.rmtree(manager.folder(ident))
            manager.store.delete(ident)
            raise
        finally:
            await file.close()

    @app.post("/api/analyses/{ident}/cancel")
    def cancel(ident: str):
        job = get_job(ident)
        if job["status"] in ACTIVE:
            manager.cancel_events[ident].set()
            return manager.store.update(ident, status="cancelling")
        return job

    @app.get("/api/analyses/{ident}/media")
    def media(ident: str):
        job = get_job(ident)
        if job["kind"] != "video":
            raise HTTPException(404, "This analysis contains skeleton data only.")
        return FileResponse(manager.folder(ident) / job["source_file"])

    @app.get("/api/analyses/{ident}/skeleton")
    def skeleton(ident: str):
        job = get_job(ident)
        if job["status"] != "completed":
            raise HTTPException(409, "Skeleton data is not ready.")
        return FileResponse(manager.folder(ident) / "skeleton.json", media_type="application/json")

    @app.get("/api/analyses/{ident}/export")
    def export(ident: str, format: str = "json"):
        job = get_job(ident)
        if job["status"] != "completed":
            raise HTTPException(409, "Analysis is not complete.")
        if format == "csv":
            out = io.StringIO(newline="")
            fields = ["index", "start", "end", "logit", "score", "contributes", "observed_frames"]
            writer = csv.DictWriter(out, fieldnames=fields)
            writer.writeheader()
            writer.writerows(job["result"]["windows"])
            return Response(out.getvalue(), media_type="text/csv", headers={"Content-Disposition": 'attachment; filename="window-scores.csv"'})
        return JSONResponse(job, headers={"Content-Disposition": 'attachment; filename="fall-review.json"'})

    @app.delete("/api/analyses/{ident}", status_code=204)
    def delete(ident: str):
        job = get_job(ident)
        if job["status"] in ACTIVE:
            raise HTTPException(409, "Cancel the running analysis before deleting it.")
        folder = manager.folder(ident)
        if folder.exists():
            shutil.rmtree(folder)
        manager.store.delete(ident)
        manager.cancel_events.pop(ident, None)
        return Response(status_code=204)

    @app.get("/api/benchmark")
    def benchmark():
        base = root / "benchmarks/erm"
        provenance = json.loads((base / "provenance.json").read_text(encoding="utf-8"))
        for name, expected in provenance["bundled_artifacts_sha256"].items():
            if sha256(root / name) != expected:
                raise HTTPException(409, f"Benchmark artifact checksum mismatch: {name}")
        def rows(name):
            with (base / name).open(newline="", encoding="utf-8") as f:
                return list(csv.DictReader(f))
        return {"metrics": rows("metrics.csv"), "predictions": rows("predictions.csv"),
                "verified": True, "description": "Saved ERM results. Four held-out datasets, three seeds. Videos and full training runs are not bundled."}

    @app.get("/api/benchmark/export")
    def benchmark_export(kind: str = "metrics"):
        if kind not in ("metrics", "predictions"):
            raise HTTPException(400, "Choose metrics or predictions.")
        benchmark()
        return FileResponse(root / f"benchmarks/erm/{kind}.csv", filename=f"erm-{kind}.csv")

    app.mount("/", StaticFiles(directory=root / "web", html=True), name="web")
    return app
