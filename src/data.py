"""Acquisition, verified recording manifest, fixed splits and video windows."""
from __future__ import annotations

import csv
import concurrent.futures
import hashlib
import io
import json
import math
import os
import re
import shutil
import threading
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import requests

DOMAINS = ("caucafall", "gmdcsa24", "le2i", "urfd")
FIELDS = ("video_id", "dataset", "source_version", "source_path", "source_sha256",
          "label", "subject_id", "scene_id", "camera_id", "recording_group_id",
          "original_fps", "width", "height", "frame_count", "duration_s",
          "annotation_path", "provenance_note")


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def checksum_file(path, algorithm):
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def zenodo_files(record_id):
    data = requests.get(f"https://zenodo.org/api/records/{record_id}", timeout=30)
    data.raise_for_status()
    return data.json()["files"]


def download_file(url, dest, checksum=None, resume=True):
    """Stream a public file, preserving completed originals and partial downloads."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if checksum and checksum.startswith("md5:"):
            h = checksum_file(dest, "md5")
            if h == checksum[4:]:
                return "verified-existing"
        elif checksum and checksum.startswith("sha256:") and digest(dest) == checksum[7:]:
            return "verified-existing"
        elif checksum is None:
            return "existing-unverified"
        raise RuntimeError(f"Existing file has wrong checksum: {dest}; refusing overwrite")
    part = dest.with_name(dest.name + ".part")
    offset = part.stat().st_size if resume and part.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with requests.get(url, headers=headers, stream=True, timeout=(20, 120)) as r:
        r.raise_for_status()
        if offset and r.status_code != 206:
            offset = 0
        with open(part, "ab" if offset else "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
    if checksum:
        algo, expected = checksum.split(":", 1)
        actual = checksum_file(part, algo)
        if actual.lower() != expected.lower():
            raise RuntimeError(f"Checksum failure for {dest}: {actual} != {expected}")
    part.rename(dest)
    return "downloaded"


def parallel_download_file(url, dest, checksum, connections=16):
    """Resumable fixed-range download for a very large public archive."""
    dest = Path(dest)
    if dest.exists():
        return download_file(url, dest, checksum, resume=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    meta_path = dest.with_name(dest.name + ".ranges.json")
    head = requests.head(url, timeout=30)
    head.raise_for_status()
    size = int(head.headers["Content-Length"])
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta["size"] != size or meta["url"] != url:
            raise ValueError("Range download metadata mismatch")
    else:
        prefix = part.stat().st_size if part.exists() else 0
        span = math.ceil((size - prefix) / connections)
        ranges = [[start, min(size - 1, start + span - 1)] for start in range(prefix, size, span)]
        meta = {"url": url, "size": size, "prefix": prefix, "ranges": ranges, "done": []}
        meta_path.write_text(json.dumps(meta) + "\n")
    fd = os.open(part, os.O_RDWR | os.O_CREAT)
    os.ftruncate(fd, size)
    lock = threading.Lock()

    def worker(index, bounds):
        start, end = bounds
        with requests.get(url, headers={"Range": f"bytes={start}-{end}"}, stream=True, timeout=(20, 180)) as response:
            response.raise_for_status()
            if response.status_code != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {start}-{end}/"):
                raise RuntimeError(f"Server did not honor range {start}-{end}")
            position = start
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    os.pwrite(fd, chunk, position)
                    position += len(chunk)
            if position != end + 1:
                raise IOError(f"Incomplete byte range {start}-{end}: stopped at {position}")
        with lock:
            meta["done"].append(index)
            temp = meta_path.with_suffix(".tmp")
            temp.write_text(json.dumps(meta) + "\n")
            temp.replace(meta_path)
            print(f"range {len(meta['done'])}/{len(meta['ranges'])} complete", flush=True)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=connections) as pool:
            futures = [pool.submit(worker, i, bounds) for i, bounds in enumerate(meta["ranges"]) if i not in meta["done"]]
            for future in concurrent.futures.as_completed(futures):
                future.result()
    finally:
        os.close(fd)
    if checksum:
        algo, expected = checksum.split(":", 1)
        actual = checksum_file(part, algo)
        if actual.lower() != expected.lower():
            raise RuntimeError(f"Checksum failure for {dest}: {actual} != {expected}")
    part.rename(dest)
    meta_path.unlink()
    return "downloaded-and-verified"


def download(cfg, dataset="all", dry_run=False, source="official", resume=True, connections=1):
    raw = Path(cfg["paths"]["data"]) / "raw"
    selected = DOMAINS if dataset == "all" else (dataset,)
    for domain in selected:
        target = raw / domain
        target.mkdir(parents=True, exist_ok=True)
        if domain in ("gmdcsa24", "le2i", "caucafall") and (domain == "gmdcsa24" or source == "mirror"):
            rid = 13354453 if domain == "gmdcsa24" else 17170592
            for entry in zenodo_files(rid):
                name = entry["key"]
                if domain == "le2i" and not re.search(r"LE2I.*\.zip$", name, re.I):
                    continue
                if domain == "caucafall" and not re.search(r"Cauca_fall.*\.zip$", name, re.I):
                    continue
                if domain == "gmdcsa24" and not name.lower().endswith((".zip", ".rar", ".7z")):
                    continue
                dest = target / name
                print(domain, name, entry["size"], entry.get("checksum"), entry["links"]["self"])
                if not dry_run:
                    if domain == "le2i" and connections > 1:
                        print(parallel_download_file(entry["links"]["self"], dest, entry.get("checksum"), connections))
                    else:
                        print(download_file(entry["links"]["self"], dest, entry.get("checksum"), resume))
        elif domain == "urfd":
            page_url = cfg["datasets"][domain]["url"]
            r = requests.get(page_url, timeout=30)
            r.raise_for_status()
            links = re.findall(r'''href\s*=\s*["']([^"']+)["']''', r.text, re.I)
            rgb_urls = sorted({urljoin(page_url, x) for x in links if re.search(r"(?:fall|adl)-\d+-cam0-rgb\.zip$", x, re.I)})
            csv_urls = sorted({urljoin(page_url, x) for x in links if re.search(r"(?:fall|adl)-\d+-data\.csv$", x, re.I)})
            if len(rgb_urls) != 70 or len(csv_urls) != 70:
                raise RuntimeError(f"URFD page exposed {len(rgb_urls)} RGB archives and {len(csv_urls)} timestamp CSVs; expected 70 each.")
            urls = rgb_urls + csv_urls
            for url in urls:
                dest = target / url.rsplit("/", 1)[-1]
                print(domain, dest.name, url)
                if not dry_run:
                    print(download_file(url, dest, resume=resume))
        else:
            print(f"{domain}: automatic {source} acquisition unavailable. Place documented source files in {target}; see README.")


def _video_info(path):
    import cv2
    if Path(path).suffix.lower() == ".json":
        playlist = json.loads(Path(path).read_text())
        fps_values, sizes, times = [], set(), []
        last_t = -float("inf")
        for segment in playlist["segments"]:
            clip = Path(path).parent / segment["path"]
            cap = cv2.VideoCapture(str(clip))
            if not cap.isOpened():
                raise ValueError(f"Undecodable playlist clip: {clip}")
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            sizes.add((int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
            cap.release()
            if fps <= 0 or count <= 0:
                raise ValueError(f"Invalid playlist clip metadata: {clip}")
            fps_values.append(fps)
            for t in segment["start"] + np.arange(count) / fps:
                if t > last_t + .001:
                    times.append(float(t)); last_t = float(t)
        if len(sizes) != 1 or not np.allclose(fps_values, fps_values[0], atol=.1):
            raise ValueError(f"Inconsistent playlist image size/FPS: {path}")
        width, height = next(iter(sizes))
        return float(np.median(fps_values)), width, height, len(times), float(times[-1] - times[0])
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Undecodable video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok, _ = cap.read()
    cap.release()
    if not ok or fps <= 0 or count <= 0 or width <= 0 or height <= 0:
        raise ValueError(f"Bad video metadata or first frame: {path}")
    return fps, width, height, count, count / fps


def _urfd_info(path):
    import cv2
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".png")]
        names.sort(key=lambda n: tuple(int(x) for x in re.findall(r"\d+", Path(n).stem)))
        if not names:
            raise ValueError(f"No PNGs: {path}")
        image = cv2.imdecode(np.frombuffer(z.read(names[0]), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"First PNG undecodable: {path}")
        height, width = image.shape[:2]
    timestamps = urfd_timestamps(path, len(names))
    if len(timestamps) != len(names):
        raise ValueError(f"URFD timestamp/frame count mismatch: {path}")
    fps = (len(timestamps) - 1) / (timestamps[-1] - timestamps[0])
    return fps, width, height, len(names), timestamps[-1] - timestamps[0]


def urfd_timestamps(path, frame_count=None):
    path = Path(path)
    csv_path = path.with_name(re.sub(r"-cam0-rgb\.zip$", "-data.csv", path.name))
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing official URFD synchronization CSV: {csv_path}")
    data = np.loadtxt(csv_path, delimiter=",", usecols=(0, 1))
    if not np.array_equal(data[:, 0], np.arange(1, len(data) + 1)) or not np.all(np.diff(data[:, 1]) > 0):
        raise ValueError(f"Invalid URFD frame numbers/timestamps: {csv_path}")
    timestamps = (data[:, 1] - data[0, 1]) / 1000
    if frame_count is not None and frame_count > len(timestamps):
        missing = frame_count - len(timestamps)
        if missing > min(30, max(1, int(.1 * frame_count))):
            raise ValueError(f"Too many trailing URFD frames without timestamps: {csv_path}: {missing}")
        cadence = float(np.median(np.diff(timestamps[-min(101, len(timestamps)):])))
        timestamps = np.r_[timestamps, timestamps[-1] + cadence * np.arange(1, missing + 1)]
    if frame_count is not None and frame_count != len(timestamps):
        raise ValueError(f"URFD timestamp/frame count mismatch: {path}")
    return timestamps


def _class_from_path(domain, path, label_map):
    parts = [p.lower() for p in path.parts]
    if domain == "gmdcsa24":
        hits = {"fall": 1, "adl": 0}
        found = {v for k, v in hits.items() if k in parts}
    elif domain == "urfd":
        m = re.match(r"^(fall|adl)-\d+-cam0-rgb\.zip$", path.name.lower())
        found = {{"fall": 1, "adl": 0}[m.group(1)]} if m else set()
    else:
        # CAUCAFall action mapping and Le2i clip boundaries vary by distribution.
        # Require an audited recording-to-label CSV, never infer from ordering.
        found = {int(label_map[str(path)])} if str(path) in label_map else set()
    if len(found) != 1:
        raise ValueError(f"Unverified/ambiguous label for {path}; provide labels.csv")
    return found.pop()


def _load_label_map(path):
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or not {"source_path", "label"}.issubset(rows[0]):
        raise ValueError(f"{path} needs source_path,label columns")
    return {str(Path(row["source_path"])): row["label"] for row in rows}


def _extract_relevant_archives(root, domain):
    """Keep downloaded ZIPs; extract only media and accompanying annotations."""
    for archive in sorted(root.rglob("*.zip")):
        if domain == "urfd" or archive.name.endswith(".part") or "extracted" in archive.parts:
            continue
        out = root / "extracted" / archive.stem
        marker = out / ".extracted_sha256"
        signature = digest(archive)
        if marker.exists():
            if marker.read_text().strip() != signature:
                raise ValueError(f"Archive changed after extraction: {archive}")
            continue
        with zipfile.ZipFile(archive) as z:
            accepted = [item for item in z.infolist() if not item.is_dir() and
                        Path(item.filename).suffix.lower() in (".avi", ".mp4", ".mov", ".mkv", ".csv", ".txt")]
            needed = sum(item.file_size for item in accepted)
            free = shutil.disk_usage(root).free
            if needed * 1.1 > free:
                raise OSError(f"Insufficient disk for {archive}: need {needed}, free {free}")
            print(f"Extracting {len(accepted)} media/annotation files from {archive}; estimated {needed} bytes")
            for item in accepted:
                relative = Path(item.filename)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"Unsafe ZIP member: {item.filename}")
                dest = out / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                with z.open(item) as src, open(dest, "wb") as dst:
                    shutil.copyfileobj(src, dst, 1024 * 1024)
        out.mkdir(parents=True, exist_ok=True)
        marker.write_text(signature + "\n")


def _caucafall_mirror_rows(root, cfg):
    """Recover independent events from the audited 258-segment mirror CSV.

The mirror contains two byte-identical names per clip. Its 100 original
subject/action paths form the actual video-level units; clips are never cases.
"""
    csv_path = root / "extracted" / "Cauca_fall" / "Cauca_fall" / "caucafall.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    with open(csv_path, newline="") as f:
        annotations = list(csv.DictReader(f))
    if len(annotations) != 258 or len({r["path"] for r in annotations}) != 100:
        raise ValueError("CAUCA mirror CSV differs from audited 258 segments / 100 recordings")
    clips_root = csv_path.parent / "clips"
    indexed = defaultdict(list)
    for clip in clips_root.rglob("*.avi"):
        if clip.name.startswith("Subject."):
            continue
        parts = clip.stem.rsplit("_", 6)
        if len(parts) != 7:
            raise ValueError(f"Unknown CAUCA clip name: {clip}")
        name, label, start, end, subject, camera, _ = parts
        indexed[(name, subject, label)].append((float(start), float(end), clip))
    if sum(map(len, indexed.values())) != 258:
        raise ValueError("CAUCA mirror action-style clip count differs from 258")
    grouped = defaultdict(list)
    for annotation in annotations:
        name = Path(annotation["path"]).name
        key = (name, annotation["subject"], annotation["label"])
        matches = [clip for start, end, clip in indexed[key]
                   if abs(start - float(annotation["start"])) < .03 and abs(end - float(annotation["end"])) < .03]
        if len(matches) != 1:
            raise ValueError(f"Non-unique CAUCA clip match: {annotation}")
        grouped[annotation["path"]].append((float(annotation["start"]), float(annotation["end"]), matches[0], annotation))
    output = root / "recordings"
    output.mkdir(exist_ok=True)
    rows, problems = [], []
    for source_name, segments in sorted(grouped.items()):
        segments.sort(key=lambda x: (x[0], x[1], str(x[2])))
        category, action = source_name.split("/", 1)
        if category not in ("adl", "backwards", "forward", "side"):
            raise ValueError(f"Unknown CAUCA action category: {category}")
        subject = segments[0][3]["subject"]
        if any(s[3]["subject"] != subject for s in segments):
            raise ValueError("CAUCA subject inconsistency")
        discontinuities = [(a[1], b[0]) for a, b in zip(segments, segments[1:]) if abs(a[1] - b[0]) > .06]
        if discontinuities:
            problems.append(f"CAUCA {source_name}: clip timeline discontinuities {discontinuities}; gaps remain missing and overlaps use the earlier clip")
        video_id = f"caucafall-s{subject}-{action.lower()}"
        index_path = output / f"{video_id}.json"
        index = {"source_archive": "Cauca_fall.zip", "source_csv_sha256": digest(csv_path),
                 "source_recording": source_name, "segments": [
                     {"path": str(clip.relative_to(output.parent)), "start": start, "end": end,
                      "sha256": digest(clip)} for start, end, clip, _ in segments]}
        # All paths in the index are relative to data/raw/caucafall, so the
        # index can be moved with that directory without changing identities.
        for segment in index["segments"]:
            segment["path"] = "../" + segment["path"]
        content = json.dumps(index, sort_keys=True, indent=2) + "\n"
        if index_path.exists() and index_path.read_text() != content:
            raise ValueError(f"Existing CAUCA recording index changed: {index_path}")
        index_path.write_text(content)
        fps, width, height, count, duration = _video_info(index_path)
        rows.append(dict(zip(FIELDS, (video_id, "caucafall", cfg["datasets"]["caucafall"]["version"],
                   str(index_path), digest(index_path), int(category != "adl"), f"Subject.{subject}", "", "1",
                   video_id, fps, width, height, count, duration, str(csv_path),
                   "Zenodo processed-clips reconstruction; not Mendeley v4 originals; timestamps from clip offsets"))))
    return rows, problems


def _le2i_mirror_rows(root, cfg):
    """Reconstruct 190 original clip groups from the mirror's own CSV.

    In the mirror's action ontology, segment label 1 denotes the fall event;
    exactly 130 groups contain one, and 60 do not. Subject 0 is an unknown
    placeholder and is deliberately not used as a person identity.
    """
    csv_path = root / "extracted" / "LE2I" / "LE2I" / "le2i.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    with open(csv_path, newline="") as f:
        annotations = list(csv.DictReader(f))
    if len(annotations) != 967 or len({r["path"] for r in annotations}) != 190:
        raise ValueError("Le2i mirror differs from audited 967 segments / 190 recordings")
    clips_root = csv_path.parent / "clips"
    indexed = defaultdict(list)
    for clip in clips_root.rglob("*.avi"):
        parts = clip.stem.rsplit("_", 6)
        if len(parts) != 7:
            raise ValueError(f"Unknown Le2i clip name: {clip}")
        prefix, label, start, end, subject, camera, _ = parts
        name = prefix.split("_", 1)[1] if "_" in prefix else prefix
        indexed[(clip.parent.name, name, label, subject, camera)].append((float(start), float(end), clip))
    if sum(map(len, indexed.values())) != 967:
        raise ValueError("Le2i clip count differs from 967")
    grouped = defaultdict(list)
    for annotation in annotations:
        scene, name = annotation["path"].split("/", 1)
        key = (scene, name, annotation["label"], annotation["subject"], annotation["cam"])
        matches = [clip for start, end, clip in indexed[key]
                   if abs(start - float(annotation["start"])) < .03 and abs(end - float(annotation["end"])) < .03]
        if len(matches) != 1:
            raise ValueError(f"Non-unique Le2i clip match: {annotation}")
        grouped[annotation["path"]].append((float(annotation["start"]), float(annotation["end"]), matches[0], annotation))
    output = root / "recordings"
    output.mkdir(exist_ok=True)
    rows, problems = [], []
    for source_name, segments in sorted(grouped.items()):
        segments.sort(key=lambda x: (x[0], x[1], str(x[2])))
        scene, name = source_name.split("/", 1)
        labels = {s[3]["label"] for s in segments}
        label = int("1" in labels)
        discontinuities = [(a[1], b[0]) for a, b in zip(segments, segments[1:]) if abs(a[1] - b[0]) > .06]
        if discontinuities:
            problems.append(f"Le2i {source_name}: {len(discontinuities)} clip timeline discontinuities; gaps remain missing and overlaps use earlier clip")
        video_id = f"le2i-{scene.lower()}-{name.lower()}"
        index_path = output / f"{video_id}.json"
        index = {"source_archive": "LE2I.zip", "source_csv_sha256": digest(csv_path),
                 "source_recording": source_name, "segments": [
                     {"path": "../" + str(clip.relative_to(root)), "start": start, "end": end,
                      "sha256": digest(clip)} for start, end, clip, _ in segments]}
        content = json.dumps(index, sort_keys=True, indent=2) + "\n"
        if index_path.exists() and index_path.read_text() != content:
            raise ValueError(f"Existing Le2i recording index changed: {index_path}")
        index_path.write_text(content)
        fps, width, height, count, duration = _video_info(index_path)
        camera = segments[0][3]["cam"]
        rows.append(dict(zip(FIELDS, (video_id, "le2i", cfg["datasets"]["le2i"]["version"],
                   str(index_path), digest(index_path), label, "", scene, camera, video_id,
                   fps, width, height, count, duration, str(csv_path),
                   "Zenodo 190-group processed-clips variant; PHARL membership unverified; subject unknown; video fall iff action label 1 present"))))
    if Counter(r["label"] for r in rows) != {1: 130, 0: 60}:
        raise ValueError("Le2i binary mapping disagrees with audited 130 Fall / 60 ADL mirror grouping")
    return rows, problems


def inspect(cfg):
    """Inspect only actual media; do not invent missing datasets or labels."""
    raw = Path(cfg["paths"]["data"]) / "raw"
    rows, problems = [], []
    for domain in DOMAINS:
        root = raw / domain
        if not root.exists():
            problems.append(f"{domain}: no files")
            continue
        if domain != "urfd":
            try:
                _extract_relevant_archives(root, domain)
            except Exception as e:
                problems.append(f"{domain}: archive extraction failed: {e}")
        if domain == "caucafall" and (root / "extracted" / "Cauca_fall" / "Cauca_fall" / "caucafall.csv").exists():
            try:
                new_rows, new_problems = _caucafall_mirror_rows(root, cfg)
                rows.extend(new_rows); problems.extend(new_problems)
            except Exception as e:
                problems.append(f"CAUCA mirror reconstruction failed: {e}")
            continue
        if domain == "le2i" and (root / "extracted" / "LE2I" / "LE2I" / "le2i.csv").exists():
            try:
                new_rows, new_problems = _le2i_mirror_rows(root, cfg)
                rows.extend(new_rows); problems.extend(new_problems)
            except Exception as e:
                problems.append(f"Le2i mirror reconstruction failed: {e}")
            continue
        label_map = _load_label_map(root / "labels.csv")
        media = sorted(p for p in root.rglob("*") if p.is_file() and
                       (p.suffix.lower() in (".avi", ".mp4", ".mov", ".mkv") or
                        (domain == "urfd" and re.match(r"^(fall|adl)-\d+-cam0-rgb\.zip$", p.name.lower()))))
        if not media:
            problems.append(f"{domain}: no directly readable media; extract archive without deleting it")
        for path in media:
            try:
                label = _class_from_path(domain, path, label_map)
                fps, w, h, count, duration = _urfd_info(path) if path.suffix.lower() == ".zip" else _video_info(path)
                rel = path.relative_to(root)
                ident = re.sub(r"[^a-z0-9]+", "-", str(rel.with_suffix("")).lower()).strip("-")
                video_id = f"{domain}-{ident}"
                subject = next((p for p in rel.parts if re.match(r"^(?:subject|person)[ _-]?\d+$", p, re.I)), "")
                scene = next((p for p in rel.parts if re.match(r"^(?:scene|camera|room)[ _-]?\w+$", p, re.I)), "")
                group = re.sub(r"-cam\d+-rgb$", "", path.stem.lower()) if domain == "urfd" else video_id
                annotation = (path.with_name(re.sub(r"-cam0-rgb\.zip$", "-data.csv", path.name)) if domain == "urfd"
                              else path.parent.parent / (path.parent.name + ".csv") if domain == "gmdcsa24" else None)
                if annotation is not None and not annotation.exists():
                    raise FileNotFoundError(f"Missing source annotation {annotation}")
                if domain == "gmdcsa24" and not any(line.split(",", 1)[0].strip() == path.name for line in annotation.read_text(errors="replace").splitlines()[1:]):
                    raise ValueError(f"GMDCSA recording absent from its class CSV: {path}")
                rows.append(dict(zip(FIELDS, (video_id, domain, cfg["datasets"][domain]["version"],
                   str(path), digest(path), label, subject, scene, "0" if domain == "urfd" else "",
                   group, fps, w, h, count, duration,
                   str(annotation) if annotation is not None else "",
                   ("official synchronization CSV; trailing frame timestamps extrapolated where CSV ends early"
                    if domain == "urfd" and len(np.loadtxt(annotation, delimiter=",", usecols=(0,))) < count
                    else "official synchronization CSV" if domain == "urfd" else "fps-derived timestamps")))))
                if domain == "urfd":
                    trailing = count - len(np.loadtxt(annotation, delimiter=",", usecols=(0,)))
                    if trailing:
                        problems.append(f"URFD {video_id}: {trailing} trailing PNGs lack CSV times; timestamps extrapolated using median preceding cadence")
            except Exception as e:
                problems.append(f"{path}: {e}")
    ids = [r["video_id"] for r in rows]
    hashes = [r["source_sha256"] for r in rows]
    if len(ids) != len(set(ids)):
        problems.append("duplicate video_id detected")
    duplicates = [h for h, n in Counter(hashes).items() if n > 1]
    if duplicates:
        problems.append(f"duplicate source hashes: {len(duplicates)}")
    out = Path(cfg["paths"]["data"]) / "manifest.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)
    return rows, problems


def write_status(cfg, rows, problems):
    """Generated audit/progress record; never present partial data as E1 results."""
    import datetime
    import subprocess
    raw = Path(cfg["paths"]["data"]) / "raw"
    lines = ["# E1 status", "", f"Audit generated: {datetime.datetime.now().astimezone().isoformat(timespec='seconds')}",
             "", "**Experiment incomplete until all four domains, pose caches, 12 real training trajectories, and target predictions exist.**", "",
             "## Real-file audit", "", "| Domain | Version | Videos | Fall | ADL | Known subjects | Median duration (s) | FPS range | Raw bytes |",
             "|---|---|---:|---:|---:|---:|---:|---|---:|"]
    for domain in DOMAINS:
        subset = [r for r in rows if r["dataset"] == domain]
        durations = [float(r["duration_s"]) for r in subset]
        fps = [float(r["original_fps"]) for r in subset]
        size = sum(p.stat().st_blocks * 512 for p in (raw / domain).rglob("*") if p.is_file()) if (raw / domain).exists() else 0
        lines.append(f"| {domain} | {cfg['datasets'][domain]['version']} | {len(subset)} | {sum(int(r['label']) for r in subset)} | {sum(1-int(r['label']) for r in subset)} | {len({r['subject_id'] for r in subset if r['subject_id']})} | {np.median(durations):.1f} | {min(fps):.2f}–{max(fps):.2f} | {size} |" if subset else
                     f"| {domain} | {cfg['datasets'][domain]['version']} | 0 | 0 | 0 | 0 | — | — | {size} |")
    lines += ["", "GMDCSA-24 v2.1 archive currently has 79 Fall and 81 ADL files, while its paper reports 81 Fall and 79 ADL. Folder names and class CSVs determine the manifest labels.",
              "CAUCAFall is reconstructed from the Zenodo mirror's 258 unique clips grouped into 100 recordings; each clip has a byte-identical duplicate. It is a dataset-version variant, not official Mendeley v4.",
              "Le2i mirror membership relative to PHARL's 190-video subset has not been independently verified.", "",
              "## Pose cache and Phase 2 readiness", "", "| Domain | Cached videos | Empty skeletons | Missing-frame fraction | Visible joints | Extreme-pose frames | Timestamp anomalies | ≥96 valid 15-fps steps |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for domain in DOMAINS:
        subset = [r for r in rows if r["dataset"] == domain]
        stats = []
        for row in subset:
            path = Path(cfg["paths"]["data"]) / "skeletons" / domain / (row["video_id"] + ".npz")
            if path.exists():
                with np.load(path) as z:
                    valid = z["valid_frame"]
                    joints = z["valid_joint"]
                    timestamps = z["timestamp"]
                    anomaly = int(not np.all(np.diff(timestamps) > 0))
                    xy = z["xy"]
                    width, height = z["image_size"]
                    extreme = 0
                    for frame_xy, frame_valid in zip(xy, joints):
                        if frame_valid.sum() >= 8:
                            points = frame_xy[frame_valid]
                            extreme += int(np.ptp(points[:, 0]) > .85 * width or np.ptp(points[:, 1]) > .95 * height)
                    stats.append((int(valid.sum()), len(valid), int(joints.sum()), joints.size, extreme, anomaly, int(row["label"])))
        if stats:
            lines.append(f"| {domain} | {len(stats)} | {sum(s[0] == 0 for s in stats)} | {sum(s[1]-s[0] for s in stats)/sum(s[1] for s in stats):.3f} | {sum(s[2] for s in stats)/sum(s[3] for s in stats):.3f} | {sum(s[4] for s in stats)} | {sum(s[5] for s in stats)} | {sum(s[0] >= 96 for s in stats)}/{len(stats)} |")
            for label, name in ((1, "Fall"), (0, "ADL")):
                klass = [s for s in stats if s[-1] == label]
                lines.append(f"| ↳ {name} | {len(klass)} | — | — | — | — | — | {sum(s[0] >= 96 for s in klass)}/{len(klass)} |")
        else:
            lines.append(f"| {domain} | 0 | — | — | — | — | — | unavailable |")
    lines.append("Extreme-pose frames are a diagnostic: visible joints span over 85% of image width or 95% of height. They remain in E1.")
    if (Path(cfg["paths"]["data"]) / "skeletons" / "le2i" / "le2i-office-video_31.npz").exists():
        lines.append("Manual overlay check: Le2i `le2i-office-video_31` has a visually suspect pose after the person leaves the frame. The recording is retained; confidence alone does not detect every pose error.")
    split_path = Path(cfg["paths"]["data"]) / "splits.json"
    lines += ["", "## Frozen source splits", ""]
    if split_path.exists():
        split = json.loads(split_path.read_text())
        lines.append(f"Split SHA-256: `{split['split_hash']}`; manifest SHA-256: `{split['manifest_sha256']}`.")
        for domain, fold in split["folds"].items():
            lines.append(f"- {domain}: {len(fold['train'])} source train, {len(fold['val'])} source validation, {len(fold['target'])} target recordings; grouping {fold['grouping']}.")
    else:
        lines.append("Not frozen yet.")
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=memory.free,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=False)
        gpu_state = f"GPU free/total MiB and utilization at audit: `{gpu.stdout.strip()}`." if gpu.returncode == 0 else "GPU state unavailable at audit."
    except FileNotFoundError:
        gpu_state = "GPU state unavailable at audit."
    lines += ["", "## Execution resources", ""]
    lines.append(gpu_state)
    lines.append("Full pose extraction and 12 real training trajectories remain pending while the shared GPU is occupied; rerun the recovery commands when sufficient memory is free.")
    lines += ["", "## Unresolved issues and deviations", ""]
    lines += ["- " + p for p in problems] if problems else ["- None from manifest inspection."]
    lines += ["", "## Recovery commands", "", "```bash",
              "source /opt/miniconda3/etc/profile.d/conda.sh", "conda activate datpt",
              "python run.py download --dataset le2i --source mirror --resume --connections 24",
              "python run.py inspect", "python run.py extract --dataset all --device cuda --resume",
              "python run.py split --seed 2026", "python run.py train --method erm --fold all --seeds 42,123,2026 --resume",
              "python run.py train --method swad --fold all --seeds 42,123,2026 --resume", "python run.py report", "```", ""]
    path = Path(cfg["paths"]["results"]) / "STATUS.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return path


def load_manifest(cfg):
    path = Path(cfg["paths"]["data"]) / "manifest.csv"
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if len({r["video_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate video IDs in manifest")
    return rows


def _choose_group_split(rows, seed, fraction):
    """Try subject grouping, then immutable recording grouping. Require both classes."""
    labels = {int(r["label"]) for r in rows}
    if labels != {0, 1}:
        raise ValueError("Both classes required in each source domain")
    subjects = [r["subject_id"] for r in rows]
    use_subject = all(subjects) and len(set(subjects)) >= 3
    key = "subject_id" if use_subject else "recording_group_id"
    groups = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    rng = np.random.default_rng(seed)
    keys = sorted(groups)
    target_count = max(1, round(len(rows) * fraction))
    best = None
    # GMDCSA has four subjects: inspect all single-subject holds explicitly.
    for _ in range(1000):
        perm = rng.permutation(keys).tolist()
        selected, count = [], 0
        for k in perm:
            selected.append(k); count += len(groups[k])
            if count >= target_count:
                break
        val = [r for k in selected for r in groups[k]]
        train = [r for k in keys if k not in selected for r in groups[k]]
        if {int(r["label"]) for r in val} != {0, 1} or {int(r["label"]) for r in train} != {0, 1}:
            continue
        error = abs(len(val) / len(rows) - fraction) + abs(np.mean([int(r["label"]) for r in val]) - np.mean([int(r["label"]) for r in rows]))
        if best is None or error < best[0]:
            best = (error, train, val)
    if best is None and use_subject:
        return _choose_group_split([{**r, "subject_id": ""} for r in rows], seed, fraction)
    if best is None:
        raise ValueError("No source train/validation grouping preserves both classes")
    return best[1], best[2], key


def make_splits(cfg, seed):
    rows = load_manifest(cfg)
    if {r["dataset"] for r in rows} != set(DOMAINS):
        raise ValueError("Four audited domains required before freezing splits")
    folds = {}
    for target in DOMAINS:
        train, val, grouping = [], [], {}
        for domain in DOMAINS:
            if domain == target:
                continue
            source = [r for r in rows if r["dataset"] == domain]
            tr, va, key = _choose_group_split(source, seed, cfg["train"]["val_fraction"])
            train += [r["video_id"] for r in tr]
            val += [r["video_id"] for r in va]
            grouping[domain] = key
        test = [r["video_id"] for r in rows if r["dataset"] == target]
        if (set(train) & set(val)) or (set(train) & set(test)) or (set(val) & set(test)):
            raise AssertionError("Split overlap")
        folds[target] = {"train": sorted(train), "val": sorted(val), "target": sorted(test), "grouping": grouping}
    artifact = {"protocol": "e1-v1", "seed": seed, "manifest_sha256": digest(Path(cfg["paths"]["data"]) / "manifest.csv"), "folds": folds}
    artifact["split_hash"] = canonical_hash(artifact)
    path = Path(cfg["paths"]["data"]) / "splits.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old != artifact:
            raise ValueError("Immutable splits.json already exists with different content")
    else:
        path.write_text(json.dumps(artifact, indent=2) + "\n")
    return artifact


def load_splits(cfg):
    artifact = json.loads((Path(cfg["paths"]["data"]) / "splits.json").read_text())
    check = {k: v for k, v in artifact.items() if k != "split_hash"}
    if canonical_hash(check) != artifact["split_hash"]:
        raise ValueError("Split hash mismatch")
    if digest(Path(cfg["paths"]["data"]) / "manifest.csv") != artifact["manifest_sha256"]:
        raise ValueError("Manifest changed after splits were frozen")
    return artifact


def resample_indices(timestamps, fps=15, tolerance=0.04):
    t = np.asarray(timestamps, dtype=np.float64)
    if len(t) == 0 or not np.all(np.diff(t) > 0):
        raise ValueError("Source timestamps must be strictly increasing")
    grid = np.arange(math.floor((t[-1] - t[0]) * fps) + 1) / fps + t[0]
    right = np.searchsorted(t, grid)
    left = np.maximum(right - 1, 0)
    right = np.minimum(right, len(t) - 1)
    idx = np.where(np.abs(t[right] - grid) < np.abs(t[left] - grid), right, left)
    return grid, np.where(np.abs(t[idx] - grid) <= tolerance, idx, -1)


def window_starts(length, size=64, stride=32):
    if length <= size:
        return [0]
    starts = list(range(0, length - size + 1, stride))
    terminal = length - size
    if starts[-1] != terminal:
        starts.append(terminal)
    return starts


def make_windows(frames, size=64, stride=32):
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 3 or frames.shape[1:] != (17, 3):
        raise ValueError("Expected [T,17,3]")
    if len(frames) == 0:
        frames = np.zeros((1, 17, 3), np.float32)
        observed = 0
    else:
        observed = len(frames)
    windows, masks = [], []
    for start in window_starts(len(frames), size, stride):
        chunk = frames[start:start + size]
        mask = np.arange(size) < max(0, min(size, observed - start))
        if len(chunk) < size:
            visible = np.flatnonzero(np.any(chunk[..., 2] > 0, axis=1))
            last_valid = chunk[visible[-1]:visible[-1] + 1] if len(visible) else chunk[-1:]
            chunk = np.concatenate([chunk, np.repeat(last_valid, size - len(chunk), axis=0)])
        windows.append(chunk.transpose(2, 0, 1)[..., None])
        masks.append(mask)
    return np.stack(windows), np.stack(masks)


def load_video(cfg, row):
    path = Path(cfg["paths"]["data"]) / "skeletons" / row["dataset"] / (row["video_id"] + ".npz")
    with np.load(path) as z:
        xy, score, valid = z["xy"], z["score"], z["valid_joint"]
        width, height = z["image_size"]
    scale = float(max(width, height))
    x = (xy[..., 0] - width / 2) / scale
    y = (xy[..., 1] - height / 2) / scale
    values = np.stack([x, y, score], axis=-1).astype(np.float32)
    values[~valid] = 0
    windows, mask = make_windows(values, cfg["preprocess"]["window_frames"], cfg["preprocess"]["stride_frames"])
    return windows, mask
