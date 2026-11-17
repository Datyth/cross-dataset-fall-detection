"""CPU-only, video-level domain diagnostics for the frozen ds-v1 protocol."""
from __future__ import annotations

import csv
import itertools
import json
import platform
import subprocess
from pathlib import Path

import numpy as np

from .data import DOMAINS, canonical_hash, digest, load_manifest, load_splits

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research/results/domain_analysis"
RGB = ["brightness", "contrast", "saturation", "red", "green", "blue"]
SK = ["missing", "lowconf", "confidence", "area", "angle", "speed", "speed_p90", "angle_std"]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    with temp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    temp.replace(path)


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def inputs():
    cfg = {"paths": {"data": str(ROOT / "research/data")}}
    return cfg, load_manifest(cfg), load_splits(cfg)


def source(row):
    p = Path(row["source_path"])
    if p.is_absolute():
        return p
    if str(p).startswith("data/"):
        return ROOT / "research" / p
    return ROOT / p


def cache(row):
    return ROOT / "research/data/skeletons" / row["dataset"] / (row["video_id"] + ".npz")


def group(row):
    return row["dataset"] + ":" + (row.get("subject_id") or row["recording_group_id"])


def p0():
    import cv2
    import sklearn
    cfg, rows, splits = inputs()
    if len(rows) != 520 or sum(int(r["label"]) for r in rows) != 289:
        raise ValueError("Expected 520 videos, 289 Fall")
    by_id = {r["video_id"]: r for r in rows}
    for fold, part in splits["folds"].items():
        tr, va, te = (set(part[k]) for k in ("train", "val", "target"))
        assert not (tr & va or tr & te or va & te)
        assert tr | va | te == set(by_id)
        assert te == {r["video_id"] for r in rows if r["dataset"] == fold}
        for d, key in part["grouping"].items():
            a = {by_id[v][key] for v in tr if by_id[v]["dataset"] == d}
            b = {by_id[v][key] for v in va if by_id[v]["dataset"] == d}
            assert not a & b
    audit = []
    for r in rows:
        errors = []
        try:
            p = source(r)
            if digest(p) != r["source_sha256"]:
                raise ValueError("Source hash mismatch")
            if p.suffix == ".json":
                for seg in json.loads(p.read_text())["segments"]:
                    if digest(p.parent / seg["path"]) != seg["sha256"]:
                        raise ValueError("Segment hash mismatch")
            with np.load(cache(r)) as z:
                assert z["source_sha256"].item() == r["source_sha256"]
                assert z["xy"].shape == (*z["score"].shape, 2)
                assert np.all(np.diff(z["timestamp"]) > 0)
                pose_hash = str(z["pose_config_hash"].item())
                fallback = bool(z["timestamp_fallback"].item())
        except Exception as e:
            errors.append(str(e) or type(e).__name__)
            pose_hash, fallback = "", False
        audit.append({**r, "cache_sha256": digest(cache(r)) if cache(r).exists() else "",
                      "pose_config_hash": pose_hash, "timestamp_fallback": fallback,
                      "error": " | ".join(errors)})
        print(f"P0 {len(audit)}/520 {r['video_id']}", flush=True)
    write_csv(OUT / "audit.csv", audit)
    settings = {"protocol": "ds-v1", "rgb_frames": 8, "confidence": .1,
                "min_geometry_joints": 5, "permutations": 1000, "seed": 2026,
                "balanced_repeats": 100, "c2st_repeats": 20,
                "manifest_sha256": digest(Path(cfg["paths"]["data"]) / "manifest.csv"),
                "splits_sha256": digest(Path(cfg["paths"]["data"]) / "splits.json"),
                "audit_sha256": digest(OUT / "audit.csv"),
                "implementation_sha256": digest(Path(__file__)),
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
                "python": platform.python_version(), "numpy": np.__version__,
                "opencv": cv2.__version__, "sklearn": sklearn.__version__}
    bad = [r for r in audit if r["error"]]
    (OUT / "audit_summary.md").write_text(f"# ds-v1 audit\n\nVideos: 520. Fall: 289. Non-Fall: 231. Errors: {len(bad)}.\n\n"
        "Subject IDs remain unknown where absent. Cache valid_joint already incorporates the pose confidence threshold, so missing and low-confidence rates may coincide.\n")
    if bad:
        raise ValueError("P0 failed, inspect audit.csv")
    path = OUT / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != settings:
        raise ValueError("Frozen ds-v1 changed, use a new output directory/protocol")
    write_json(path, settings)


def checked_inputs():
    settings = json.loads((OUT / "protocol.json").read_text())
    cfg, rows, splits = inputs()
    for name in ("manifest", "splits"):
        extension = ".csv" if name == "manifest" else ".json"
        assert digest(Path(cfg["paths"]["data"]) / (name + extension)) == settings[name + "_sha256"]
    assert digest(Path(__file__)) == settings["implementation_sha256"]
    assert digest(OUT / "audit.csv") == settings["audit_sha256"]
    return rows


def rgb_features(row):
    import cv2
    from .pose import _source_frames
    r = {**row, "source_path": str(source(row))}
    # Stream all frames to preserve reconstructed playlists and timestamp ordering.
    first, last = None, None
    for t, _ in _source_frames(r):
        if first is None:
            first = t
        last = t
    if first is None or last < first:
        raise ValueError("No decoded frames")
    targets = first + (np.arange(8) + .5) / 8 * (last - first)
    best = [None] * 8
    distances = np.full(8, np.inf)
    for index, (t, frame) in enumerate(_source_frames(r)):
        delta = np.abs(targets - t)
        for k in np.flatnonzero(delta < distances):
            distances[k] = delta[k]
            best[k] = (index, frame.copy())
    chosen = dict(best)
    features = []
    for frame in chosen.values():
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(float)
        y = rgb @ np.array([.299, .587, .114])
        vmax, vmin = rgb.max(axis=-1), rgb.min(axis=-1)
        sat = np.divide(vmax-vmin, vmax, out=np.zeros_like(vmax), where=vmax > 0)
        features.append([y.mean(), y.std(), sat.mean(), *rgb.mean(axis=(0, 1)), (y < 50).mean()])
    a = np.array(features)
    return {**dict(zip(RGB + ["dark_fraction"], a.mean(axis=0))),
            "temporal_brightness_std": a[:, 0].std(), "unique_frames": len(a)}


def skeleton_features(row):
    with np.load(cache(row)) as z:
        observed = z["source_indices"] >= 0
        xy = z["xy"].astype(float) / z["image_size"]
        score, valid, ts = z["score"], z["valid_joint"].astype(bool), z["timestamp"]
        fallback = bool(z["timestamp_fallback"].item())
    good = valid & np.isfinite(xy).all(-1) & (score >= .1) & observed[:, None]
    area, angles = [], []
    for t in np.flatnonzero(observed):
        if good[t].sum() >= 5:
            points = xy[t, good[t]]
            area.append(float(np.prod(np.ptp(points, axis=0))))
        if good[t, [5, 6, 11, 12]].all():
            torso = xy[t, [5, 6]].mean(0) - xy[t, [11, 12]].mean(0)
            if np.linalg.norm(torso) > 1e-8:
                angles.append(float(abs(np.arctan2(torso[0], -torso[1]))))
    dt = np.diff(ts)
    adjacent = good[1:] & good[:-1] & (dt[:, None] > 0) & (dt[:, None] <= .1)
    if fallback:
        adjacent[:] = False
    speed = (np.linalg.norm(np.diff(xy, axis=0), axis=-1) / dt[:, None])[adjacent]
    def stat(x, fn):
        return float(fn(x)) if len(x) else float("nan")
    n = int(observed.sum()) * 17
    if n == 0:
        raise ValueError("No observed frames")
    return {"missing": float(1-valid[observed].mean()),
            "lowconf": float((~valid[observed] | (score[observed] < .1)).mean()),
            "confidence": stat(score[valid & observed[:, None]], np.mean),
            "area": stat(area, np.median), "angle": stat(angles, np.median),
            "speed": stat(speed, np.mean), "speed_p90": stat(speed, lambda a: np.quantile(a, .9)),
            "angle_std": stat(angles, np.std), "timestamp_fallback": fallback}


def extract(kind):
    rows = checked_inputs()
    output = OUT / f"{kind}_video_features.csv"
    previous = {r["video_id"]: r for r in read_csv(output)} if output.exists() else {}
    result = []
    for row in rows:
        if row["video_id"] in previous:
            result.append(previous[row["video_id"]])
        else:
            value = rgb_features(row) if kind == "rgb" else skeleton_features(row)
            result.append({"video_id": row["video_id"], "dataset": row["dataset"],
                           "label": row["label"], "group": group(row), **value})
            write_csv(output, result + [v for k, v in previous.items() if k not in {r["video_id"] for r in result}])
        print(f"{kind} {len(result)}/520 {row['video_id']}", flush=True)
    write_csv(output, result)
    summaries, keys = [], RGB if kind == "rgb" else SK
    for d in DOMAINS:
        for label in ("0", "1"):
            subset = [r for r in result if r["dataset"] == d and str(r["label"]) == label]
            for key in keys:
                a = np.array([float(r[key]) for r in subset])
                finite = a[np.isfinite(a)]
                summaries.append({"dataset": d, "label": label, "feature": key, "n": len(a),
                                  "missing": int((~np.isfinite(a)).sum()),
                                  "median": float(np.median(finite)) if len(finite) else "",
                                  "q25": float(np.quantile(finite, .25)) if len(finite) else "",
                                  "q75": float(np.quantile(finite, .75)) if len(finite) else ""})
    write_csv(OUT / f"{kind}_summary.csv", summaries)
    feature_plot(result, keys, kind)


def feature_plot(rows, keys, kind):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(keys), 2, figsize=(11, 2.3 * len(keys)), squeeze=False)
    for i, key in enumerate(keys):
        for j, label in enumerate(("0", "1")):
            arrays = [np.array([float(r[key]) for r in rows if r["dataset"] == d and str(r["label"]) == label]) for d in DOMAINS]
            axes[i, j].boxplot([a[np.isfinite(a)] for a in arrays], tick_labels=DOMAINS)
            axes[i, j].set_title(f"{key}, {'Fall' if label == '1' else 'Non-Fall'}")
    fig.tight_layout()
    fig.savefig(OUT / f"{kind}_distributions.png", dpi=200)
    plt.close(fig)


def prepare(x):
    x = np.asarray(x, float).copy()
    if np.any(np.isfinite(x).sum(0) == 0):
        raise ValueError("Entire feature undefined, inspect summary before testing")
    med = np.nanmedian(np.where(np.isfinite(x), x, np.nan), axis=0)
    missing = ~np.isfinite(x)
    x[missing] = np.broadcast_to(med, x.shape)[missing]
    flags = missing[:, missing.any(0)]
    if flags.shape[1]:
        x = np.column_stack((x, flags))
    sd = x.std(0)
    return ((x-x.mean(0))[:, sd > 0] / sd[sd > 0])


def kernel(x, bandwidth):
    distance = np.maximum(0, (x*x).sum(1)[:, None] + (x*x).sum(1)[None, :] - 2*x@x.T)
    return np.exp(-distance / (2*bandwidth**2))


def mmd(k, n):
    return float(k[:n, :n].mean() + k[n:, n:].mean() - 2*k[:n, n:].mean())


def permutation(x, n, bandwidth, rng, repeats=1000):
    k = kernel(x, bandwidth)
    observed = mmd(k, n)
    # Compute batch statistics by matrix multiplication, bounded CPU memory.
    exceed = 0
    for start in range(0, repeats, 50):
        count = min(50, repeats-start)
        w = np.full((count, len(x)), -1/(len(x)-n))
        for row in w:
            row[rng.choice(len(x), n, replace=False)] = 1/n
        values = np.einsum("ij,ij->i", w@k, w)
        exceed += int((values >= observed-1e-12).sum())
    return observed, (exceed+1)/(repeats+1)


def holm(p):
    result = np.full(len(p), np.nan)
    order = sorted((i for i, v in enumerate(p) if np.isfinite(v)), key=lambda i: p[i])
    previous = 0.
    for rank, i in enumerate(order):
        previous = max(previous, min(1., (len(p)-rank)*p[i]))
        result[i] = previous
    return result


def c2st(rows, keys, rng):
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import roc_auc_score
    domains = sorted({r["dataset"] for r in rows})
    values = []
    for repeat in range(20):
        chosen = []
        for d in domains:
            for label in ("0", "1"):
                ids = [i for i, r in enumerate(rows) if r["dataset"] == d and str(r["label"]) == label]
                if len(ids) < 20:
                    return {"status": "not_estimable", "auc": "", "q05": "", "q95": ""}
                chosen.extend(rng.choice(ids, 20, replace=False))
        selected = [rows[i] for i in chosen]
        x = np.array([[float(r[k]) for k in keys] for r in selected])
        x[~np.isfinite(x)] = np.nan
        y = np.array([domains.index(r["dataset"]) for r in selected])
        labels = np.array([int(r["label"]) for r in selected])
        groups = np.array([r["group"] for r in selected])
        cv = StratifiedGroupKFold(4, shuffle=True, random_state=2026+repeat)
        pred = np.full(len(y), np.nan)
        for tr, te in cv.split(x, y*2+labels, groups):
            if any(len(np.unique(v)) < 2 for v in (y[tr], y[te], labels[tr], labels[te])):
                return {"status": "not_estimable", "auc": "", "q05": "", "q95": ""}
            model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                                  LogisticRegression(C=1., max_iter=1000))
            model.fit(x[tr], y[tr])
            pred[te] = model.predict_proba(x[te])[:, 1]
        values.append(roc_auc_score(y, pred))
    return {"status": "ok", "auc": float(np.median(values)),
            "q05": float(np.quantile(values, .05)), "q95": float(np.quantile(values, .95))}


def p3():
    checked_inputs()
    rng = np.random.default_rng(2026)
    rgb = read_csv(OUT / "rgb_video_features.csv")
    sk = read_csv(OUT / "skeleton_video_features.csv")
    expected = {r["video_id"] for r in checked_inputs()}
    for rows in (rgb, sk):
        assert len(rows) == 520 and {r["video_id"] for r in rows} == expected
    balanced = []
    for _ in range(100):
        ids = []
        for d in DOMAINS:
            for label in ("0", "1"):
                candidates = [r["video_id"] for r in sk if r["dataset"] == d and r["label"] == label]
                ids.extend(rng.choice(candidates, 20, replace=False).tolist())
        balanced.append(ids)
    write_json(OUT / "balanced_samples.json", balanced)
    tests, c2 = [], []
    for family, rows, keys, label in (("rgb", rgb, RGB, None), ("skeleton", sk, SK, None),
                                      ("skeleton_fall", sk, SK, "1"), ("skeleton_nonfall", sk, SK, "0")):
        rows = [r for r in rows if label is None or r["label"] == label]
        x = prepare([[float(r[k]) for k in keys] for r in rows])
        a, b = rng.integers(len(x), size=(2, 10000))
        distances = np.linalg.norm(x[a]-x[b], axis=1)
        if not (distances > 0).any():
            raise ValueError("No nonzero distances")
        bandwidth = float(np.median(distances[distances > 0]))
        for d1, d2 in itertools.combinations(DOMAINS, 2):
            i = [j for j, r in enumerate(rows) if r["dataset"] == d1]
            j = [j for j, r in enumerate(rows) if r["dataset"] == d2]
            obs, p_video = permutation(x[i+j], len(i), bandwidth, rng)
            means, counts = [], []
            for indices in (i, j):
                groups = sorted({rows[k]["group"] for k in indices})
                counts.append(len(groups))
                means.extend([x[[k for k in indices if rows[k]["group"] == g]].mean(0) for g in groups])
            grouped, p_group = permutation(np.array(means), counts[0], bandwidth, rng)
            sensitivity = []
            for ids in balanced:
                ids = set(ids)
                aa = [k for k in i if rows[k]["video_id"] in ids]
                bb = [k for k in j if rows[k]["video_id"] in ids]
                sensitivity.append(mmd(kernel(x[aa+bb], bandwidth), len(aa)))
            tests.append({"family": family, "domain1": d1, "domain2": d2,
                          "n1": len(i), "n2": len(j), "groups1": counts[0], "groups2": counts[1],
                          "bandwidth": bandwidth, "mmd2": obs, "p_video_exploratory": p_video,
                          "group_mmd2": grouped, "p_raw": p_group,
                          "balanced_median": float(np.median(sensitivity)),
                          "balanced_q05": float(np.quantile(sensitivity, .05)),
                          "balanced_q95": float(np.quantile(sensitivity, .95))})
            print(f"P3 {family} {d1}/{d2}", flush=True)
            if family.startswith("skeleton"):
                # Only the all-class family can supply balanced class subsets.
                if label is None:
                    c2.append({"domain1": d1, "domain2": d2,
                               **c2st([rows[k] for k in i+j], keys, rng)})
    for row, adjusted in zip(tests, holm([r["p_raw"] for r in tests])):
        row["p_holm"] = float(adjusted)
    write_csv(OUT / "mmd_tests.csv", tests)
    write_csv(OUT / "c2st.csv", c2)
    write_json(OUT / "analysis_inputs.json", {p.name: digest(p) for p in
               (OUT / "rgb_video_features.csv", OUT / "skeleton_video_features.csv", OUT / "protocol.json")})
    matrix_plot(tests)
    (OUT / "REPORT.md").write_text("# Domain analysis ds-v1\n\n"
        "P0–P3 complete. See rgb_summary.csv, skeleton_summary.csv, mmd_tests.csv and c2st.csv.\n\n"
        "MMD measures differences in these selected features. It does not establish a cause of classifier errors. "
        "p_raw/p_holm use one mean vector per subject where available, otherwise per recording group. "
        "Video permutation p-values are exploratory. Group permutation assumes exchangeable independent groups. "
        "GMDCSA has four subjects, so inference is coarse. Le2i/URFD lack reliable subject identities. "
        "Holm covers all 24 grouped tests. Balanced sampling quantiles and C2ST repeats describe stability, not independent replications. "
        "C2ST uses group-held-out predictions with preprocessing fit within training folds. "
        "Cache validity masks already threshold confidence, so missing and low-confidence rates are not independent measures.\n")


def matrix_plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    for ax, family in zip(axes, ("rgb", "skeleton", "skeleton_fall", "skeleton_nonfall")):
        matrix = np.zeros((4, 4))
        for r in rows:
            if r["family"] == family:
                i, j = DOMAINS.index(r["domain1"]), DOMAINS.index(r["domain2"])
                matrix[i, j] = matrix[j, i] = r["mmd2"]
        im = ax.imshow(matrix, cmap="Blues")
        ax.set_xticks(range(4), DOMAINS, rotation=60)
        ax.set_yticks(range(4), DOMAINS)
        ax.set_title(family)
        fig.colorbar(im, ax=ax, shrink=.6)
    fig.tight_layout()
    fig.savefig(OUT / "mmd_matrices.png", dpi=200)
    plt.close(fig)
