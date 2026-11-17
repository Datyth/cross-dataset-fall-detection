"""E2 paired-design invariants against frozen E1 metadata."""
import math
from collections import Counter

import numpy as np
import torch

from src import e2
from src.data import DOMAINS, load_manifest


def test_e2_split_covers_original_targets_without_group_leakage():
    cfg = e2.load_config()
    artifact = e2.make_split(cfg)
    rows = {r["video_id"]: r for r in load_manifest(cfg)}
    for fold in DOMAINS:
        parts = artifact["folds"][fold]
        all_tests = [vid for part in parts for vid in part["test"]]
        assert len(all_tests) == len(set(all_tests))
        assert len(all_tests) == sum(r["dataset"] == fold for r in rows.values())
        group_key = "subject_id" if fold in ("caucafall", "gmdcsa24") else (
            "scene_id" if fold == "le2i" else "recording_group_id")
        for part in parts:
            assert set(part["train"]).isdisjoint(part["test"])
            assert set(part["val"]).isdisjoint(part["test"])
            assert set(part["target_train"]).isdisjoint(part["test"])
            assert {int(rows[vid]["label"]) for vid in part["test"]} == {0, 1}
            assert {rows[vid][group_key] for vid in part["target_train"]}.isdisjoint(
                {rows[vid][group_key] for vid in part["test"]})


def test_e2_epoch_matches_e1_budget_and_excludes_test_and_validation():
    cfg = e2.load_config()
    artifact = e2.make_split(cfg)
    rows = {r["video_id"]: r for r in load_manifest(cfg)}
    for fold in DOMAINS:
        for part in artifact["folds"][fold]:
            order = e2.epoch_order(rows, part, seed=42, epoch=0)
            source_counts = Counter(rows[vid]["dataset"] for vid in part["source_train"])
            assert len(order) == 3 * max(source_counts.values())
            assert math.ceil(len(order) / 64) == e2.horizon(fold, 42)[1] // e2.horizon(fold, 42)[0]
            assert set(order).isdisjoint(set(part["test"]) | set(part["val"]))
            counts = Counter(rows[vid]["dataset"] for vid in order)
            assert max(counts.values()) - min(counts.values()) <= 1
            assert order == e2.epoch_order(rows, part, seed=42, epoch=0)


def test_e2_report_pairs_every_original_prediction(tmp_path, monkeypatch):
    cfg = e2.load_config()
    artifact = e2.make_split(cfg)

    class DummyModel:
        def __init__(self, **_kwargs):
            pass

        def to(self, _device):
            return self

        def load_state_dict(self, _state):
            pass

        def eval(self):
            return self

    monkeypatch.setattr(e2, "paths", lambda: (tmp_path, tmp_path))
    monkeypatch.setattr(e2, "verify_run", lambda *_args: {})
    monkeypatch.setattr(e2.torch, "load", lambda *_args, **_kwargs: {"model": {}, "threshold": .5})
    monkeypatch.setattr(e2, "CTRGCN", DummyModel)
    monkeypatch.setattr(e2, "load_video", lambda *_args: (np.zeros((1, 3, 64, 17, 1), np.float32), None))
    monkeypatch.setattr(e2, "video_logit", lambda *_args: torch.tensor(0.))
    monkeypatch.setattr(e2, "digest", lambda *_args: "synthetic-checkpoint")
    monkeypatch.setattr(e2, "_paired_interval", lambda *_args: (0., 0.))
    cfg["train"]["device"] = "cpu"
    result = e2.report(cfg, artifact)
    assert result["predictions"] == 1560
    predictions = e2.read_csv(tmp_path / "predictions.csv")
    assert len(predictions) == len({(r["fold"], r["seed"], r["video_id"]) for r in predictions}) == 1560
    assert len(e2.read_csv(tmp_path / "metrics.csv")) == 12
    assert "±" in (tmp_path / "REPORT.md").read_text()
