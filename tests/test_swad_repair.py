from collections import Counter

import numpy as np
import pytest
import torch

from src.swad_repair import correct_average, legacy_membership, recovery_plan, replay_prefix
from src.train import LossValley, StateAverage, verify_swad_summary


SETTINGS = {"n_converge": 3, "n_tolerance": 6, "tolerance_ratio": .3}
LOSSES = [.5, .4, .3, .2, .19, .18, .15, .16, .17, .16, .17, .18, .19, .2]


def trace_for(losses, counts=None):
    counts = counts or [8] * len(losses)
    step, trace = 0, []
    for i, loss in enumerate(losses):
        step += counts[i]
        trace.append({"epoch": i + 1, "step": step, "source_val_macro_bce": loss})
    return trace


def test_dense_valley_never_counts_a_state_twice_and_finish_is_idempotent():
    trace = trace_for(LOSSES)
    valley = LossValley(**SETTINGS)
    for i, row in enumerate(trace):
        segment = StateAverage()
        basis = torch.zeros(len(trace)); basis[i] = 1
        segment.add({"w": basis}, 8, row["step"] - 7, row["step"])
        valley.update(segment, row["source_val_macro_bce"])
    average = valley.finish()
    expected = torch.zeros(len(trace)); expected[4:] = .1  # epochs 5..14 once each
    torch.testing.assert_close(average.state["w"], expected)
    assert average.count == 80
    assert average.start_step == 33
    assert average.end_step == 112
    before = average.state["w"].clone()
    assert valley.finish().count == 80
    torch.testing.assert_close(before, average.state["w"], rtol=0, atol=0)
    verify_swad_summary(valley.summary())


@pytest.mark.parametrize("variable_counts", [False, True])
def test_prefix_correction_matches_full_fresh_average(variable_counts):
    counts = [2 + i % 3 for i in range(len(LOSSES))] if variable_counts else [8] * len(LOSSES)
    trace = trace_for(LOSSES, counts)
    plan = recovery_plan(trace, SETTINGS)
    selected, _first, _threshold, _dead = legacy_membership(trace, SETTINGS)
    assert Counter(selected)[6] == 2  # old code duplicates epoch 7
    rng = np.random.default_rng(2026)
    states = [{"w": torch.from_numpy(rng.normal(size=128).astype(np.float32))} for _ in trace]
    old_mean, n = None, 0
    for i in selected:
        if old_mean is None:
            old_mean = {k: v.clone() for k, v in states[i].items()}; n = counts[i]
        else:
            for k in old_mean:
                old_mean[k].add_((states[i][k] - old_mean[k]) * (counts[i] / (n + counts[i])))
            n += counts[i]
    removed = [(states[e - 1], copies * counts[e - 1]) for e, copies in plan["removed_epochs"].items()]
    actual = correct_average(old_mean, n, plan["corrected"]["averaged_states"], removed)
    fixed = StateAverage()
    for i in sorted(set(selected)):
        fixed.add(states[i], counts[i], trace[i]["step"] - counts[i] + 1, trace[i]["step"])
    torch.testing.assert_close(actual["w"], fixed.state["w"], rtol=1e-5, atol=2e-6)
    assert fixed.count == plan["corrected"]["averaged_states"]
    assert plan["replay_until_epoch"] == 7


def test_unchanged_average_is_copied_exactly_and_bad_counts_fail():
    state = {"w": torch.randn(32)}
    same = correct_average(state, 10, 10, [])
    torch.testing.assert_close(same["w"], state["w"], rtol=0, atol=0)
    assert same["w"].data_ptr() != state["w"].data_ptr()
    with pytest.raises(ValueError, match="counts"):
        correct_average(state, 10, 9, [])


def test_averages_reject_overlap_and_legacy_summary_even_if_count_matches():
    average = StateAverage(); average.add({"w": torch.ones(1)}, 2, 1, 2)
    with pytest.raises(ValueError, match="Overlapping"):
        average.add({"w": torch.ones(1)}, 2, 2, 3)
    with pytest.raises(ValueError, match="range"):
        StateAverage().add({"w": torch.ones(1)}, 4, 1, 2)
    with pytest.raises(ValueError, match="repair-swad"):
        verify_swad_summary({"valid": True, "start_step": 1, "end_step": 8, "averaged_states": 8})


def test_replay_rejects_target_leakage_before_loading_data(monkeypatch):
    from src import swad_repair
    monkeypatch.setattr(swad_repair, "load_splits", lambda cfg: {"folds": {"heldout": {
        "train": ["target"], "val": ["val"], "target": ["target"]}}})
    monkeypatch.setattr(swad_repair, "load_manifest", lambda cfg: [
        {"video_id": "target", "dataset": "heldout"}, {"video_id": "val", "dataset": "source"}])
    with pytest.raises(AssertionError, match="leakage"):
        replay_prefix({}, "heldout", 42, [], {7: 1}, {})


def test_report_preflight_rejects_uncorrected_run_before_target_inference(tmp_path, monkeypatch):
    from src import report
    called = []

    def bad(*args):
        raise ValueError("Legacy SWAD: repair-swad required")

    monkeypatch.setattr(report, "validate_completed_run", bad)
    monkeypatch.setattr(report, "load_manifest", lambda cfg: called.append("manifest"))
    cfg = {"paths": {"results": str(tmp_path)}, "train": {"seeds": [42]}}
    with pytest.raises(ValueError, match="repair-swad"):
        report.report(cfg, force=True)
    assert called == []
    assert list(tmp_path.iterdir()) == []


def test_corrected_report_pipeline_writes_complete_outputs(tmp_path):
    """Synthetic end-to-end artifacts, not experimental/scientific results."""
    import csv
    import json
    from pathlib import Path
    import pandas as pd
    import yaml
    from src.data import DOMAINS, canonical_hash, digest, make_splits
    from src.model import CTRGCN
    from src.report import report, report_preflight
    from src.train import _save_checkpoint, implementation_hash

    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "config.yaml").read_text())
    data, out = tmp_path / "data", tmp_path / "results"
    data.mkdir()
    cfg["paths"] = {"data": str(data), "results": str(out)}
    cfg["model"]["base_channels"] = 8
    cfg["train"].update(device="cpu", seeds=[42, 123])
    cfg["evaluation"]["bootstrap_resamples"] = 4
    rows = []
    for domain in DOMAINS:
        folder = data / "skeletons" / domain; folder.mkdir(parents=True)
        for subject in range(4):
            for label in (0, 1):
                vid = f"{domain}-s{subject}-{label}"
                rows.append({"video_id": vid, "dataset": domain, "label": label,
                             "subject_id": f"s{subject}", "recording_group_id": vid, "duration_s": 4.})
                np.savez_compressed(folder / f"{vid}.npz",
                                    xy=np.full((64, 17, 2), 100. + label * 50, np.float32),
                                    score=np.ones((64, 17), np.float32),
                                    valid_joint=np.ones((64, 17), bool),
                                    valid_frame=np.ones(64, bool), image_size=np.array([320, 240]))
    with (data / "manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    splits = make_splits(cfg, 2026)
    summary = {"averaging_version": 2, "valid": True, "converge_step": 3,
               "start_step": 1, "end_step": 3, "averaged_states": 3, "dead_valley": False}
    for fold in DOMAINS:
        for seed in cfg["train"]["seeds"]:
            directory = out / "runs" / f"{fold}_{seed}"; directory.mkdir(parents=True)
            torch.manual_seed(seed)
            model = CTRGCN(**cfg["model"]).eval()
            hashes = {}
            for method, name in (("erm", "best_erm.pt"), ("swad", "swad.pt")):
                hashes[method] = _save_checkpoint(directory / name, model, cfg, splits["split_hash"],
                                                  seed, fold, method, .5, .7, dict(summary))
            info = {"fold": fold, "seed": seed, "config_hash": canonical_hash(cfg),
                    "split_hash": splits["split_hash"], "implementation_hash": implementation_hash(),
                    "erm_checkpoint_hash": hashes["erm"],
                    "swad": {**summary, "checkpoint_hash": hashes["swad"]}}
            (directory / "provenance.json").write_text(json.dumps(info))
    assert report_preflight(cfg)["count"] == 8
    assert report(cfg) == len(rows) * 2 * 2
    assert len(pd.read_csv(out / "metrics.csv")) == 16
    assert len(pd.read_csv(out / "source_validation.csv")) == 16
    for name in ("REPORT.md", "target_metrics.png", "source_vs_target.png", "confusion_matrices.png"):
        assert (out / name).stat().st_size > 0
    assert "Averaging version 2" in (out / "REPORT.md").read_text()
    previous = digest(out / "predictions.csv")
    with pytest.raises(RuntimeError, match="predictions.csv exists"):
        report(cfg)
    assert digest(out / "predictions.csv") == previous
