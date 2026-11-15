import numpy as np
import pytest
import torch

import csv
from pathlib import Path

from src.data import (DOMAINS, resample_indices, window_starts, make_windows,
                      _choose_group_split, make_splits, load_splits, load_video)
from src.model import COCO_JOINTS, COCO_EDGES, coco_adjacency, CTRGCN, mil_logit
from src.train import (LossValley, StateAverage, select_threshold, set_seed,
                       recalibrate_bn, rng_state, restore_rng, verify_implementation)
from src.train import video_batch_logits, video_logit
from src.report import video_metrics, macro_seed_metrics


def test_coco_graph():
    assert len(COCO_JOINTS) == 17
    assert COCO_JOINTS[11] == "left_hip" and COCO_JOINTS[16] == "right_ankle"
    assert (11, 13) in COCO_EDGES and (12, 14) in COCO_EDGES
    assert coco_adjacency().shape == (3, 17, 17)
    reached = {0}
    while True:
        new = reached | {v for u, v in COCO_EDGES if u in reached} | {u for u, v in COCO_EDGES if v in reached}
        if new == reached:
            break
        reached = new
    assert len(reached) == 17


def test_resample_and_windows():
    grid, idx = resample_indices([0, 1/30, 2/30, 3/30, 4/30], 15, .01)
    np.testing.assert_array_equal(idx, [0, 2, 4])
    with pytest.raises(ValueError):
        resample_indices([0, 0, .1])
    assert window_starts(20) == [0]
    assert window_starts(64) == [0]
    assert window_starts(101) == [0, 32, 37]
    x = np.ones((20, 17, 3), np.float32)
    windows, masks = make_windows(x)
    assert windows.shape == (1, 3, 64, 17, 1)
    assert masks[0].sum() == 20
    assert np.all(windows[0, :, -1] == windows[0, :, 19])
    x[-1, :, 2] = 0
    x[-2, :, 0] = 7
    padded, mask = make_windows(x)
    assert mask[0].sum() == 20
    assert padded[0, 0, -1, 0, 0] == 7  # last visible pose, not an empty last frame


def test_model_and_mil_gradient():
    set_seed(42)
    model = CTRGCN(base_channels=8)
    out = model(torch.randn(2, 3, 64, 17, 1))
    assert out.shape == (2,)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(mil_logit(out), torch.tensor(1.))
    loss.backward()
    assert torch.isfinite(loss)
    assert model.fc.weight.grad.abs().sum() > 0
    logits = torch.tensor([1., 3., 2.], requires_grad=True)
    score = mil_logit(logits)
    assert score.item() == 2.5
    score.backward()
    assert logits.grad.tolist() == [0., .5, .5]


def test_binary_head_starts_without_saturated_predictions():
    set_seed(42)
    model = CTRGCN()
    x = torch.randn(2, 3, 64, 17, 1)
    with torch.no_grad():
        logits = model(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([0., 1.]))
    assert torch.isfinite(logits).all()
    assert logits.abs().max() < 2
    assert loss < 1


def test_mixed_video_forward_keeps_every_window_and_video_mil():
    class Echo(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.))
            self.seen = []

        def forward(self, x):
            self.seen.append(x[:, 0, 0, 0, 0].tolist())
            return x[:, 0, 0, 0, 0] * self.scale

    model = Echo()
    videos = [np.broadcast_to(np.array(v, np.float32)[:, None, None, None, None],
                               (len(v), 3, 4, 17, 1)).copy() for v in ([1, 5, 2], [9, 3])]
    mixed = video_batch_logits(model, videos, "cpu", 2)
    assert model.seen == [[1., 9.], [5., 3.], [2.]]
    torch.testing.assert_close(mixed, torch.tensor([3.5, 6.]))
    mixed.sum().backward()
    assert model.scale.grad.item() == 9.5
    # In eval mode grouping/chunk size must not change video probabilities.
    set_seed(42)
    real = CTRGCN(base_channels=8).eval()
    with torch.no_grad():
        mixed = video_batch_logits(real, videos, "cpu", 4)
        separate = torch.stack([video_logit(real, v, "cpu", 2) for v in videos])
    torch.testing.assert_close(mixed, separate, rtol=1e-5, atol=1e-6)


def test_bn_uses_only_source_train_and_preserves_weights(monkeypatch):
    from src import train
    cfg = {"train": {"bn_calibration_batch_size": 8, "bn_calibration_seed": 2026}}
    train_ids = [f"v{i}" for i in range(16)]
    rows = {vid: {"video_id": vid} for vid in train_ids}
    seen = []

    def load(_cfg, row):
        vid = row["video_id"]
        seen.append(vid)
        value = -1 if int(vid[1:]) % 2 == 0 else 1
        return np.full((1, 3, 4, 17, 1), value, np.float32), None

    monkeypatch.setattr(train, "load_video", load)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.BatchNorm1d(3 * 4 * 17),
                                torch.nn.Dropout(.9), torch.nn.Linear(3 * 4 * 17, 1))
    weights = {k: p.clone() for k, p in model.named_parameters()}
    dropout_training = []
    hook = model[2].register_forward_pre_hook(lambda m, _x: dropout_training.append(m.training))
    count = recalibrate_bn(model, cfg, rows, train_ids, "cpu", forbidden_ids=["val", "target"])
    hook.remove()
    assert count == 2
    assert set(seen) == set(train_ids)
    assert not any(dropout_training)
    assert not model.training
    assert model[1].momentum == .1
    assert model[1].running_var.min() > .5  # includes BETWEEN-video variation
    first = model[1].running_var.clone()
    for k, p in model.named_parameters():
        torch.testing.assert_close(p, weights[k], rtol=0, atol=0)
    recalibrate_bn(model, cfg, rows, list(reversed(train_ids)), "cpu")
    torch.testing.assert_close(first, model[1].running_var, rtol=0, atol=0)
    seen.clear()
    with pytest.raises(AssertionError, match="Validation/target"):
        recalibrate_bn(model, cfg, rows, train_ids + ["target"], "cpu", forbidden_ids=["target"])
    assert seen == []


def test_resume_restores_rng_and_rejects_old_implementation():
    import random
    set_seed(2026)
    state = rng_state()
    expected = (random.random(), np.random.rand(), torch.rand(4))
    random.random(); np.random.rand(); torch.rand(9)
    restore_rng(state)
    actual = (random.random(), np.random.rand(), torch.rand(4))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
    with pytest.raises(ValueError, match="archive"):
        verify_implementation({}, "new")
    with pytest.raises(ValueError, match="archive"):
        verify_implementation({"implementation_hash": "old"}, "new")
    verify_implementation({"implementation_hash": "new"}, "new")


def test_interrupted_training_matches_uninterrupted_and_finalizes(tmp_path, monkeypatch):
    import copy
    from src import train

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.BatchNorm1d(3 * 4 * 17),
                                           torch.nn.Dropout(.3), torch.nn.Linear(3 * 4 * 17, 1))

        def forward(self, x):
            return self.net(x).squeeze(-1)

    rows, train_ids, val_ids = [], [], []
    for domain in ("a", "b", "c"):
        for split, ids in (("train", train_ids), ("val", val_ids)):
            for label in (0, 1):
                vid = f"{domain}-{split}-{label}"
                rows.append({"video_id": vid, "dataset": domain, "label": str(label)})
                ids.append(vid)
                path = tmp_path / "skeletons" / domain / f"{vid}.npz"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
    artifact = {"split_hash": "fixed", "folds": {"heldout": {
        "train": train_ids, "val": val_ids, "target": ["target-never-read"]}}}

    def load(_cfg, row):
        assert row["video_id"] != "target-never-read"
        value = float(row["label"]) * 2 - 1
        return np.full((2, 3, 4, 17, 1), value, np.float32), None

    monkeypatch.setattr(train, "load_splits", lambda _cfg: artifact)
    monkeypatch.setattr(train, "load_manifest", lambda _cfg: rows)
    monkeypatch.setattr(train, "load_video", load)
    monkeypatch.setattr(train, "_model", lambda _cfg, device: TinyModel().to(device))
    cfg = {"paths": {"data": str(tmp_path), "results": str(tmp_path / "continuous")}, "model": {},
           "train": {"videos_per_batch": 2, "device": "cpu", "learning_rate": .003,
                     "weight_decay": .0001, "gradient_accumulation": 2, "window_chunk": 4,
                     "grad_clip": 1., "max_epochs": 3, "patience_after_swad_convergence": 100,
                     "bn_calibration_batch_size": 6, "bn_calibration_seed": 2026},
           "swad": {"n_converge": 3, "n_tolerance": 6, "tolerance_ratio": .3,
                    "validation_interval_epochs": 1},
           "evaluation": {"threshold_grid": [.25, .5, .75]}}
    continuous = train.train_fold(cfg, "heldout", 42, stop_after_epoch=2)
    assert continuous["incomplete"]
    original = torch.load(continuous["checkpoint"], weights_only=False)
    resumed_cfg = copy.deepcopy(cfg)
    resumed_cfg["paths"]["results"] = str(tmp_path / "resumed")
    first = train.train_fold(resumed_cfg, "heldout", 42, stop_after_epoch=1)
    with pytest.raises(RuntimeError, match="use --resume"):
        train.train_fold(resumed_cfg, "heldout", 42, stop_after_epoch=2)
    resumed = train.train_fold(resumed_cfg, "heldout", 42, resume=True, stop_after_epoch=2)
    actual = torch.load(resumed["checkpoint"], weights_only=False)
    assert original["step"] == actual["step"] == 4  # two updates per epoch, including partial group
    assert original["best"] == actual["best"]
    for k in original["model"]:
        torch.testing.assert_close(actual["model"][k], original["model"][k], rtol=0, atol=0)
    finished = train.train_fold(resumed_cfg, "heldout", 42, resume=True)
    assert "erm_checkpoint_hash" in finished
    ck = Path(resumed_cfg["paths"]["results"]) / "runs" / "heldout_42" / "best_erm.pt"
    saved = torch.load(ck, weights_only=False)
    assert saved["implementation_hash"] == train.implementation_hash()
    assert finished == train.train_fold(resumed_cfg, "heldout", 42, resume=True)
    # Constant weights/loss produce a valid valley; exercise the real SWAD
    # checkpoint and source-only BN path without relying on lucky convergence.
    swad_cfg = copy.deepcopy(cfg)
    swad_cfg["paths"]["results"] = str(tmp_path / "swad")
    swad_cfg["train"]["learning_rate"] = 0.
    paired = train.train_fold(swad_cfg, "heldout", 42)
    assert paired["swad"]["valid"]
    run = Path(swad_cfg["paths"]["results"]) / "runs" / "heldout_42"
    erm = torch.load(run / "best_erm.pt", weights_only=False)
    swad = torch.load(run / "swad.pt", weights_only=False)
    assert swad["method"] == "swad"
    for k in erm["model"]:
        torch.testing.assert_close(swad["model"][k], erm["model"][k], rtol=0, atol=0)


def test_ctr_gcn_overfits_two_synthetic_videos():
    set_seed(42)
    model = CTRGCN(base_channels=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
    x = torch.zeros(2, 3, 16, 17, 1)
    x[1, 0] = .7
    x[1, 2] = 1
    labels = torch.tensor([0., 1.])
    for _ in range(80):
        model.train(); optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), labels)
        loss.backward(); optimizer.step()
        if loss.item() < .02:
            break
    assert loss.item() < .05


def test_group_split_and_threshold():
    rows = []
    for subject in range(4):
        for label in (0, 1):
            rows.append({"video_id": f"v{subject}-{label}", "subject_id": f"s{subject}",
                         "recording_group_id": f"v{subject}-{label}", "label": str(label)})
    train, val, key = _choose_group_split(rows, 2026, .2)
    assert key == "subject_id"
    assert set(r["subject_id"] for r in train).isdisjoint(r["subject_id"] for r in val)
    assert {int(r["label"]) for r in train} == {0, 1}
    assert {int(r["label"]) for r in val} == {0, 1}
    threshold = select_threshold([("a", 0, .2), ("b", 1, .4), ("c", 1, .8)], [.3, .5])
    assert threshold == .3


def test_fourfold_target_exclusion_and_frozen_hash(tmp_path):
    data = tmp_path / "data"; data.mkdir()
    rows = []
    for domain in DOMAINS:
        for subject in range(4):
            for label in (0, 1):
                vid = f"{domain}-{subject}-{label}"
                rows.append({"video_id": vid, "dataset": domain, "subject_id": f"person-{subject}",
                             "recording_group_id": vid, "label": label})
    with open(data / "manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    cfg = {"paths": {"data": str(data)}, "train": {"val_fraction": .25}}
    artifact = make_splits(cfg, 2026)
    assert load_splits(cfg) == artifact
    for domain in DOMAINS:
        fold = artifact["folds"][domain]
        assert all(x.startswith(domain + "-") for x in fold["target"])
        assert not set(fold["target"]) & (set(fold["train"]) | set(fold["val"]))
    with pytest.raises(ValueError, match="Immutable"):
        make_splits(cfg, 42)


def test_missing_joint_zero_and_frame_normalization(tmp_path):
    cfg = {"paths": {"data": str(tmp_path)}, "preprocess": {"window_frames": 64, "stride_frames": 32}}
    folder = tmp_path / "skeletons" / "d"; folder.mkdir(parents=True)
    xy = np.zeros((2, 17, 2), np.float32); xy[:, :, 0] = 100; xy[:, :, 1] = 50
    valid = np.ones((2, 17), bool); valid[0, 0] = False
    np.savez_compressed(folder / "v.npz", xy=xy, score=np.ones((2, 17), np.float32),
                        valid_joint=valid, image_size=np.array([200, 100]))
    windows, mask = load_video(cfg, {"dataset": "d", "video_id": "v"})
    assert windows.shape == (1, 3, 64, 17, 1)
    assert mask.sum() == 2
    assert windows[0, :, 0, 0, 0].tolist() == [0., 0., 0.]
    assert windows[0, :, 0, 1, 0].tolist() == [0., 0., 1.]


def test_video_metrics():
    rows = [{"video_id": str(i), "y_true": y, "y_prob": p, "threshold": .5}
            for i, (y, p) in enumerate([(1, .9), (1, .8), (0, .1), (0, .7)])]
    m = video_metrics(rows)
    assert m["tp"] == 2 and m["fp"] == 1 and m["fn"] == 0
    assert m["f1"] == pytest.approx(.8)
    assert m["pr_auc"] == pytest.approx(1.)


def test_macro_domain_equal_weight():
    metrics = [{"fold": domain, "seed": 42, "method": "erm", "f1": score}
               for domain, score in zip(DOMAINS, (.0, .0, .0, 1.0))]
    assert macro_seed_metrics(metrics, "erm", "f1", [42]).tolist() == [.25]
    with pytest.raises(ValueError):
        macro_seed_metrics(metrics[:-1], "erm", "f1", [42])


def test_swad_dense_average_and_valley():
    weighted = StateAverage()
    weighted.add({"w": torch.tensor([2.])}, 2, 1, 2)
    weighted.add({"w": torch.tensor([8.])}, 1, 3, 3)
    assert weighted.count == 3 and weighted.state["w"].item() == pytest.approx(4.)
    def segment(value, step):
        result = StateAverage()
        result.add({"w": torch.tensor([value])}, start_step=step, end_step=step)
        return result
    valley = LossValley(3, 6, .3)
    for step, loss in enumerate([.5, .4, .3, .31, .32, .8, .9, 1.0], 1):
        valley.update(segment(float(step), step), loss)
    final = valley.finish()
    assert final is not None
    assert final.count >= 1
    assert valley.summary()["converge_step"] == 3
    assert final.start_step <= final.end_step


def test_seed_replay():
    set_seed(2026)
    a = torch.rand(3)
    set_seed(2026)
    torch.testing.assert_close(a, torch.rand(3))
