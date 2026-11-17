import numpy as np
import torch

from src import pdi_train
from src.model import CTRGCN
from src.pdi import Predictor, augment_past, losses, make_target, pair_candidates, update_ema


def test_pair_rejects_padding_and_missing_future():
    windows = np.zeros((3, 3, 64, 17, 1), np.float32)
    windows[:, 2] = 1
    masks = np.ones((3, 64), bool)
    masks[1, 63] = False
    windows[2, 2, 48:53] = 0
    assert pair_candidates(windows, masks) == [0]


def test_augmentation_preserves_missing_joints():
    x = np.ones((3, 48, 17, 1), np.float32)
    x[:, :, 0] = 0
    result = augment_past(x, np.random.default_rng(42), .01, .1)
    assert np.all(result[:, :, 0] == 0)
    assert np.any(result[2] > 0)
    assert not np.array_equal(result, x)


def test_pdi_gradients_and_ema_target():
    torch.set_num_threads(2)
    model = CTRGCN(base_channels=8)
    target = make_target(model)
    predictor = Predictor(32)
    past = torch.randn(2, 3, 48, 17, 1)
    future = torch.randn(2, 3, 16, 17, 1)
    prediction, consistency, _ = losses(model, target, predictor, past, past.clone(), future, True)
    assert torch.isfinite(prediction + consistency)
    assert abs(float(consistency.detach())) < 1e-5
    (prediction + consistency).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.blocks.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in predictor.parameters())
    assert all(p.grad is None for p in target.parameters())
    old = next(target.parameters()).clone()
    with torch.no_grad():
        next(model.parameters()).add_(1)
    update_ema(target, model, .99)
    torch.testing.assert_close(next(target.parameters()), old + .01)


def test_pdi_train_checkpoint_and_resume_on_synthetic_cache(tmp_path, monkeypatch):
    base, pdi = pdi_train.configs()
    base["model"]["base_channels"] = 8
    base["train"].update(device="cpu", videos_per_batch=6, window_chunk=8,
                         bn_calibration_batch_size=12)
    pdi.update(max_epochs=1, aux_batch_size=2, results_dir=str(tmp_path))
    rows = []
    for domain in ("caucafall", "gmdcsa24", "le2i", "urfd"):
        for label in (0, 1):
            for split in ("train", "val"):
                rows.append({"video_id": f"{domain}-{label}-{split}", "dataset": domain,
                             "label": str(label)})
    partition = {"train": [r["video_id"] for r in rows if r["dataset"] != "caucafall" and r["video_id"].endswith("train")],
                 "val": [r["video_id"] for r in rows if r["dataset"] != "caucafall" and r["video_id"].endswith("val")],
                 "target": [r["video_id"] for r in rows if r["dataset"] == "caucafall"]}
    monkeypatch.setattr(pdi_train, "load_splits", lambda _cfg: {"split_hash": "synthetic", "folds": {"caucafall": partition}})
    monkeypatch.setattr(pdi_train, "load_manifest", lambda _cfg: rows)

    def load(_cfg, row):
        windows = np.ones((1, 3, 64, 17, 1), np.float32)
        windows[:, 0] *= int(row["label"])
        return windows, np.ones((1, 64), bool)

    monkeypatch.setattr(pdi_train, "load_video", load)
    monkeypatch.setattr("src.train.load_video", load)
    for method in pdi_train.METHODS:
        result = pdi_train.train(base, pdi, method, "caucafall", 42)
        assert result["epochs"] == 1
        assert pdi_train.verify_run(base, pdi, method, "caucafall", 42) == result
        assert pdi_train.train(base, pdi, method, "caucafall", 42) == result
    paired = pdi_train.report(base, pdi, "caucafall", 42)
    assert paired["p3"]["n"] == paired["p4"]["n"] == 4
    assert (tmp_path / "paired/caucafall/42/metrics.json").is_file()
