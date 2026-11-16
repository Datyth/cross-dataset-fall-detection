"""Verify the actual local runtime, weights, classifier and optional pose execution."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from service.baseline_runtime import runtime_status, weight_status, BaselineInferencer
from service.inference import ModelRegistry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--initialize", action="store_true")
    args = parser.parse_args()
    report = {"runtime": runtime_status(), "weights": weight_status(ROOT / "data/model_weights")}
    registry = ModelRegistry()
    report["models"] = registry.public()
    if not all(report[k]["available"] for k in ("runtime", "weights")):
        print(json.dumps(report, indent=2))
        return 1
    ck = registry.get("urfd_2026_erm")
    if args.initialize:
        import copy
        import cv2
        import matplotlib
        import torch
        torch.set_num_threads(4)
        cfg = copy.deepcopy(ck["config"])
        cfg["paths"]["data"] = str(ROOT / "data")
        inferencer = BaselineInferencer(cfg, "cpu")
        image = cv2.imread(str(Path(matplotlib.get_data_path()) / "sample_data/grace_hopper.jpg"))
        predictions = list(inferencer([image], return_vis=False, bbox_thr=.3))[0]["predictions"][0]
        if not predictions or len(predictions[0]["keypoints"]) != 17:
            raise RuntimeError("Original pose smoke test did not return COCO-17")
        report["initialized"] = True
        report["extractor"] = inferencer.provenance
        (ROOT / ".baseline-runtime.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
