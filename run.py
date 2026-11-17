#!/usr/bin/env python3
"""E1 reproducible command-line entry point."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description="E1 four-domain skeleton fall-detection benchmark")
    parser.add_argument("--config", default="config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("download")
    d.add_argument("--dataset", choices=("caucafall", "gmdcsa24", "le2i", "urfd", "all"), default="all")
    d.add_argument("--source", choices=("official", "mirror"), default="official")
    d.add_argument("--dry-run", action="store_true")
    d.add_argument("--resume", action="store_true")
    d.add_argument("--connections", type=int, default=1, help="parallel HTTP ranges for large Le2i mirror")
    i = sub.add_parser("inspect")
    i.add_argument("--check-splits", action="store_true")
    e = sub.add_parser("extract")
    e.add_argument("--dataset", choices=("caucafall", "gmdcsa24", "le2i", "urfd", "all"), default="all")
    e.add_argument("--device", default="cuda")
    e.add_argument("--limit", type=int)
    e.add_argument("--video-id", help="extract one exact manifest recording")
    e.add_argument("--resume", action="store_true")
    e.add_argument("--force", action="store_true", help="replace existing pose caches for selected videos")
    p = sub.add_parser("preview", help="save deterministic pose overlay contact sheets for cached videos")
    p.add_argument("--dataset", choices=("caucafall", "gmdcsa24", "le2i", "urfd", "all"), default="all")
    p.add_argument("--limit", type=int, default=5)
    s = sub.add_parser("split")
    s.add_argument("--seed", type=int, default=2026)
    t = sub.add_parser("train")
    t.add_argument("--method", choices=("erm", "swad"), default="erm")
    t.add_argument("--fold", choices=("caucafall", "gmdcsa24", "le2i", "urfd", "all"), default="all")
    t.add_argument("--seeds", default="42,123,2026")
    t.add_argument("--smoke-test", action="store_true")
    t.add_argument("--resume", action="store_true")
    t.add_argument("--stop-after-epoch", type=int,
                   help="stop at this epoch checkpoint without finalizing; resume with the same config")
    repair = sub.add_parser("repair-swad", help="correct legacy SWAD using a verified short source-only replay")
    repair.add_argument("--fold", choices=("caucafall", "gmdcsa24", "le2i", "urfd", "all"), default="all")
    repair.add_argument("--seeds", default="42,123,2026")
    repair.add_argument("--resume", action="store_true", help="verify and skip already corrected runs")
    repair.add_argument("--dry-run", action="store_true", help="validate old artifacts and print recovery plans without writing")
    predict = sub.add_parser("predict-video", help="classify one RGB video with a saved checkpoint")
    predict.add_argument("--video", required=True, help="path to an RGB video")
    predict.add_argument("--checkpoint", required=True, help="path to best_erm.pt or a corrected swad.pt")
    predict.add_argument("--device", default="cuda")
    predict.add_argument("--output", help="write an annotated MP4 with person box, skeleton and prediction")
    r = sub.add_parser("report")
    r.add_argument("--force", action="store_true")
    r.add_argument("--check-only", action="store_true", help="validate all checkpoints without target inference or output writes")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    print(json.dumps({"command": args.command, "config": args.config, "settings": vars(args)}, sort_keys=True), flush=True)
    if args.command == "download":
        from src.data import download
        download(cfg, args.dataset, args.dry_run, args.source, args.resume, args.connections)
    elif args.command == "inspect":
        from src.data import inspect, load_splits, write_status
        rows, problems = inspect(cfg)
        print("manifest", Path(cfg["paths"]["data"]) / "manifest.csv", "rows", len(rows))
        for domain, count in sorted(Counter((r["dataset"], r["label"]) for r in rows).items()):
            print(domain, count)
        if args.check_splits:
            print("split hash", load_splits(cfg)["split_hash"])
        status = write_status(cfg, rows, problems)
        print("status", status)
        if problems:
            print("Inspection issues:", *problems, sep="\n- ")
    elif args.command == "extract":
        from src.pose import extract
        print(extract(cfg, args.dataset, args.device, args.limit, args.resume, args.force, args.video_id))
    elif args.command == "preview":
        from src.pose import preview
        print(*preview(cfg, args.dataset, args.limit), sep="\n")
    elif args.command == "split":
        from src.data import make_splits
        print("Frozen split hash", make_splits(cfg, args.seed)["split_hash"])
    elif args.command == "train":
        from src.train import synthetic_smoke, train_fold
        if args.smoke_test:
            print(synthetic_smoke(cfg, "cpu"))
        else:
            folds = tuple(cfg["datasets"]) if args.fold == "all" else (args.fold,)
            seeds = [int(x) for x in args.seeds.split(",")]
            for fold in folds:
                for seed in seeds:
                    print(train_fold(cfg, fold, seed, resume=args.resume,
                                     stop_after_epoch=args.stop_after_epoch))
    elif args.command == "repair-swad":
        from src.swad_repair import repair_swad
        folds = tuple(cfg["datasets"]) if args.fold == "all" else (args.fold,)
        for fold in folds:
            for seed in [int(x) for x in args.seeds.split(",")]:
                print(json.dumps(repair_swad(cfg, fold, seed, args.resume, args.dry_run)), flush=True)
    elif args.command == "predict-video":
        from src.predict import predict_video
        predict_video(cfg, args.video, args.checkpoint, args.device, args.output)
    elif args.command == "report":
        from src.report import report, report_preflight
        if args.check_only:
            print(json.dumps(report_preflight(cfg)))
        else:
            print("Target prediction rows", report(cfg, args.force))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"E1 failed: {exc}", file=sys.stderr)
        raise
