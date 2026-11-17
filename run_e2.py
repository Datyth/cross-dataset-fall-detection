#!/usr/bin/env python3
"""Prepare, validate, train and report the paired E2 experiment."""
import argparse
import json

from src import e2


def main():
    parser = argparse.ArgumentParser(description="E2 paired domain-exposure ERM experiment")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare", help="freeze four target-test parts per domain")
    check = sub.add_parser("preflight", help="verify E1 lineage and all pose caches")
    check.add_argument("--cpu", action="store_true", help="skip CUDA availability check for CPU tests")
    train = sub.add_parser("train", help="run all 48 trajectories, resuming interrupted work")
    train.add_argument("--fold", choices=(*e2.DOMAINS, "all"), default="all")
    train.add_argument("--part", type=int, choices=e2.PARTS)
    train.add_argument("--seed", type=int, choices=e2.SEEDS)
    train.add_argument("--stop-after-epoch", type=int, help="smoke test; resume the same run later")
    sub.add_parser("report", help="verify all runs, infer target videos and write paired report")
    args = parser.parse_args()
    cfg = e2.load_config()
    if args.command == "prepare":
        result = {"split_hash": e2.split(cfg, write=True)["split_hash"]}
    elif args.command == "preflight":
        result = e2.preflight(cfg, check_cuda=not args.cpu)
    elif args.command == "train":
        artifact = e2.split(cfg)
        folds = e2.DOMAINS if args.fold == "all" else (args.fold,)
        parts = e2.PARTS if args.part is None else (args.part,)
        seeds = e2.SEEDS if args.seed is None else (args.seed,)
        result = []
        for fold in folds:
            for part in parts:
                for seed in seeds:
                    info = e2.train_run(cfg, artifact, fold, part, seed, args.stop_after_epoch)
                    result.append({k: info[k] for k in ("fold", "part", "seed")})
    else:
        artifact = e2.split(cfg)
        result = e2.report(cfg, artifact)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
