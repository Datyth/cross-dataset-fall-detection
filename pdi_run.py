#!/usr/bin/env python3
"""Pilot comparison of future prediction with and without PDI consistency."""
import argparse
import json

import torch

from src.pdi import Predictor, losses, make_target, update_ema
from src import pdi_train


def smoke():
    from src.model import CTRGCN
    torch.set_num_threads(2)
    online = CTRGCN(base_channels=8)
    target = make_target(online)
    predictor = Predictor(32)
    past = torch.randn(2, 3, 48, 17, 1)
    future = torch.randn(2, 3, 16, 17, 1)
    prediction, consistency, _ = losses(online, target, predictor, past, past.clone(), future, True)
    (prediction + consistency).backward()
    if not torch.isfinite(prediction + consistency) or not online.fc.weight.grad is None:
        # The auxiliary losses must reach the backbone; fc is classification-only.
        raise AssertionError("Unexpected auxiliary gradient")
    if not any(p.grad is not None and p.grad.abs().sum() > 0 for p in online.blocks.parameters()):
        raise AssertionError("PDI backbone gradient missing")
    if any(p.grad is not None for p in target.parameters()):
        raise AssertionError("EMA target received gradient")
    update_ema(target, online, .99)
    return {"finite": True, "online_grad": True, "target_grad": False}


def main():
    parser = argparse.ArgumentParser(description="PDI source-only pilot")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("audit")
    sub.add_parser("smoke")
    training = sub.add_parser("train")
    training.add_argument("--method", choices=pdi_train.METHODS, required=True)
    training.add_argument("--fold", choices=("caucafall", "gmdcsa24", "le2i", "urfd"), default="caucafall")
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--stop-after-epoch", type=int)
    reporting = sub.add_parser("report")
    reporting.add_argument("--fold", choices=("caucafall", "gmdcsa24", "le2i", "urfd"), default="caucafall")
    reporting.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.command == "smoke":
        result = smoke()
    else:
        base, pdi = pdi_train.configs()
        if args.command == "audit":
            result = pdi_train.audit(base, pdi)
        elif args.command == "train":
            result = pdi_train.train(base, pdi, args.method, args.fold, args.seed, args.stop_after_epoch)
        else:
            result = pdi_train.report(base, pdi, args.fold, args.seed)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
