#!/usr/bin/env python3
"""CPU domain diagnostics. Run p0, p1, p2, p3 in order."""
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
from src.domain_analysis import p0, extract, p3

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("stage", choices=("p0", "p1", "p2", "p3"))
args = parser.parse_args()
if args.stage == "p0":
    p0()
elif args.stage == "p1":
    extract("rgb")
elif args.stage == "p2":
    extract("skeleton")
else:
    p3()
