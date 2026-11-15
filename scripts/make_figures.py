#!/usr/bin/env python3
"""README figures following Chen Liu's Figures for Papers skill conventions.

Reference: https://github.com/ChenLiu-1996/figures4papers
See THIRD_PARTY_NOTICES.md. Inputs are bundled audit metadata and ERM metrics.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PALETTE = {
    "blue_main": "#0F4D92", "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE", "green_3": "#8BCF8B",
    "red_strong": "#B64342", "neutral": "#CFCECE", "teal": "#42949E",
}
INK = "#182E43"
MUTED = "#526579"
DOMAINS = ("caucafall", "gmdcsa24", "le2i", "urfd")
NAMES = ("CAUCAFall", "GMDCSA-24", "Le2i", "URFD")


@dataclass(frozen=True)
class FigureStyle:
    font_size: int = 15
    axes_linewidth: float = 2


def apply_publication_style(style=FigureStyle()):
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": style.font_size, "axes.linewidth": style.axes_linewidth,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "svg.fonttype": "none", "pdf.fonttype": 42,
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED,
        "ytick.color": MUTED, "figure.facecolor": "white", "savefig.facecolor": "white",
    })


def finalize_figure(fig, out_path, formats=("png", "pdf", "svg"), dpi=300):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        destination = out_path.with_suffix(f".{extension}")
        fig.savefig(destination, dpi=dpi, bbox_inches="tight", pad_inches=.18)
        print(destination.relative_to(ROOT) if destination.is_relative_to(ROOT) else destination)
    plt.close(fig)


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def pipeline(out_dir):
    rows = read_csv(ROOT / "data/manifest.csv")
    counts = [sum(r["dataset"] == domain for r in rows) for domain in DOMAINS]
    fig = plt.figure(figsize=(15, 7.3))
    fig.text(.04, .94, "Learning to detect falls across datasets", size=26, weight="bold")
    fig.text(.04, .89, f"{len(rows)} audited recordings  /  4 held-out domains  /  3 training seeds", color=MUTED, size=15)
    ax = fig.add_axes([.025, .53, .95, .29])
    ax.set(xlim=(-.01, 1.015), ylim=(0, 1))
    ax.axis("off")
    stages = [
        ("RGB recordings", "Four datasets\nOne label per video", "#EDF3FA"),
        ("Frozen pose", "RTMDet-m + RTMPose-m\n17 joints · one person", "#EDF3FA"),
        ("Temporal windows", "15 fps · 64 frames\nStride 32 · x, y, confidence", "#EDF3FA"),
        ("CTR-GCN", "Graph + temporal blocks\nOne logit per window", "#EDF3FA"),
        ("Video prediction", "Top-two-logit MIL\nFall / Non-Fall", PALETTE["green_1"]),
    ]
    for i, (title, detail, face) in enumerate(stages):
        x = .012 + i * .202
        ax.add_patch(FancyBboxPatch((x, .12), .169, .75, boxstyle="round,pad=0.012,rounding_size=0.03",
                                   edgecolor=PALETTE["blue_main"], facecolor=face, linewidth=1.6))
        ax.text(x + .015, .70, f"0{i+1}", size=13, color=PALETTE["blue_secondary"], weight="bold")
        ax.text(x + .0845, .51, title, ha="center", size=14, weight="bold")
        ax.text(x + .0845, .27, detail, ha="center", va="center", size=10.7, linespacing=1.7, color=MUTED)
        if i < 4:
            ax.annotate("", xy=(x+.195, .50), xytext=(x+.178, .50),
                        arrowprops={"arrowstyle": "-|>", "color": PALETTE["blue_main"], "lw": 1.8})

    grid = fig.add_axes([.04, .10, .47, .34])
    grid.set(xlim=(-1.55, 4.1), ylim=(4.55, -1.5))
    grid.axis("off")
    grid.text(-1.5, -1.2, "Leave one dataset out", size=17, weight="bold")
    for col, (name, count) in enumerate(zip(NAMES, counts)):
        grid.text(col+.45, -.05, f"{name}\nn = {count}", ha="center", va="bottom", size=10, color=MUTED)
    for row in range(4):
        grid.text(-.12, row+.55, f"Fold {row+1}", ha="right", va="center", size=12)
        for col in range(4):
            target = row == col
            grid.add_patch(Rectangle((col, row+.16), .90, .77,
                           facecolor=PALETTE["blue_main"] if target else "#E8F1EC",
                           edgecolor="white", linewidth=1.2))
            grid.text(col+.45, row+.55, "TARGET" if target else "source",
                      ha="center", va="center", color="white" if target else "#385449", size=10,
                      weight="bold" if target else "normal")

    note = fig.add_axes([.57, .12, .39, .31])
    note.axis("off")
    note.add_patch(FancyBboxPatch((.01, .02), .97, .96, boxstyle="round,pad=0.01,rounding_size=.035",
                                 facecolor="#F3F6F9", edgecolor="none"))
    note.text(.06, .82, "Source-only selection", size=17, weight="bold")
    note.text(.06, .61, "Checkpoint · threshold · SWAD loss valley", size=12, color=MUTED)
    note.text(.06, .43, "BatchNorm calibrated on source training windows", size=11.5, color=MUTED)
    note.plot([.06, .92], [.32, .32], color="#D6E0E9", lw=1, transform=note.transAxes)
    note.text(.06, .18, "Evaluate the unseen target once per run", size=12, color=PALETTE["blue_main"], weight="bold")
    fig.text(.04, .035, "Schematic of the implemented pipeline. Dataset counts come from the frozen manifest.", size=10, color=MUTED)
    finalize_figure(fig, out_dir / "pipeline")


def benchmark(out_dir):
    rows = read_csv(ROOT / "benchmarks/erm/metrics.csv")
    if len(rows) != 12 or any(r["method"] != "erm" for r in rows):
        raise ValueError("Expected the 12-run ERM snapshot")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.7))
    fig.subplots_adjust(top=.73, bottom=.19, left=.065, right=.98, wspace=.24)
    fig.text(.045, .93, "ERM baseline on unseen datasets", size=25, weight="bold")
    fig.text(.045, .85, "4-fold leave-one-dataset-out  /  mean ± sample SD over 3 seeds", size=14, color=MUTED)
    for ax, key, label in zip(axes, ("f1", "pr_auc"), ("Fall F1", "Average precision")):
        values = [np.array([float(r[key]) for r in rows if r["fold"] == domain]) for domain in DOMAINS]
        means = np.array([v.mean() for v in values])
        stds = np.array([v.std(ddof=1) for v in values])
        x = np.arange(4)
        ax.bar(x, means, width=.58, color=PALETTE["blue_main"], edgecolor=INK, linewidth=1.2,
               yerr=stds, capsize=5, error_kw={"elinewidth": 1.5, "capthick": 1.5}, zorder=3)
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax.text(i, mean+std+.025, f"{mean:.3f}", ha="center", va="bottom", size=13, weight="bold")
        ax.set(xticks=x, xticklabels=NAMES, ylim=(0, 1.13), yticks=np.arange(0, 1.01, .25), ylabel=label)
        ax.tick_params(axis="x", labelsize=12, length=0, pad=12)
        ax.tick_params(axis="y", labelsize=11)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#E6EBF0", linewidth=.8)
        ax.spines["left"].set_bounds(0, 1)
        macro = [np.mean([float(r[key]) for r in rows if int(r["seed"]) == seed]) for seed in (42, 123, 2026)]
        ax.set_title(f"Macro: {np.mean(macro):.3f} ± {np.std(macro, ddof=1):.3f}", loc="left", size=15, pad=14)
    fig.text(.045, .055, "ERM snapshot verified from saved video predictions. SWAD scores await completion of legacy averaging repairs.", size=10.5, color=MUTED)
    finalize_figure(fig, out_dir / "benchmark")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/figures")
    args = parser.parse_args()
    apply_publication_style()
    pipeline(args.output_dir)
    benchmark(args.output_dir)


if __name__ == "__main__":
    main()
