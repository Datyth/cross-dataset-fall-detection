#!/usr/bin/env python3
"""Paper-style PDI architecture schematic following figures4papers conventions.

Uses the publication typography, semantic palette, and PNG/PDF/SVG export
helpers already adapted in scripts/make_figures.py from Chen Liu's repo.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from make_figures import INK, MUTED, PALETTE, apply_publication_style, finalize_figure

ROOT = Path(__file__).resolve().parents[1]
BLUE_FILL = "#EAF2FA"
GREEN_FILL = PALETTE["green_1"]
GRAY_FILL = "#F2F4F6"
RED_FILL = "#F9E7E5"


def node(ax, x, y, width, height, title, detail="", *, fill=GRAY_FILL, edge=None):
    edge = edge or PALETTE["blue_main"]
    patch = FancyBboxPatch((x, y), width, height,
                           boxstyle="round,pad=0.012,rounding_size=0.11",
                           linewidth=1.35, edgecolor=edge, facecolor=fill)
    ax.add_patch(patch)
    if detail:
        ax.text(x + width / 2, y + height * .63, title, ha="center", va="center",
                fontsize=11.3, fontweight="bold", color=INK)
        ax.text(x + width / 2, y + height * .31, detail, ha="center", va="center",
                fontsize=8.6, color=MUTED, linespacing=1.25)
    else:
        ax.text(x + width / 2, y + height / 2, title, ha="center", va="center",
                fontsize=11.3, fontweight="bold", color=INK)


def arrow(ax, start, end, *, color=PALETTE["blue_main"], style="-", width=1.7):
    ax.annotate("", xy=end, xytext=start,
                arrowprops={"arrowstyle": "-|>", "color": color, "lw": width,
                            "linestyle": style, "mutation_scale": 12,
                            "shrinkA": 0, "shrinkB": 0})


def figure(out_dir):
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(16, 9.0))
    ax.set(xlim=(0, 16), ylim=(0, 9))
    ax.axis("off")

    fig.text(.035, .955, "Predictive dynamics invariance for fall detection",
             fontsize=20, fontweight="bold", color=INK)
    fig.text(.035, .918, "Video-level classifier with a training-only future-representation objective",
             fontsize=10.5, color=MUTED)

    ax.text(.45, 8.15, "A  VIDEO CLASSIFICATION", fontsize=12.5, fontweight="bold",
            color=PALETTE["blue_main"])
    ax.text(15.5, 8.15, "TRAIN + INFERENCE", ha="right", fontsize=9.5,
            fontweight="bold", color=PALETTE["blue_main"])
    ax.plot([.45, 15.55], [7.98, 7.98], color="#D6E0E9", lw=1)

    top_y, top_h = 6.5, 1.1
    node(ax, .55, top_y, 2.45, top_h, "Skeleton video", "COCO-17 · 15 fps", fill=GRAY_FILL)
    node(ax, 3.4, top_y, 2.45, top_h, "Temporal windows", "64 frames · stride 32", fill=BLUE_FILL)
    node(ax, 6.25, top_y, 2.45, top_h, "CTR-GCN", "shared online encoder", fill=BLUE_FILL)
    node(ax, 9.1, top_y, 2.45, top_h, "Video logit", "mean of top 2 window logits", fill=GREEN_FILL,
         edge="#5E9E61")
    node(ax, 11.95, top_y, 3.45, top_h, "Fall / Non-Fall", "prediction; BCE loss in training", fill=GREEN_FILL,
         edge="#5E9E61")
    for x in (3.0, 5.85, 8.7, 11.55):
        arrow(ax, (x + .07, top_y + .55), (x + .32, top_y + .55))

    ax.text(.45, 5.96, "B  FUTURE REPRESENTATION LEARNING", fontsize=12.5,
            fontweight="bold", color=PALETTE["blue_main"])
    ax.text(15.5, 5.96, "TRAINING ONLY", ha="right", fontsize=9.5,
            fontweight="bold", color=PALETTE["red_strong"])
    ax.plot([.45, 15.55], [5.79, 5.79], color="#D6E0E9", lw=1)

    node(ax, .55, 2.78, 2.35, 1.18, "One valid window", "64 observed frames", fill=GRAY_FILL)
    node(ax, 3.35, 4.33, 2.1, 1.02, "Past", "first 48 frames", fill=BLUE_FILL)
    node(ax, 3.35, 1.25, 2.1, 1.02, "Future", "last 16 frames", fill=GRAY_FILL)
    node(ax, 5.93, 2.93, 2.08, 1.02, "Augment past", "noise + joint dropout",
         fill=RED_FILL, edge=PALETTE["red_strong"])
    arrow(ax, (2.9, 3.67), (3.35, 4.76))
    arrow(ax, (2.9, 3.04), (3.35, 1.76), color=MUTED)
    arrow(ax, (5.45, 4.61), (6.05, 3.95), color=PALETTE["red_strong"])

    node(ax, 8.38, 4.03, 2.0, 1.02, "CTR-GCN", "shared online weights", fill=BLUE_FILL)
    node(ax, 10.8, 4.03, 1.8, 1.02, "MLP", "two predictions", fill=BLUE_FILL)
    node(ax, 8.38, 1.25, 2.0, 1.02, "CTR-GCN EMA", "no gradient", fill=GRAY_FILL, edge=MUTED)
    node(ax, 10.8, 1.25, 1.8, 1.02, "Target", "future features", fill=GRAY_FILL, edge=MUTED)
    arrow(ax, (5.45, 4.85), (8.38, 4.85))
    arrow(ax, (8.01, 3.75), (8.38, 4.3), color=PALETTE["red_strong"])
    arrow(ax, (10.38, 4.54), (10.8, 4.54))
    arrow(ax, (9.38, 4.03), (9.38, 2.27), color=MUTED, style="--", width=1.25)
    ax.text(9.6, 3.13, "EMA update", fontsize=8.3, color=MUTED)
    arrow(ax, (5.45, 1.76), (8.38, 1.76), color=MUTED)
    arrow(ax, (10.38, 1.76), (10.8, 1.76), color=MUTED)

    node(ax, 13.05, 4.03, 2.3, 1.02, "Consistency loss", "P4: compare predictions",
         fill=RED_FILL, edge=PALETTE["red_strong"])
    node(ax, 13.05, 1.25, 2.3, 1.02, "Prediction loss", "predict vs future target",
         fill=GREEN_FILL, edge="#5E9E61")
    arrow(ax, (12.6, 4.54), (13.05, 4.54), color=PALETTE["red_strong"])
    arrow(ax, (12.32, 4.03), (13.6, 2.27), color="#5E9E61")
    arrow(ax, (12.6, 1.76), (13.05, 1.76), color=MUTED)

    ax.plot([.45, 15.55], [.67, .67], color="#D6E0E9", lw=1)
    ax.text(.55, .35, "P3: classification + future prediction", fontsize=10, color=INK)
    ax.text(6.6, .35, "P4: P3 + predictive consistency", fontsize=10,
            fontweight="bold", color=PALETTE["blue_main"])
    ax.text(15.42, .35, "Inference uses panel A only", fontsize=10,
            ha="right", color=MUTED)

    finalize_figure(fig, out_dir / "pdi_architecture")


if __name__ == "__main__":
    figure(ROOT / "docs/figures")
