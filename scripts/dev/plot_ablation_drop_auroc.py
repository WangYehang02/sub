#!/usr/bin/env python3
"""NeurIPS-style bar chart: Avg AUROC drop vs Full FMGAD for ablation variants."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def _paper_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#111111",
            "axes.titlecolor": "#111111",
            "xtick.color": "#111111",
            "ytick.color": "#111111",
            "text.color": "#111111",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.45,
            "xtick.major.width": 0.45,
            "ytick.major.width": 0.45,
            "grid.linewidth": 0.35,
            "grid.alpha": 0.32,
        }
    )


def _spines_light(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.45)
    ax.spines["bottom"].set_linewidth(0.45)


def main() -> None:
    _paper_style()

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "figure"
    out_dir.mkdir(parents=True, exist_ok=True)

    full_avg = 0.6993
    variants = [
        ("w/o Residual", 0.6704),
        ("w/o Proto", 0.6663),
        ("w/o Guidance", 0.6621),
        ("w/o Smooth", 0.6502),
        ("w/o Polarity", 0.4751),
        ("w/o Virtual Neighbor", 0.6769),
    ]
    short_labels = [
        "w/o Resid.",
        "w/o Proto.",
        "w/o Guid.",
        "w/o Smooth",
        "w/o Polarity",
        "w/o VN",
    ]
    drops = np.array([full_avg - v[1] for v in variants], dtype=float)
    polarity_idx = 4  # "w/o Polarity"

    fig_w, fig_h = 6.5, 3.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), layout="constrained")

    n = len(short_labels)
    colors = ["#4C72B0"] * n
    edgecolors = ["#555555"] * n
    linewidths = [0.45] * n
    # Subtle emphasis for w/o Polarity (slightly darker fill + slightly thicker edge)
    colors[polarity_idx] = "#3D5A80"
    edgecolors[polarity_idx] = "#2A2A2A"
    linewidths[polarity_idx] = 0.95

    x = np.arange(n)
    bars = ax.bar(
        x,
        drops,
        color=colors,
        edgecolor=edgecolors,
        linewidth=linewidths,
        zorder=2,
    )

    ax.set_ylabel("Avg. AUROC Drop")
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=30, ha="right")
    ax.set_ylim(0, max(drops) * 1.12)
    ax.yaxis.grid(True, linestyle="--", which="major", zorder=0)
    ax.set_axisbelow(True)
    _spines_light(ax)

    for rect, val in zip(bars, drops):
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            rect.get_height() + 0.002,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )

    pdf_path = out_dir / "ablation_drop_auroc.pdf"
    png_path = out_dir / "ablation_drop_auroc.png"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
