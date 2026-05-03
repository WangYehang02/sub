#!/usr/bin/env python3
"""NeurIPS-style bar chart: DiffGAD / FMGAD runtime slowdown per dataset + Avg."""

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

    # Per-dataset slowdown; Avg = ratio of mean wall times (2064.54 / 164.81), not mean of per-dataset ratios
    datasets = ["Books", "Disney", "Enron", "Reddit", "Weibo", "Avg"]
    slowdowns = np.array([29.23, 42.04, 9.71, 11.36, 11.56, 12.53], dtype=float)
    avg_idx = len(datasets) - 1

    fig_w, fig_h = 6.5, 3.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), layout="constrained")

    x = np.arange(len(datasets))
    # Desaturated, paper-friendly red
    base_face = "#B89090"
    base_edge = "#6E6E6E"
    face = [base_face] * len(datasets)
    edge = [base_edge] * len(datasets)
    lw = [0.45] * len(datasets)
    hatch = [""] * len(datasets)
    face[avg_idx] = "#A88888"
    edge[avg_idx] = "#555555"
    lw[avg_idx] = 0.75
    hatch[avg_idx] = "///"

    bars = ax.bar(
        x,
        slowdowns,
        color=face,
        edgecolor=edge,
        linewidth=lw,
        hatch=hatch,
        zorder=2,
    )

    ax.set_ylabel("Runtime Ratio: DiffGAD / FMGAD (×)")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ymax = max(slowdowns) * 1.12
    ax.set_ylim(0, ymax)
    ax.yaxis.grid(True, linestyle="--", which="major", zorder=0)
    ax.set_axisbelow(True)
    _spines_light(ax)

    for rect, val in zip(bars, slowdowns):
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            rect.get_height() + ymax * 0.015,
            f"{val:.2f}×",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )

    pdf_path = out_dir / "runtime_speedup_diffgad.pdf"
    png_path = out_dir / "runtime_speedup_diffgad.png"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
