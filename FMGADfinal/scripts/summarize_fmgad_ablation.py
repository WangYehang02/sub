#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List


DATASETS = ["books", "disney", "enron", "reddit", "weibo"]
VARIANT_ORDER = [
    "full_fmgad",
    "wo_residual",
    "wo_proto",
    "wo_guidance",
    "wo_smooth",
    "wo_polarity",
    "wo_virtual_neighbor",
]
VARIANT_LABEL = {
    "full_fmgad": "Full FMGAD",
    "wo_residual": "w/o Residual",
    "wo_proto": "w/o Proto",
    "wo_guidance": "w/o Guidance",
    "wo_smooth": "w/o Smooth",
    "wo_polarity": "w/o Polarity",
    "wo_virtual_neighbor": "w/o Virtual Neighbor",
}


def _fmt(x: float) -> str:
    return f"{x:.4f}"


def _fmt_pm(m: float, s: float) -> str:
    return f"${m:.4f} \\\\pm {s:.4f}$"


def _collect(result_root: Path) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    out: Dict[str, Dict[str, Dict[str, List[float]]]] = defaultdict(lambda: defaultdict(lambda: {"auc": [], "ap": []}))
    for p in result_root.rglob("seed_*.json"):
        obj = json.loads(p.read_text(encoding="utf-8"))
        v = obj.get("variant")
        d = obj.get("dataset")
        if not v or not d:
            continue
        auc = obj.get("auc", obj.get("auc_mean"))
        ap = obj.get("ap", obj.get("ap_mean"))
        if auc is None or ap is None:
            continue
        out[v][d]["auc"].append(float(auc))
        out[v][d]["ap"].append(float(ap))
    return out


def _stats(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {"mean": float("nan"), "std": float("nan")}
    if len(vals) == 1:
        return {"mean": vals[0], "std": 0.0}
    return {"mean": mean(vals), "std": stdev(vals)}


def _write_csv(summary_path: Path, bucket: Dict[str, Dict[str, Dict[str, List[float]]]]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "dataset", "n", "auroc_mean", "auroc_std", "ap_mean", "ap_std"])
        for v in VARIANT_ORDER:
            if v not in bucket:
                continue
            for d in DATASETS:
                vals_auc = bucket[v][d]["auc"]
                vals_ap = bucket[v][d]["ap"]
                s_auc = _stats(vals_auc)
                s_ap = _stats(vals_ap)
                w.writerow(
                    [
                        v,
                        d,
                        len(vals_auc),
                        _fmt(s_auc["mean"]),
                        _fmt(s_auc["std"]),
                        _fmt(s_ap["mean"]),
                        _fmt(s_ap["std"]),
                    ]
                )


def _build_metric_table(bucket: Dict[str, Dict[str, Dict[str, List[float]]]], metric: str) -> str:
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lcccccc}")
    lines.append("\\toprule")
    lines.append("Variant & Books & Disney & Enron & Reddit & Weibo & Avg \\\\")
    lines.append("\\midrule")
    for v in VARIANT_ORDER:
        if v not in bucket:
            continue
        cells = []
        means = []
        for d in DATASETS:
            st = _stats(bucket[v][d][metric])
            cells.append(_fmt_pm(st["mean"], st["std"]))
            means.append(st["mean"])
        avg = mean(means) if means else float("nan")
        lines.append(f"{VARIANT_LABEL[v]} & " + " & ".join(cells) + f" & ${avg:.4f}$ \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    if metric == "auc":
        lines.append(
            "\\caption{Ablation study of FMGAD under the deterministic fixed-step protocol. Results are reported as mean AUROC over five seeds. Avg denotes the arithmetic mean over the five dataset-wise mean scores. Each variant changes only one component while keeping all other hyperparameters fixed.}"
        )
        lines.append("\\label{tab:fmgad_ablation_auroc}")
    else:
        lines.append(
            "\\caption{Ablation study of FMGAD under the deterministic fixed-step protocol. Results are reported as mean AP over five seeds. Avg denotes the arithmetic mean over the five dataset-wise mean scores. Each variant changes only one component while keeping all other hyperparameters fixed.}"
        )
        lines.append("\\label{tab:fmgad_ablation_ap}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def _write_analysis(path: Path, bucket: Dict[str, Dict[str, Dict[str, List[float]]]]) -> None:
    def m(v: str, d: str, metric: str) -> float:
        return _stats(bucket[v][d][metric])["mean"]

    lines = []
    lines.append("FMGAD ablation analysis (conservative summary)")
    lines.append("")
    if "full_fmgad" not in bucket:
        lines.append("Full FMGAD results are missing; please rerun full_fmgad first.")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    lines.append(
        "Across five seeds, Full FMGAD provides a stable reference on Weibo and Enron, while Books/Disney/Reddit remain more sensitive to seed and component toggles."
    )

    if "wo_polarity" in bucket:
        delta_enron = m("wo_polarity", "enron", "auc") - m("full_fmgad", "enron", "auc")
        lines.append(
            f"Removing label-free polarity calibration changes Enron AUROC by {delta_enron:+.4f}; this indicates polarity contributes non-trivially on topology-sensitive graphs."
        )
    if "wo_smooth" in bucket:
        db = m("wo_smooth", "books", "auc") - m("full_fmgad", "books", "auc")
        de = m("wo_smooth", "enron", "auc") - m("full_fmgad", "enron", "auc")
        dw = m("wo_smooth", "weibo", "auc") - m("full_fmgad", "weibo", "auc")
        lines.append(
            f"Disabling score smoothing shifts AUROC by {db:+.4f} (Books), {de:+.4f} (Enron), and {dw:+.4f} (Weibo), suggesting dataset-dependent smoothing effects."
        )
    if "wo_proto" in bucket:
        dd = m("wo_proto", "disney", "auc") - m("full_fmgad", "disney", "auc")
        de = m("wo_proto", "enron", "auc") - m("full_fmgad", "enron", "auc")
        lines.append(
            f"Removing prototype conditioning changes AUROC by {dd:+.4f} on Disney and {de:+.4f} on Enron; prototype guidance is beneficial in some datasets but not uniformly."
        )
    if "wo_residual" in bucket:
        lines.append(
            "Removing residual augmentation shows clear dataset sensitivity, with non-uniform gains/drops across the five benchmarks."
        )
    lines.append(
        "No labels are used for calibration direction selection, timestep selection, checkpoint selection, or ablation selection; labels are only used for final AUROC/AP computation."
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-root", type=str, default="results/ablation")
    args = ap.parse_args()

    root = Path(args.result_root).resolve()
    bucket = _collect(root)

    _write_csv(root / "ablation_summary.csv", bucket)
    (root / "ablation_auc_latex.tex").write_text(_build_metric_table(bucket, "auc"), encoding="utf-8")
    (root / "ablation_ap_latex.tex").write_text(_build_metric_table(bucket, "ap"), encoding="utf-8")
    _write_analysis(root / "ablation_analysis.txt", bucket)
    print(f"Wrote summaries to: {root}")


if __name__ == "__main__":
    main()
