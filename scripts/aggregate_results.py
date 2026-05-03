#!/usr/bin/env python3
"""Aggregate JSON metrics written by run_single / run_all_5seeds into a CSV table."""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _parse_stem(stem: str) -> Optional[Tuple[str, int]]:
    m = re.match(r"^(.+)_seed(\d+)$", stem)
    if not m:
        return None
    return m.group(1), int(m.group(2))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=str, default="results/main_runs", help="Directory containing *_seed*.json")
    ap.add_argument("--output", type=str, default="results/main_table.csv")
    args = ap.parse_args()
    in_dir = Path(args.input).resolve()
    if not in_dir.is_dir():
        raise SystemExit(f"input directory not found: {in_dir}")

    rows_out: List[Dict[str, Any]] = []
    for p in sorted(in_dir.glob("*.json")):
        if p.name == "run_meta.json":
            continue
        parsed = _parse_stem(p.stem)
        if not parsed:
            continue
        dataset, seed = parsed
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        rows_out.append(
            {
                "dataset": dataset,
                "seed": seed,
                "auc_mean": data.get("auc_mean"),
                "ap_mean": data.get("ap_mean"),
                "auc_std": data.get("auc_std"),
                "ap_std": data.get("ap_std"),
                "path": str(p),
            }
        )

    rows_out.sort(key=lambda r: (r["dataset"], r["seed"]))
    out_csv = Path(args.output).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "seed", "auc_mean", "ap_mean", "auc_std", "ap_std", "path"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows_out:
            w.writerow(r)

    # Per-dataset mean ± stdev across seeds (AUROC / AP)
    by_d: Dict[str, List[Tuple[float, float]]] = {}
    for r in rows_out:
        if r["auc_mean"] is None or r["ap_mean"] is None:
            continue
        by_d.setdefault(str(r["dataset"]), []).append((float(r["auc_mean"]), float(r["ap_mean"])))

    summary_lines = ["dataset,n_seeds,auc_mean,auc_std,ap_mean,ap_std"]
    for d in sorted(by_d.keys()):
        xs = [a for a, _ in by_d[d]]
        ys = [b for _, b in by_d[d]]
        n = len(xs)
        auc_m = statistics.mean(xs) if n else float("nan")
        ap_m = statistics.mean(ys) if n else float("nan")
        auc_s = statistics.stdev(xs) if n > 1 else 0.0
        ap_s = statistics.stdev(ys) if n > 1 else 0.0
        summary_lines.append(f"{d},{n},{auc_m:.6f},{auc_s:.6f},{ap_m:.6f},{ap_s:.6f}")

    summary_path = out_csv.with_name(out_csv.stem + "_summary_by_dataset.csv")
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("Wrote", out_csv, "and", summary_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
