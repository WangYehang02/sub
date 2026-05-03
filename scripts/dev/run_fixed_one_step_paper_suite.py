#!/usr/bin/env python3
"""
Orchestrate: (1) verify main metrics vs reference, (2) ablation 175 jobs, (3) summarize+plots,
(4) runtime FMGAD vs DiffGAD sequential, (5) runtime plot.

Example:
  python scripts/dev/run_fixed_one_step_paper_suite.py --base-dir /mnt/yehang/FMGADfinal_runs/fixed_one_step_suite_20260503
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PY = sys.executable


def run(cmd: list[str], cwd: Path | None = None) -> int:
    print("+", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(cwd or REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", type=str, required=True)
    ap.add_argument("--skip-verify", action="store_true")
    ap.add_argument("--skip-ablation", action="store_true")
    ap.add_argument("--skip-runtime", action="store_true")
    ap.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7")
    args = ap.parse_args()

    base = Path(args.base_dir).expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)
    verify_dir = base / "verify_main"
    ablation_dir = base / "ablation"
    runtime_dir = base / "runtime"

    if not args.skip_verify:
        verify_dir.mkdir(parents=True, exist_ok=True)
        rc = run(
            [
                PY,
                str(REPO / "scripts/dev/verify_main_metrics_targets.py"),
                "--output-dir",
                str(verify_dir),
                "--gpus",
                args.gpus,
                "--max-workers",
                "8",
            ]
        )
        if rc != 0:
            print("VERIFY FAILED (exit {}). Stopping per user protocol.".format(rc), flush=True)
            return rc

    if not args.skip_ablation:
        ablation_dir.mkdir(parents=True, exist_ok=True)
        rc = run(
            [
                PY,
                str(REPO / "scripts/run_ablation.py"),
                "--repo-root",
                str(REPO),
                "--result-root",
                str(ablation_dir),
                "--datasets",
                "books,disney,enron,reddit,weibo",
                "--seeds",
                "42,0,1,2,3",
                "--gpus",
                args.gpus,
                "--max-workers",
                "8",
            ]
        )
        if rc != 0:
            print("Ablation runner returned", rc, flush=True)
            return rc
        rc = run(
            [
                PY,
                str(REPO / "scripts/dev/summarize_fmgad_ablation.py"),
                "--result-root",
                str(ablation_dir),
            ]
        )
        if rc != 0:
            return rc
        csv_path = ablation_dir / "ablation_summary.csv"
        rc = run(
            [
                PY,
                str(REPO / "scripts/dev/plot_ablation_drop_auroc.py"),
                "--summary-csv",
                str(csv_path),
                "--out-dir",
                str(ablation_dir),
            ]
        )
        if rc != 0:
            return rc

    if not args.skip_runtime:
        runtime_dir.mkdir(parents=True, exist_ok=True)
        rc = run(
            [
                PY,
                str(REPO / "scripts/dev/run_runtime_fmgad_vs_diffgad_one_step.py"),
                "--out-root",
                str(runtime_dir),
            ]
        )
        if rc != 0:
            print("Runtime suite returned", rc, flush=True)
            return rc
        rc = run(
            [
                PY,
                str(REPO / "scripts/dev/plot_runtime_speedup_from_csv.py"),
                "--runtime-csv",
                str(runtime_dir / "runtime_summary.csv"),
                "--out-dir",
                str(runtime_dir),
            ]
        )
        if rc != 0:
            return rc

    print("Suite outputs under:", base, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
