#!/usr/bin/env python3
"""
Re-run Full FMGAD (repo configs/*.yaml, sample_steps as in YAML) on 5 datasets × 5 seeds.
Compare AUROC/AP mean & sample std (ddof=1) to reference targets; exit non-zero if out of tolerance.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Tuple

import yaml

REPO = Path(__file__).resolve().parents[2]

REFERENCE = {
    "books": {"auc_m": 0.621840, "auc_s": 0.025319, "ap_m": 0.037936, "ap_s": 0.011638},
    "disney": {"auc_m": 0.532486, "auc_s": 0.210563, "ap_m": 0.075896, "ap_s": 0.041900},
    "enron": {"auc_m": 0.838607, "auc_s": 0.019953, "ap_m": 0.002449, "ap_s": 0.001120},
    "reddit": {"auc_m": 0.550287, "auc_s": 0.026484, "ap_m": 0.037067, "ap_s": 0.003277},
    "weibo": {"auc_m": 0.942080, "auc_s": 0.000034, "ap_m": 0.343513, "ap_s": 0.000054},
}


def _model_root() -> str | None:
    mnt = Path("/mnt/yehang/FMGADfinal_runs/models")
    try:
        if mnt.parent.is_dir() and os.access(mnt.parent, os.W_OK):
            mnt.mkdir(parents=True, exist_ok=True)
            return str(mnt)
    except OSError:
        pass
    return None


def _one(job: Tuple[str, int, str, Path]) -> Dict[str, Any]:
    dataset, seed, gpu, out_root = job
    tag = f"verify_main_{dataset}_seed{seed}_{os.getpid()}"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["FMGAD_RUN_TAG_SUFFIX"] = tag
    mr = env.get("FMGAD_MODEL_ROOT") or _model_root()
    if mr:
        env["FMGAD_MODEL_ROOT"] = mr

    result_path = out_root / f"{dataset}_seed{seed}.json"
    cmd = [
        sys.executable,
        str(REPO / "main_train.py"),
        "--config",
        str(REPO / "configs" / f"{dataset}.yaml"),
        "--seed",
        str(seed),
        "--device",
        "0",
        "--result-file",
        str(result_path),
    ]
    p = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True)
    row: Dict[str, Any] = {"dataset": dataset, "seed": seed, "returncode": p.returncode}
    if p.returncode == 0 and result_path.is_file():
        with open(result_path, "r", encoding="utf-8") as f:
            j = json.load(f)
        row["auc_mean"] = float(j["auc_mean"])
        row["ap_mean"] = float(j["ap_mean"])
    else:
        row["auc_mean"] = None
        row["ap_mean"] = None
        row["stderr"] = (p.stderr or "")[-800:]
    return row


def _within(ref_m: float, ref_s: float, got_m: float, got_s: float, *, dataset: str, metric: str) -> bool:
    """Loose tolerance: mean shift vs reference uncertainty; std tracked but not hard-failed for Disney."""
    dm = abs(got_m - ref_m)
    ds = abs(got_s - ref_s)
    # Mean: allow 2.5 * ref_std or floor
    tol_m = max(0.025, 2.5 * ref_s)
    if dataset == "disney" and metric == "auc":
        tol_m = max(0.12, 3.0 * ref_s)
    if dataset == "disney" and metric == "ap":
        tol_m = max(0.06, 2.0 * ref_s)
    if not (dm <= tol_m):
        return False
    # Std: soft check (Weibo tiny std)
    tol_s = max(0.02, 3.0 * ref_s)
    if dataset == "weibo":
        tol_s = max(0.0002, 5.0 * ref_s)
    if ds > tol_s:
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7")
    ap.add_argument("--max-workers", type=int, default=8)
    args = ap.parse_args()

    out_root = Path(args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    datasets = list(REFERENCE.keys())
    seeds = [42, 0, 1, 2, 3]
    gpu_list = [g.strip() for g in args.gpus.split(",") if g.strip()] or ["0"]
    jobs: List[Tuple[str, int, str, Path]] = []
    idx = 0
    for d in datasets:
        for s in seeds:
            jobs.append((d, s, gpu_list[idx % len(gpu_list)], out_root))
            idx += 1

    rows: List[Dict[str, Any]] = []
    mw = min(args.max_workers, len(jobs), max(len(gpu_list), 1))
    with ProcessPoolExecutor(max_workers=mw) as ex:
        futs = {ex.submit(_one, j): j for j in jobs}
        for k, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            r = rows[-1]
            print(f"[{k}/{len(jobs)}] {r['dataset']} seed={r['seed']} rc={r['returncode']} auc={r.get('auc_mean')}", flush=True)

    # snapshot configs sample_steps
    cfg_snap = {}
    for d in datasets:
        with open(REPO / "configs" / f"{d}.yaml", "r", encoding="utf-8") as f:
            cfg_snap[d] = yaml.safe_load(f)

    by_d: Dict[str, List[Tuple[float, float]]] = {d: [] for d in datasets}
    for r in rows:
        if r.get("auc_mean") is not None:
            by_d[r["dataset"]].append((float(r["auc_mean"]), float(r["ap_mean"])))

    report_lines = ["# Full FMGAD verification (fixed one-step configs)", ""]
    all_ok = True
    for d in datasets:
        ref = REFERENCE[d]
        vals = by_d[d]
        if len(vals) != 5:
            all_ok = False
            report_lines.append(f"## {d}: FAIL (missing runs, n={len(vals)})")
            continue
        aucs = [x[0] for x in vals]
        aps = [x[1] for x in vals]
        got_am, got_as = mean(aucs), stdev(aucs)
        got_pm, got_ps = mean(aps), stdev(aps)
        ok_a = _within(ref["auc_m"], ref["auc_s"], got_am, got_as, dataset=d, metric="auc")
        ok_p = _within(ref["ap_m"], ref["ap_s"], got_pm, got_ps, dataset=d, metric="ap")
        ok = ok_a and ok_p
        all_ok = all_ok and ok
        status = "PASS" if ok else "FAIL"
        report_lines.append(f"## {d}: {status}")
        report_lines.append(
            f"- Ref AUROC: {ref['auc_m']:.6f} ± {ref['auc_s']:.6f} | Got: {got_am:.6f} ± {got_as:.6f}"
        )
        report_lines.append(
            f"- Ref AP: {ref['ap_m']:.6f} ± {ref['ap_s']:.6f} | Got: {got_pm:.6f} ± {got_ps:.6f}"
        )
        report_lines.append(f"- YAML sample_steps: {cfg_snap[d].get('sample_steps')}")
        report_lines.append("")

    meta = {
        "all_pass": all_ok,
        "reference": REFERENCE,
        "rows": rows,
        "report_md": "\n".join(report_lines),
    }
    with open(out_root / "verify_report.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    (out_root / "VERIFY_REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print("\n".join(report_lines))
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
