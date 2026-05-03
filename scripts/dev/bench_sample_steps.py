#!/usr/bin/env python3
"""
Run FMGAD on multiple datasets/seeds with sample_steps overridden (default 1), via temp YAML.
Writes one JSON per run and summary.json under --output-dir.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

REPO = Path(__file__).resolve().parents[2]


def _default_model_root() -> str | None:
    mnt = Path("/mnt/yehang/FMGADfinal_runs/models")
    try:
        if mnt.parent.is_dir() and os.access(mnt.parent, os.W_OK):
            mnt.mkdir(parents=True, exist_ok=True)
            return str(mnt)
    except OSError:
        pass
    return None


def _one(job: Tuple[str, int, str, int, Path, str]) -> Dict[str, Any]:
    dataset, seed, gpu, sample_steps, out_dir, py_exe = job
    cfg_path = REPO / "configs" / f"{dataset}.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
    cfg["sample_steps"] = int(sample_steps)
    tmp_dir = out_dir / "_tmp_cfgs"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_cfg = tmp_dir / f"{dataset}_s{sample_steps}_seed{seed}.yaml"
    with open(tmp_cfg, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=False)

    result_path = out_dir / f"{dataset}_seed{seed}.json"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["FMGAD_RUN_TAG_SUFFIX"] = f"seed{seed}_steps{sample_steps}"
    if not env.get("FMGAD_MODEL_ROOT"):
        mr = _default_model_root()
        if mr:
            env["FMGAD_MODEL_ROOT"] = mr

    cmd = [
        py_exe,
        str(REPO / "main_train.py"),
        "--config",
        str(tmp_cfg),
        "--seed",
        str(seed),
        "--result-file",
        str(result_path),
    ]
    proc = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True)
    row: Dict[str, Any] = {
        "dataset": dataset,
        "seed": seed,
        "sample_steps": int(sample_steps),
        "gpu": gpu,
        "returncode": proc.returncode,
    }
    if proc.returncode == 0 and result_path.is_file():
        with open(result_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        row["auc_mean"] = payload.get("auc_mean")
        row["ap_mean"] = payload.get("ap_mean")
    else:
        row["auc_mean"] = None
        row["ap_mean"] = None
        row["stderr_tail"] = (proc.stderr or "")[-2000:]
    log_path = out_dir / f"{dataset}_seed{seed}.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(proc.stdout or "")
        if proc.stderr:
            f.write("\n--- stderr ---\n")
            f.write(proc.stderr)
    row["log_file"] = str(log_path)
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", type=str, default="books,disney,enron,reddit,weibo")
    ap.add_argument("--seeds", type=str, default="42,0,1,2,3")
    ap.add_argument("--sample-steps", type=int, default=1)
    ap.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7")
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Absolute or relative (under repo) directory for JSON/logs/tmp yaml",
    )
    args = ap.parse_args()

    dsets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    gpu_list = [g.strip() for g in args.gpus.split(",") if g.strip()] or ["0"]
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = (REPO / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ss = int(args.sample_steps)
    py_exe = sys.executable
    jobs: List[Tuple[str, int, str, int, Path, str]] = []
    idx = 0
    for d in dsets:
        for s in seeds:
            g = gpu_list[idx % len(gpu_list)]
            jobs.append((d, s, g, ss, out_dir, py_exe))
            idx += 1

    mw = min(int(args.max_workers), len(jobs), max(len(gpu_list), 1))
    rows: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=mw) as ex:
        futs = {ex.submit(_one, j): j for j in jobs}
        for k, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            r = rows[-1]
            print(
                f"[{k}/{len(jobs)}] {r['dataset']} seed={r['seed']} steps={ss} rc={r['returncode']} auc={r.get('auc_mean')}",
                flush=True,
            )

    from collections import defaultdict
    from statistics import mean, stdev

    by_d: Dict[str, List[float]] = defaultdict(list)
    by_ap: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        if r.get("auc_mean") is not None:
            by_d[r["dataset"]].append(float(r["auc_mean"]))
        if r.get("ap_mean") is not None:
            by_ap[r["dataset"]].append(float(r["ap_mean"]))

    summary: Dict[str, Any] = {}
    for d in sorted(by_d.keys()):
        xs = by_d[d]
        ys = by_ap.get(d, [])
        summary[d] = {
            "mean_auc": mean(xs),
            "std_auc": stdev(xs) if len(xs) > 1 else 0.0,
            "n": len(xs),
            "mean_ap": mean(ys) if ys else None,
            "std_ap": stdev(ys) if len(ys) > 1 else 0.0 if ys else None,
        }

    meta = {
        "sample_steps": ss,
        "datasets": dsets,
        "seeds": seeds,
        "output_dir": str(out_dir),
        "summary_by_dataset": summary,
        "rows": rows,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print("Wrote", out_dir / "summary.json", flush=True)
    return 0 if all(r.get("returncode") == 0 for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
