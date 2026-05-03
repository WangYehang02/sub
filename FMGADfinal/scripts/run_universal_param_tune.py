#!/usr/bin/env python3
"""
在 seeds × 五数据集 上网格搜索 universal 极性 YAML 超参，目标：最大化 25 次运行的平均 AUROC。

输出目录：results/universal_param_tune/<run_tag>/trial_XXXX/{dataset}_seed{s}.json
并行：每个 (trial, dataset, seed) 为独立子进程，按 GPU 轮询分配。
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

REPO = Path(__file__).resolve().parents[1]
DATASETS = ("books", "disney", "enron", "reddit", "weibo")

UNIVERSAL_BASE: Dict[str, Any] = {
    "polarity_adapter": "universal_no_y",
    "polarity_use_local_probe": True,
    "polarity_use_nk_probe": True,
    "polarity_use_structural_probe": True,
    "polarity_gate_tau": 0.1,
    "polarity_gate_margin": 0.012,
    "polarity_gate_min_confidence": 0.06,
    "polarity_gate_topk_percent": 0.05,
    "polarity_struct_lcc_threshold": 0.04,
    "polarity_struct_deg_threshold": 0.04,
    "polarity_struct_density_gap": 0.02,
    "polarity_autovote_fallback": True,
    "polarity_vote_q": 0.1,
    "polarity_vote_margin": 1,
    "polarity_min_confidence": 0.14,
    "polarity_lcc_rho_strong": 0.04,
    "polarity_deg_rho_strong": 0.04,
    "polarity_connectivity_rel_gap": 0.02,
    "lcc_spearman_threshold": -0.05,
    "polarity_verbose": False,
    "polarity_unsup_proxy_q": 0.05,
}


def _load_yaml(p: Path) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.Loader)


def _trial_grid(grid: str) -> List[Dict[str, Any]]:
    """返回若干组「相对 UNIVERSAL_BASE 的完整 universal 字段字典」。"""
    if grid == "small":
        taus = [0.03, 0.06, 0.10]
        margins = [0.015, 0.035]
        gmins = [0.07, 0.12]
        pmins = [0.14, 0.22]
        vqs = [0.10]
    elif grid == "medium":
        # 54 trials × 25 jobs：tau×margin×gmin × pmin(2)；vote_q 固定为 0.1 控制总时长
        taus = [0.03, 0.055, 0.10]
        margins = [0.012, 0.025, 0.04]
        gmins = [0.06, 0.10, 0.14]
        pmins = [0.14, 0.22]
        vqs = [0.10]
    else:  # large — 4×3×2×2×2 = 96 trials × 25
        taus = [0.03, 0.055, 0.08, 0.11]
        margins = [0.01, 0.025, 0.045]
        gmins = [0.05, 0.12]
        pmins = [0.14, 0.22]
        vqs = [0.08, 0.12]

    trials: List[Dict[str, Any]] = []
    for tau, mar, gmin, pmin, vq in itertools.product(taus, margins, gmins, pmins, vqs):
        u = deepcopy(UNIVERSAL_BASE)
        u["polarity_gate_tau"] = float(tau)
        u["polarity_gate_margin"] = float(mar)
        u["polarity_gate_min_confidence"] = float(gmin)
        u["polarity_min_confidence"] = float(pmin)
        u["polarity_vote_q"] = float(vq)
        trials.append(u)
    return trials


def _build_cfg(dataset: str, seed: int, universal: Dict[str, Any], exp_tag: str) -> Dict[str, Any]:
    base_path = REPO / "configs" / f"{dataset}_best.yaml"
    cfg = _load_yaml(base_path)
    cfg["dataset"] = dataset
    for k, v in universal.items():
        cfg[k] = v
    cfg["exp_tag"] = exp_tag
    return cfg


def _one_task(args: Tuple[int, Dict[str, Any], str, int, str, str, str]) -> Dict[str, Any]:
    trial_id, universal, dataset, seed, run_tag, gpu, root_s = args
    root = Path(root_s)
    tdir = root / f"trial_{trial_id:04d}"
    tdir.mkdir(parents=True, exist_ok=True)
    exp_tag = f"univ_tune_t{trial_id:04d}_{dataset}_s{seed}_{run_tag}"
    cfg = _build_cfg(dataset, seed, universal, exp_tag)
    cfg_path = tdir / f"{dataset}_seed{seed}.yaml"
    result_path = tdir / f"{dataset}_seed{seed}.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    py = Path(sys.executable)
    cmd = [
        str(py),
        str(REPO / "main_train.py"),
        "--config",
        str(cfg_path),
        "--seed",
        str(seed),
        "--result-file",
        str(result_path),
    ]
    t0 = time.perf_counter()
    p = subprocess.run(
        cmd,
        cwd=str(REPO),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    elapsed = time.perf_counter() - t0
    log_path = tdir / f"{dataset}_seed{seed}.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(p.stdout or "")

    row: Dict[str, Any] = {
        "trial_id": trial_id,
        "dataset": dataset,
        "seed": seed,
        "gpu": gpu,
        "returncode": p.returncode,
        "elapsed_sec": elapsed,
        "result_file": str(result_path),
    }
    if p.returncode == 0 and result_path.is_file():
        with open(result_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        row["auc_mean"] = payload.get("auc_mean")
    else:
        row["auc_mean"] = None
        row["error"] = "failed or missing json"
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=("small", "medium", "large"), default="medium")
    ap.add_argument("--seeds", type=str, default="42,0,1,2,3")
    ap.add_argument("--datasets", type=str, default=",".join(DATASETS))
    ap.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7")
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    dsets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    gpu_list = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpu_list:
        gpu_list = ["0"]

    trials_univ = _trial_grid(args.grid)
    run_tag = time.strftime("%Y%m%d_%H%M%S")
    root = REPO / "results" / "universal_param_tune" / run_tag

    jobs: List[Tuple[int, Dict[str, Any], str, int, str, str, str]] = []
    idx = 0
    for tid, univ in enumerate(trials_univ):
        for d in dsets:
            for s in seeds:
                g = gpu_list[idx % len(gpu_list)]
                jobs.append((tid, univ, d, s, run_tag, g, str(root)))
                idx += 1

    meta = {
        "run_tag": run_tag,
        "root": str(root),
        "n_trials": len(trials_univ),
        "n_jobs": len(jobs),
        "grid": args.grid,
        "seeds": seeds,
        "datasets": dsets,
    }
    print(json.dumps(meta, indent=2), flush=True)

    if args.dry_run:
        print("DRY RUN", len(jobs), "tasks (no files written)", flush=True)
        return

    root.mkdir(parents=True, exist_ok=True)
    trial_defs = []
    for tid, univ in enumerate(trials_univ):
        path = root / f"trial_{tid:04d}_universal.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(univ, f, indent=2, sort_keys=True)
        trial_defs.append({"trial_id": tid, "universal_json": str(path)})
    with open(root / "trial_index.json", "w", encoding="utf-8") as f:
        json.dump({"run_tag": run_tag, "grid": args.grid, "trials": trial_defs}, f, indent=2)
    with open(root / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    rows: List[Dict[str, Any]] = []
    mw = min(args.max_workers, len(jobs), max(len(gpu_list), 1))
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=mw) as ex:
        futs = {ex.submit(_one_task, j): j for j in jobs}
        for k, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            r = rows[-1]
            print(
                f"[{k}/{len(jobs)}] trial={r['trial_id']} {r['dataset']} seed={r['seed']} "
                f"rc={r['returncode']} auc={r.get('auc_mean')} ({r.get('elapsed_sec', 0):.1f}s)",
                flush=True,
            )
            if k % 25 == 0:
                _write_partial_leaderboard(root, rows, trials_univ)

    elapsed = time.perf_counter() - t0
    _write_final_summary(root, rows, trials_univ, elapsed)


def _write_partial_leaderboard(root: Path, rows: List[Dict[str, Any]], trials_univ: List[Dict[str, Any]]) -> None:
    by_t: Dict[int, List[float]] = {}
    for r in rows:
        if r.get("auc_mean") is None:
            continue
        by_t.setdefault(int(r["trial_id"]), []).append(float(r["auc_mean"]))
    lines = []
    for tid, aucs in sorted(by_t.items()):
        if len(aucs) < 25:
            continue
        lines.append({"trial_id": tid, "mean_auc": sum(aucs) / len(aucs), "n": len(aucs)})
    lines.sort(key=lambda x: -x["mean_auc"])
    with open(root / "leaderboard_partial.json", "w", encoding="utf-8") as f:
        json.dump(lines[:20], f, indent=2)


def _write_final_summary(
    root: Path,
    rows: List[Dict[str, Any]],
    trials_univ: List[Dict[str, Any]],
    elapsed: float,
) -> None:
    by_t: Dict[int, List[Dict[str, Any]]] = {}
    for r in rows:
        tid = int(r["trial_id"])
        by_t.setdefault(tid, []).append(r)

    summaries = []
    for tid, rs in sorted(by_t.items()):
        aucs = [float(x["auc_mean"]) for x in rs if x.get("auc_mean") is not None]
        failed = sum(1 for x in rs if x.get("auc_mean") is None)
        mean_auc = sum(aucs) / len(aucs) if aucs else None
        per_ds: Dict[str, List[float]] = {}
        for x in rs:
            if x.get("auc_mean") is None:
                continue
            per_ds.setdefault(x["dataset"], []).append(float(x["auc_mean"]))
        mean_per_ds = {d: sum(v) / len(v) for d, v in per_ds.items()}
        summaries.append(
            {
                "trial_id": tid,
                "mean_auc_all": mean_auc,
                "mean_auc_per_dataset": mean_per_ds,
                "n_ok": len(aucs),
                "n_failed": failed,
                "universal": trials_univ[tid],
            }
        )

    summaries.sort(key=lambda x: (x["mean_auc_all"] is not None, x["mean_auc_all"] or 0.0), reverse=True)
    out = {
        "elapsed_sec": elapsed,
        "ranked_trials": summaries,
        "best": summaries[0] if summaries else None,
    }
    with open(root / "summary_ranked.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    best = out["best"]
    print("\n=== DONE ===", flush=True)
    print(f"elapsed_sec={elapsed:.1f}", flush=True)
    if best:
        print(f"BEST trial_id={best['trial_id']} mean_auc_all={best['mean_auc_all']:.6f}", flush=True)
        print("universal overrides (full):", flush=True)
        print(json.dumps(best["universal"], indent=2, sort_keys=True), flush=True)

    md = root / "summary_ranked.md"
    with open(md, "w", encoding="utf-8") as f:
        f.write("# Universal polarity param tune\n\n")
        f.write(f"- elapsed: {elapsed:.1f}s\n")
        if best:
            f.write(f"- **best trial_id**: {best['trial_id']}\n")
            f.write(f"- **mean AUROC (25 runs)**: {best['mean_auc_all']:.6f}\n\n")
            f.write("```json\n")
            f.write(json.dumps(best["universal"], indent=2, sort_keys=True))
            f.write("\n```\n\n")
        f.write("| rank | trial_id | mean_auc | n_ok |\n")
        f.write("|---:|---:|---:|---:|\n")
        for i, s in enumerate(summaries[:30], 1):
            f.write(
                f"| {i} | {s['trial_id']} | {s['mean_auc_all']} | {s['n_ok']} |\n"
            )
    print("Wrote", root / "summary_ranked.json", md, flush=True)


if __name__ == "__main__":
    main()
