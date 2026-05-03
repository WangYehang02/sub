#!/usr/bin/env python3
"""
五数据集 × 多 seed：直接使用各 configs/{dataset}.yaml（不在此脚本里改极性超参），
跑 main_train 并汇总 auc_mean / ap_mean。

默认 seeds=42,0,1,2,3；结果写入 results/best_yaml_5x5_sweep/<run_tag>/。
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

REPO = Path(__file__).resolve().parents[1]
DATASETS = ("books", "disney", "enron", "reddit", "weibo")


def _mean_std(xs: List[float]) -> Tuple[float, float]:
    """样本均值与样本标准差（n=1 时 std 记为 0）。"""
    if not xs:
        return float("nan"), float("nan")
    m = float(statistics.mean(xs))
    if len(xs) < 2:
        return m, 0.0
    return m, float(statistics.stdev(xs))


def _load_yaml(p: Path) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.Loader)


def _build_cfg(dataset: str, seed: int, run_tag: str) -> Dict[str, Any]:
    base_path = REPO / "configs" / f"{dataset}.yaml"
    if not base_path.is_file():
        raise FileNotFoundError(base_path)
    cfg = _load_yaml(base_path)
    cfg["dataset"] = dataset
    cfg["exp_tag"] = f"bestyaml5x5_{dataset}_seed{seed}_{run_tag}"
    return cfg


def _one_job(args: Tuple[str, int, str, str, Path]) -> Dict[str, Any]:
    dataset, seed, run_tag, gpu, run_root = args
    work = run_root / "_tmp_cfgs"
    work.mkdir(parents=True, exist_ok=True)
    cfg_path = work / f"{dataset}_seed{seed}.yaml"
    result_path = run_root / "runs" / f"{dataset}_seed{seed}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = _build_cfg(dataset, seed, run_tag)
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
    log_path = run_root / "runs" / f"{dataset}_seed{seed}.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(p.stdout or "")

    row: Dict[str, Any] = {
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
        row["ap_mean"] = payload.get("ap_mean")
    else:
        row["auc_mean"] = None
        row["ap_mean"] = None
        row["error"] = "failed or missing json"
    return row


def _write_summary_artifacts(run_root: Path, run_tag: str, rows: List[Dict[str, Any]]) -> None:
    rows = sorted(rows, key=lambda r: (r["dataset"], r["seed"]))
    by_d_auc: Dict[str, List[float]] = {}
    by_d_ap: Dict[str, List[float]] = {}
    all_auc: List[float] = []
    all_ap: List[float] = []
    for r in rows:
        if r.get("auc_mean") is not None:
            au = float(r["auc_mean"])
            by_d_auc.setdefault(r["dataset"], []).append(au)
            all_auc.append(au)
        if r.get("ap_mean") is not None:
            apv = float(r["ap_mean"])
            by_d_ap.setdefault(r["dataset"], []).append(apv)
            all_ap.append(apv)

    per_dataset: Dict[str, Dict[str, Any]] = {}
    for d in sorted(set(by_d_auc.keys()) | set(by_d_ap.keys())):
        aucs = by_d_auc.get(d, [])
        aps = by_d_ap.get(d, [])
        am, asd = _mean_std(aucs)
        pm, psd = _mean_std(aps)
        per_dataset[d] = {
            "n": len(aucs),
            "auc_mean": am,
            "auc_std": asd,
            "ap_mean": pm,
            "ap_std": psd,
        }

    g_am, g_asd = _mean_std(all_auc)
    g_pm, g_psd = _mean_std(all_ap)

    summary = {
        "run_tag": run_tag,
        "per_dataset_mean_std": per_dataset,
        "all_runs_mean_std": {
            "n": len(all_auc),
            "auc_mean": g_am,
            "auc_std": g_asd,
            "ap_mean": g_pm,
            "ap_std": g_psd,
        },
        "jobs": rows,
    }
    summary_path = run_root / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    md_path = run_root / "summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Best YAML 五数据集 × seeds AUC / AP（均值 ± 标准差）\n\n")
        f.write(f"- run_tag: `{run_tag}`\n")
        f.write(f"- 配置来源: `configs/{{dataset}}.yaml`（仅覆盖 `exp_tag`）\n")
        f.write(
            f"- **全部运行（跨数据集×seed）**: AUROC {g_am:.4f} ± {g_asd:.4f}，"
            f"AP {g_pm:.4f} ± {g_psd:.4f}（n={len(all_auc)}）\n\n"
        )
        f.write("## 各数据集（跨 seeds：样本标准差）\n\n")
        f.write("| dataset | AUC (mean ± std) | AP (mean ± std) | n |\n")
        f.write("|---|---|---|---:|\n")
        for d in sorted(per_dataset.keys()):
            s = per_dataset[d]
            f.write(
                f"| {d} | {s['auc_mean']:.4f} ± {s['auc_std']:.4f} | "
                f"{s['ap_mean']:.4f} ± {s['ap_std']:.4f} | {s['n']} |\n"
            )
        f.write("\n## 明细\n\n")
        f.write("| dataset | seed | AUC | AP | rc |\n|---|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['dataset']} | {r['seed']} | {r.get('auc_mean')} | {r.get('ap_mean')} | {r['returncode']} |\n"
            )
    print("Wrote", summary_path, md_path, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--from-summary-json",
        type=str,
        default=None,
        help="给定已有 run 的 summary.json 路径，仅重算 mean±std 并覆盖同目录 summary.json / summary.md（不重训）。",
    )
    ap.add_argument("--seeds", type=str, default="42,0,1,2,3")
    ap.add_argument("--datasets", type=str, default=",".join(DATASETS))
    ap.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7")
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.from_summary_json:
        p = Path(args.from_summary_json).resolve()
        if not p.is_file():
            raise SystemExit(f"missing file: {p}")
        data = json.loads(p.read_text(encoding="utf-8"))
        rows = list(data.get("jobs", []))
        run_tag = str(data.get("run_tag") or p.parent.name)
        _write_summary_artifacts(p.parent, run_tag, rows)
        return

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    dsets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    gpu_list = [g.strip() for g in args.gpus.split(",") if g.strip()] or ["0"]

    run_tag = time.strftime("%Y%m%d_%H%M%S")
    run_root = REPO / "results" / "best_yaml_5x5_sweep" / run_tag
    run_root.mkdir(parents=True, exist_ok=True)

    jobs: List[Tuple[str, int, str, str, Path]] = []
    idx = 0
    for d in dsets:
        for s in seeds:
            g = gpu_list[idx % len(gpu_list)]
            jobs.append((d, s, run_tag, g, run_root))
            idx += 1

    meta = {"run_tag": run_tag, "root": str(run_root), "n_jobs": len(jobs), "seeds": seeds, "datasets": dsets}
    print(json.dumps(meta, indent=2), flush=True)

    if args.dry_run:
        print("DRY RUN", len(jobs))
        return

    mw = min(args.max_workers, len(jobs), max(len(gpu_list), 1))
    progress_path = run_root / "progress.jsonl"
    rows: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=mw) as ex:
        futs = {ex.submit(_one_job, j): j for j in jobs}
        for k, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            r = rows[-1]
            with open(progress_path, "a", encoding="utf-8") as pf:
                pf.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(
                f"[{k}/{len(jobs)}] {r['dataset']} seed={r['seed']} rc={r['returncode']} auc={r.get('auc_mean')}",
                flush=True,
            )

    _write_summary_artifacts(run_root, run_tag, rows)


if __name__ == "__main__":
    main()
