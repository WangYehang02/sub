#!/usr/bin/env python3
"""
对比「仅 Rule1」与「仅 Rule2」极性路径在五数据集上的表现。

- Rule1：polarity_adapter=nk（非 auto_vote/universal），并打开 smoothgnn_polarity + nk_polarity，
  使分数在 score_mode 前走 calibrate_polarity_robust / calibrate_polarity_with_neighbor_knowledge。
- Rule2：polarity_adapter=auto_vote，关闭 smoothgnn_polarity / nk_polarity，仅末端 auto_vote。

各数据集仍以 configs/{dataset}_best.yaml 为基底，只覆盖上述字段；其余训练超参不变。
"""
from __future__ import annotations

import argparse
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
DEFAULT_SEEDS = (42, 0, 1, 2, 3)

RULE1_OVR = {
    "polarity_adapter": "nk",
    "smoothgnn_polarity": True,
    "nk_polarity": True,
}

RULE2_OVR = {
    "polarity_adapter": "auto_vote",
    "smoothgnn_polarity": False,
    "nk_polarity": False,
    "polarity_vote_q": 0.10,
    "polarity_vote_margin": 1,
    "polarity_min_confidence": 0.10,
    "polarity_lcc_rho_strong": 0.04,
    "polarity_deg_rho_strong": 0.04,
    "polarity_connectivity_rel_gap": 0.02,
    "lcc_spearman_threshold": -0.05,
    "polarity_verbose": False,
}


def _load_yaml(p: Path) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.Loader)


def _one_job(args: Tuple[str, str, int, str, str, str]) -> Dict[str, Any]:
    rule_tag, dataset, seed, run_tag, gpu, root_s = args
    root = Path(root_s)
    root.mkdir(parents=True, exist_ok=True)
    base = _load_yaml(REPO / "configs" / f"{dataset}_best.yaml")
    cfg = deepcopy(base)
    cfg["dataset"] = dataset
    ovr = RULE1_OVR if rule_tag == "rule1" else RULE2_OVR
    cfg.update(ovr)
    cfg["exp_tag"] = f"rulebench_{rule_tag}_{dataset}_s{seed}_{run_tag}"

    cfg_path = root / f"{rule_tag}_{dataset}_seed{seed}.yaml"
    result_path = root / f"{rule_tag}_{dataset}_seed{seed}.json"
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
    log_path = root / f"{rule_tag}_{dataset}_seed{seed}.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(p.stdout or "")

    row: Dict[str, Any] = {
        "rule": rule_tag,
        "dataset": dataset,
        "seed": seed,
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
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default=",".join(map(str, DEFAULT_SEEDS)))
    ap.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7")
    ap.add_argument("--max-workers", type=int, default=8)
    args = ap.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    gpu_list = [g.strip() for g in args.gpus.split(",") if g.strip()] or ["0"]
    run_tag = time.strftime("%Y%m%d_%H%M%S")
    root = REPO / "results" / "rule1_rule2_five_bench" / run_tag
    root.mkdir(parents=True, exist_ok=True)

    jobs: List[Tuple[str, str, int, str, str, str]] = []
    idx = 0
    for rule in ("rule1", "rule2"):
        for d in DATASETS:
            for s in seeds:
                g = gpu_list[idx % len(gpu_list)]
                jobs.append((rule, d, s, run_tag, g, str(root)))
                idx += 1

    rows: List[Dict[str, Any]] = []
    mw = min(args.max_workers, len(jobs), len(gpu_list))
    with ProcessPoolExecutor(max_workers=max(1, mw)) as ex:
        for fut in as_completed([ex.submit(_one_job, j) for j in jobs]):
            rows.append(fut.result())
            r = rows[-1]
            print(
                f"[{len(rows)}/{len(jobs)}] {r['rule']} {r['dataset']} s={r['seed']} "
                f"auc={r.get('auc_mean')} rc={r['returncode']}",
                flush=True,
            )

    # 汇总
    by_rule_ds: Dict[str, Dict[str, List[float]]] = {"rule1": {}, "rule2": {}}
    for r in rows:
        if r.get("auc_mean") is None:
            continue
        by_rule_ds.setdefault(r["rule"], {}).setdefault(r["dataset"], []).append(float(r["auc_mean"]))

    summary: Dict[str, Any] = {"run_tag": run_tag, "root": str(root), "per_dataset_mean_auc": {}, "macro_mean_auc": {}}
    for rule in ("rule1", "rule2"):
        ds_means = {}
        all_aucs: List[float] = []
        for d in DATASETS:
            xs = by_rule_ds.get(rule, {}).get(d, [])
            if xs:
                m = sum(xs) / len(xs)
                ds_means[d] = m
                all_aucs.extend(xs)
        summary["per_dataset_mean_auc"][rule] = ds_means
        summary["macro_mean_auc"][rule] = sum(all_aucs) / len(all_aucs) if all_aucs else None

    with open(root / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": sorted(rows, key=lambda x: (x["rule"], x["dataset"], x["seed"]))}, f, indent=2)

    # markdown 表
    md = root / "summary.md"
    with open(md, "w", encoding="utf-8") as f:
        f.write("# Rule1 vs Rule2（五数据集 × seeds）\n\n")
        f.write(f"目录: `{root}`\n\n")
        f.write("| dataset | Rule1 mean AUC | Rule2 mean AUC |\n")
        f.write("|---|---:|---:|\n")
        for d in DATASETS:
            m1 = summary["per_dataset_mean_auc"]["rule1"].get(d)
            m2 = summary["per_dataset_mean_auc"]["rule2"].get(d)
            f.write(f"| {d} | {m1} | {m2} |\n")
        f.write("\n")
        f.write(
            f"| **25-run 全局平均** | **{summary['macro_mean_auc']['rule1']}** | **{summary['macro_mean_auc']['rule2']}** |\n"
        )
    print("Wrote", root / "summary.json", md, flush=True)


if __name__ == "__main__":
    main()
