#!/usr/bin/env python3
"""
Five datasets × five seeds in parallel with polarity_adapter universal_no_y overrides.
Each job starts from configs/{dataset}.yaml and overlays universal fields and exp_tag.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

REPO = Path(__file__).resolve().parents[1]
DATASETS = ("books", "disney", "enron", "reddit", "weibo")
DEFAULT_SEEDS = (42, 3407, 2026, 17, 12345)

UNIVERSAL_OVR: Dict[str, Any] = {
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


def _build_cfg(dataset: str, seed: int, run_tag: str) -> Dict[str, Any]:
    base_path = REPO / "configs" / f"{dataset}.yaml"
    if not base_path.is_file():
        raise FileNotFoundError(base_path)
    cfg = _load_yaml(base_path)
    cfg["dataset"] = dataset
    for k, v in UNIVERSAL_OVR.items():
        cfg[k] = v
    cfg["exp_tag"] = f"univ5x5_{dataset}_seed{seed}_{run_tag}"
    return cfg


def _one_job(args: Tuple[str, int, str, str, int]) -> Dict[str, Any]:
    dataset, seed, run_tag, gpu, idx = args
    work = REPO / "results" / "universal_5x5_sweep" / "_tmp_cfgs"
    work.mkdir(parents=True, exist_ok=True)
    cfg_path = work / f"{dataset}_seed{seed}.yaml"
    out_dir = REPO / "results" / "universal_5x5_sweep" / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / f"{dataset}_seed{seed}.json"

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
    log_path = out_dir / f"{dataset}_seed{seed}.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(p.stdout or "")

    row: Dict[str, Any] = {
        "dataset": dataset,
        "seed": seed,
        "gpu": gpu,
        "returncode": p.returncode,
        "elapsed_sec": elapsed,
        "result_file": str(result_path),
        "log_file": str(log_path),
    }
    if p.returncode == 0 and result_path.is_file():
        with open(result_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        row["auc_mean"] = payload.get("auc_mean")
        row["ap_mean"] = payload.get("ap_mean")
        row["polarity_graph_signals"] = payload.get("polarity_graph_signals")
        ud = payload.get("universal_polarity_diagnostics") or {}
        row["universal_flipped"] = ud.get("flipped")
        row["universal_decision"] = ud.get("decision")
        row["universal_fallback"] = ud.get("fallback")
    else:
        row["auc_mean"] = None
        row["error"] = "failed or missing json"
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=str, default=None, help="Comma-separated GPU ids; default cycles 0..n-1")
    ap.add_argument("--seeds", type=str, default=",".join(map(str, DEFAULT_SEEDS)))
    ap.add_argument("--datasets", type=str, default=",".join(DATASETS))
    ap.add_argument("--max-workers", type=int, default=None, help="Parallel workers; default matches GPU list size")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    dsets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    for d in dsets:
        if d not in DATASETS:
            raise SystemExit(f"unknown dataset: {d}")

    import torch

    n_gpu = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    if args.gpus:
        gpu_list = [g.strip() for g in args.gpus.split(",") if g.strip()]
    else:
        gpu_list = [str(i % max(n_gpu, 1)) for i in range(len(dsets) * len(seeds))]

    run_tag = time.strftime("%Y%m%d_%H%M%S")
    jobs: List[Tuple[str, int, str, str, int]] = []
    idx = 0
    for d in dsets:
        for s in seeds:
            g = gpu_list[idx % len(gpu_list)]
            jobs.append((d, s, run_tag, g, idx))
            idx += 1

    out_root = REPO / "results" / "universal_5x5_sweep"
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / f"summary_{run_tag}.json"

    if args.dry_run:
        print("DRY RUN jobs:", len(jobs))
        for j in jobs[:10]:
            print(j)
        return

    max_workers = args.max_workers or max(len(gpu_list), 1)
    max_workers = min(max_workers, len(jobs))

    rows: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_one_job, j): j for j in jobs}
        for fut in as_completed(futs):
            rows.append(fut.result())
            print(
                f"[done] {rows[-1]['dataset']} seed={rows[-1]['seed']} rc={rows[-1]['returncode']} auc={rows[-1].get('auc_mean')}",
                flush=True,
            )

    rows.sort(key=lambda r: (r["dataset"], r["seed"]))
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"run_tag": run_tag, "jobs": rows}, f, indent=2, ensure_ascii=False)

    md_path = out_root / f"summary_{run_tag}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Universal polarity 5×5 sweep\n\n")
        f.write(f"Summary: `{summary_path}`\n\n")
        f.write("| dataset | seed | AUC | AP | rc | flipped | decision | fallback |\n")
        f.write("|---|---:|---:|---:|---:|---|---|---|\n")
        for r in rows:
            f.write(
                f"| {r['dataset']} | {r['seed']} | {r.get('auc_mean')} | {r.get('ap_mean')} | {r['returncode']} | "
                f"{r.get('universal_flipped')} | {r.get('universal_decision')} | {r.get('universal_fallback')} |\n"
            )
    print("Wrote", summary_path, md_path, flush=True)


if __name__ == "__main__":
    main()
