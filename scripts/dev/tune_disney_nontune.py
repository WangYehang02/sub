#!/usr/bin/env python3
"""
Disney-only hyperparameter search: maximize mean AUROC over fixed seeds.
All polarity / legacy flip-related YAML keys are frozen from configs/books.yaml
(same universal_no_y block as other datasets); only non-polarity knobs are searched.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

REPO = Path(__file__).resolve().parents[2]
DISNEY_BASE = REPO / "configs" / "disney.yaml"
POLARITY_REF = REPO / "configs" / "books.yaml"


def _resolve_storage_root(explicit: str | None) -> Path:
    """Default large runs under /mnt/yehang when writable; otherwise fall back to repo root."""
    if explicit:
        p = Path(explicit).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p
    mnt = Path("/mnt/yehang/FMGADfinal_runs")
    try:
        if mnt.parent.is_dir() and os.access(mnt.parent, os.W_OK):
            mnt.mkdir(parents=True, exist_ok=True)
            return mnt
    except OSError:
        pass
    return REPO


def _load_yaml(p: Path) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.Loader)


def _polarity_block_from_ref(ref: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in ref.items():
        if k.startswith("polarity_") or k.startswith("smoothgnn_") or k.startswith("nk_"):
            out[k] = copy.deepcopy(v)
        elif k == "lcc_spearman_threshold":
            out[k] = copy.deepcopy(v)
    return out


def _build_cfg(trial_id: int, tune: Dict[str, Any], run_tag: str, polar_frozen: Dict[str, Any]) -> Dict[str, Any]:
    cfg = _load_yaml(DISNEY_BASE)
    cfg["dataset"] = "disney"
    for k, v in tune.items():
        cfg[k] = v
    for k, v in polar_frozen.items():
        cfg[k] = v
    cfg["exp_tag"] = f"disney_tune_t{trial_id:04d}_{run_tag}"
    return cfg


def _one_job(args: Tuple[int, Dict[str, Any], int, str, str, Path, Dict[str, Any], str]) -> Dict[str, Any]:
    trial_id, tune, seed, run_tag, gpu, run_root, polar_frozen, model_root = args
    work = run_root / "_tmp_cfgs"
    work.mkdir(parents=True, exist_ok=True)
    cfg_path = work / f"t{trial_id:04d}_seed{seed}.yaml"
    result_path = run_root / "runs" / f"t{trial_id:04d}_seed{seed}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = _build_cfg(trial_id, tune, run_tag, polar_frozen)
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["FMGAD_MODEL_ROOT"] = model_root
    # Same exp_tag across seeds would race on proto_dm_self.pt etc.; suffix gives per-seed checkpoint dirs.
    env["FMGAD_RUN_TAG_SUFFIX"] = f"seed{seed}"
    cmd = [
        sys.executable,
        str(REPO / "main_train.py"),
        "--config",
        str(cfg_path),
        "--seed",
        str(seed),
        "--result-file",
        str(result_path),
    ]
    t0 = time.perf_counter()
    p = subprocess.run(cmd, cwd=str(REPO), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    elapsed = time.perf_counter() - t0
    log_path = run_root / "runs" / f"t{trial_id:04d}_seed{seed}.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(p.stdout or "")

    row: Dict[str, Any] = {
        "trial_id": trial_id,
        "seed": seed,
        "tune": tune,
        "gpu": gpu,
        "returncode": p.returncode,
        "elapsed_sec": elapsed,
    }
    if p.returncode == 0 and result_path.is_file():
        with open(result_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        row["auc_mean"] = payload.get("auc_mean")
    else:
        row["auc_mean"] = None
        row["error"] = "failed or missing json"
    return row


def _random_trials(n: int, rng: random.Random) -> List[Dict[str, Any]]:
    ae_alpha = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    ae_dropout = [0.25, 0.3, 0.35, 0.4, 0.45]
    ae_lr = [0.005, 0.01, 0.015, 0.02, 0.025, 0.035, 0.05]
    proto_alpha = [0.001, 0.003, 0.005, 0.01]
    residual_scale = [5.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0]
    weight = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5]
    use_virtual_neighbors = [True, False]
    sample_steps = [50, 75]
    trials: List[Dict[str, Any]] = []
    seen = set()
    while len(trials) < n:
        t = {
            "ae_alpha": rng.choice(ae_alpha),
            "ae_dropout": rng.choice(ae_dropout),
            "ae_lr": rng.choice(ae_lr),
            "proto_alpha": rng.choice(proto_alpha),
            "residual_scale": rng.choice(residual_scale),
            "weight": rng.choice(weight),
            "use_virtual_neighbors": rng.choice(use_virtual_neighbors),
            "use_score_smoothing": True,
            "sample_steps": rng.choice(sample_steps),
            "flow_t_sampling": "logit_normal",
            "hid_dim": None,
            "num_trial": 1,
        }
        key = tuple(sorted((k, str(v)) for k, v in t.items()))
        if key in seen:
            continue
        seen.add(key)
        trials.append(t)
    return trials


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=str, default="42,0,1,2,3", help="Comma-separated seeds (default: 42,0,1,2,3)")
    ap.add_argument("--n-trials", type=int, default=36, help="Number of random search trials (non-polarity)")
    ap.add_argument("--random-seed", type=int, default=0, help="RNG seed for constructing the trial list")
    ap.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7")
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Root for logs/json/tmp yaml and models/; default /mnt/yehang/FMGADfinal_runs if writable else repo root",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    gpu_list = [g.strip() for g in args.gpus.split(",") if g.strip()] or ["0"]
    rng = random.Random(int(args.random_seed))
    tune_list = _random_trials(int(args.n_trials), rng)
    polar_frozen = _polarity_block_from_ref(_load_yaml(POLARITY_REF))

    run_tag = time.strftime("%Y%m%d_%H%M%S")
    storage = _resolve_storage_root(args.output_root)
    if storage.resolve() == REPO.resolve():
        run_root = REPO / "results" / "tune_disney_nontune" / run_tag
        model_root = str(REPO / "models")
    else:
        run_root = storage / "tune_disney_nontune" / run_tag
        model_root = str(storage / "models")
    run_root.mkdir(parents=True, exist_ok=True)

    with open(run_root / "polarity_frozen_from_books.yaml", "w", encoding="utf-8") as f:
        yaml.dump(polar_frozen, f, default_flow_style=False, allow_unicode=False)

    jobs: List[Tuple[int, Dict[str, Any], int, str, str, Path, Dict[str, Any]]] = []
    idx = 0
    for tid, tune in enumerate(tune_list):
        for s in seeds:
            g = gpu_list[idx % len(gpu_list)]
            jobs.append((tid, tune, s, run_tag, g, run_root, polar_frozen, model_root))
            idx += 1

    meta = {
        "run_tag": run_tag,
        "storage_root": str(storage),
        "model_root": model_root,
        "root": str(run_root),
        "n_tune_trials": len(tune_list),
        "n_jobs": len(jobs),
        "seeds": seeds,
        "polarity_reference": str(POLARITY_REF),
    }
    print(json.dumps(meta, indent=2), flush=True)

    for tid, tune in enumerate(tune_list):
        with open(run_root / f"trial_{tid:04d}_tune.yaml", "w", encoding="utf-8") as f:
            yaml.dump(tune, f, default_flow_style=False, allow_unicode=False)

    if args.dry_run:
        print("DRY RUN", len(jobs), "jobs")
        return

    rows: List[Dict[str, Any]] = []
    mw = min(args.max_workers, len(jobs), max(len(gpu_list), 1))
    with ProcessPoolExecutor(max_workers=mw) as ex:
        futs = {ex.submit(_one_job, j): j for j in jobs}
        for k, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            r = rows[-1]
            print(
                f"[{k}/{len(jobs)}] trial={r['trial_id']} seed={r['seed']} rc={r['returncode']} auc={r.get('auc_mean')}",
                flush=True,
            )

    by_t: Dict[int, List[float]] = {}
    for r in rows:
        if r.get("auc_mean") is None:
            continue
        by_t.setdefault(int(r["trial_id"]), []).append(float(r["auc_mean"]))

    summaries = []
    for tid, tune in enumerate(tune_list):
        aucs = by_t.get(tid, [])
        summaries.append(
            {
                "trial_id": tid,
                "mean_auc": sum(aucs) / len(aucs) if aucs else None,
                "n_ok": len(aucs),
                "tune": tune,
            }
        )
    n_seeds = len(seeds)
    full_ok = [s for s in summaries if s.get("mean_auc") is not None and int(s.get("n_ok", 0)) == n_seeds]
    if full_ok:
        full_ok.sort(key=lambda x: float(x["mean_auc"]), reverse=True)
        best = full_ok[0]
        best_mode = "all_seeds_ok"
    else:
        summaries.sort(key=lambda x: (x["mean_auc"] is not None, x["mean_auc"] or 0.0), reverse=True)
        best = summaries[0] if summaries else None
        best_mode = "fallback_partial_runs"

    out = {
        "meta": meta,
        "ranked": summaries,
        "best": best,
        "best_selection_mode": best_mode if best else None,
        "ranked_full_seeds_only": sorted(
            [s for s in summaries if int(s.get("n_ok", 0)) == n_seeds and s.get("mean_auc") is not None],
            key=lambda x: float(x["mean_auc"]),
            reverse=True,
        ),
        "raw": rows,
    }
    with open(run_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    md = run_root / "summary.md"
    with open(md, "w", encoding="utf-8") as f:
        f.write("# Disney non-polarity tuning\n\n")
        f.write(f"- seeds: `{args.seeds}`\n")
        f.write(f"- polarity frozen from: `{POLARITY_REF}`\n")
        if best and best.get("mean_auc") is not None:
            f.write(f"- **selection**: `{out['best_selection_mode']}` (prefer trials with all {n_seeds} seeds)\n")
            f.write(f"- **best trial_id**: {best['trial_id']}\n")
            f.write(f"- **mean AUROC** (over successful seeds): {best['mean_auc']:.6f} (n_ok={best['n_ok']})\n\n")
            f.write("```yaml\n")
            f.write(yaml.dump(best["tune"], default_flow_style=False, allow_unicode=False))
            f.write("```\n\n")
        f.write("| rank | trial_id | mean_auc | n_ok |\n|---:|---:|---:|---:|\n")
        for i, s in enumerate(summaries[:25], 1):
            f.write(f"| {i} | {s['trial_id']} | {s.get('mean_auc')} | {s['n_ok']} |\n")

    if best and best.get("tune"):
        merged = _load_yaml(DISNEY_BASE)
        for k, v in best["tune"].items():
            merged[k] = v
        for k, v in polar_frozen.items():
            merged[k] = v
        merged["dataset"] = "disney"
        merged["exp_tag"] = "fmgad_disney"
        with open(run_root / "recommended_disney.yaml", "w", encoding="utf-8") as f:
            yaml.dump(merged, f, default_flow_style=False, allow_unicode=False)

    print("Wrote", run_root / "summary.json", md, flush=True)
    if best:
        print("BEST mean_auc=", best.get("mean_auc"), "trial_id=", best.get("trial_id"), flush=True)


if __name__ == "__main__":
    main()
