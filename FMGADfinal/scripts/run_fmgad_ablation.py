#!/usr/bin/env python3
import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


DATASET_CONFIG = {
    "books": "configs/books_best.yaml",
    "disney": "configs/disney_best.yaml",
    "enron": "configs/enron_best.yaml",
    "reddit": "configs/reddit_best.yaml",
    "weibo": "configs/weibo_best.yaml",
}

DEFAULT_SEEDS = [42, 0, 1, 2, 3]

VARIANT_OVERRIDES = {
    "full_fmgad": {},
    "wo_residual": {"residual_scale": 0.0},
    "wo_proto": {"use_proto": False},
    "wo_guidance": {"weight": 0.0},
    "wo_smooth": {"use_score_smoothing": False, "score_smoothing_alpha": 0.0},
    "wo_polarity": {
        "polarity_adapter": "none",
        "nk_polarity": False,
        "smoothgnn_polarity": False,
    },
    "wo_virtual_neighbor": {"use_virtual_neighbors": False},
}

REQUIRED_VARIANTS = [
    "full_fmgad",
    "wo_residual",
    "wo_proto",
    "wo_guidance",
    "wo_smooth",
    "wo_polarity",
]


def _parse_csv_list(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _load_yaml(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _save_yaml(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=False)


def _append_error(err_path: Path, msg: str) -> None:
    err_path.parent.mkdir(parents=True, exist_ok=True)
    with open(err_path, "a", encoding="utf-8") as f:
        f.write(msg.rstrip() + "\n")


def _sanity_static_source(repo_root: Path) -> None:
    model_path = repo_root / "model.py"
    txt = model_path.read_text(encoding="utf-8")
    if "num_steps = 1" not in txt:
        raise RuntimeError("Sanity check failed: inference step is not hard-fixed to 1 in model.py")
    bad_patterns = [
        "max(auc, 1-auc)",
        "max(auc, 1 - auc)",
        "max(pyg_auc, 1-pyg_auc)",
        "max(pyg_auc, 1 - pyg_auc)",
    ]
    low = txt.replace(" ", "").lower()
    for p in bad_patterns:
        if p.replace(" ", "").lower() in low:
            raise RuntimeError(f"Sanity check failed: found forbidden pattern `{p}` in model.py")


def _build_override(base_cfg: Dict, variant: str, dataset: str, seed: int, run_tag: str) -> Dict:
    cfg = deepcopy(base_cfg)
    ov = deepcopy(VARIANT_OVERRIDES[variant])

    # Global protocol constraints.
    ov["sample_steps"] = 1
    ov["exp_tag"] = f"ablation_{variant}_{dataset}_seed{seed}_{run_tag}"

    cfg.update(ov)
    return cfg


def _sanity_variant_override(variant: str, cfg: Dict) -> None:
    if int(cfg.get("sample_steps", -1)) != 1:
        raise RuntimeError(f"{variant}: inference_steps must be 1")

    if variant == "wo_polarity":
        if bool(cfg.get("nk_polarity", True)) or bool(cfg.get("smoothgnn_polarity", True)):
            raise RuntimeError("wo_polarity: polarity flags must be disabled")
        if str(cfg.get("polarity_adapter", "nk")) == "auto_vote":
            raise RuntimeError("wo_polarity: polarity_adapter cannot be auto_vote")

    if variant == "wo_proto" and bool(cfg.get("use_proto", True)):
        raise RuntimeError("wo_proto: use_proto must be False")

    if variant == "wo_smooth":
        if bool(cfg.get("use_score_smoothing", True)):
            raise RuntimeError("wo_smooth: use_score_smoothing must be False")
        if float(cfg.get("score_smoothing_alpha", 1.0)) != 0.0:
            raise RuntimeError("wo_smooth: score_smoothing_alpha must be 0.0")


def _task_run(
    repo_root: Path,
    result_root: Path,
    task: Tuple[str, str, int, int],
    force: bool,
    errors_log: Path,
) -> None:
    variant, dataset, seed, gpu = task
    out_dir = result_root / variant / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"seed_{seed}.json"
    out_log = out_dir / f"seed_{seed}.log"
    tmp_cfg = result_root / "_tmp_configs" / variant / dataset / f"seed_{seed}.yaml"

    if out_json.exists() and not force:
        return

    base_cfg = _load_yaml(repo_root / DATASET_CONFIG[dataset])
    run_tag = str(int(time.time()))
    merged_cfg = _build_override(base_cfg, variant, dataset, seed, run_tag)
    _sanity_variant_override(variant, merged_cfg)
    _save_yaml(tmp_cfg, merged_cfg)

    cmd = [
        sys.executable,
        "main_train.py",
        "--device",
        str(gpu),
        "--config",
        str(tmp_cfg),
        "--seed",
        str(seed),
        "--result-file",
        str(out_json),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("FMGAD_POLARITY_DEBUG", "0")

    with open(out_log, "w", encoding="utf-8") as f:
        p = subprocess.run(
            cmd,
            cwd=str(repo_root),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=False,
        )

    if p.returncode != 0:
        _append_error(errors_log, f"[FAIL] variant={variant} dataset={dataset} seed={seed} gpu={gpu} rc={p.returncode}")
        return

    if not out_json.exists():
        _append_error(errors_log, f"[FAIL] missing result json: variant={variant} dataset={dataset} seed={seed}")
        return

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    log_text = out_log.read_text(encoding="utf-8", errors="ignore")

    # Runtime sanity checks requested by user.
    if "steps:1" not in log_text:
        _append_error(errors_log, f"[FAIL] steps!=1 in log: variant={variant} dataset={dataset} seed={seed}")
        return
    if variant == "wo_proto" and "Training FM proto model..." in log_text:
        _append_error(errors_log, f"[FAIL] wo_proto still trained proto branch: dataset={dataset} seed={seed}")
        return

    payload["dataset"] = dataset
    payload["variant"] = variant
    payload["seed"] = int(seed)
    payload["config_overrides"] = VARIANT_OVERRIDES[variant]
    payload["inference_steps"] = 1
    payload["polarity_enabled"] = bool(merged_cfg.get("nk_polarity", False) or merged_cfg.get("smoothgnn_polarity", False))
    payload["smoothing_enabled"] = bool(merged_cfg.get("use_score_smoothing", False))
    payload["proto_enabled"] = bool(merged_cfg.get("use_proto", True))
    payload["residual_scale"] = float(merged_cfg.get("residual_scale", 10.0))
    payload["guidance_weight"] = float(merged_cfg.get("weight", 1.0))

    # Normalize metric keys.
    if "auc" not in payload and "auc_mean" in payload:
        payload["auc"] = float(payload["auc_mean"])
    if "ap" not in payload and "ap_mean" in payload:
        payload["ap"] = float(payload["ap_mean"])

    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _worker(
    repo_root: Path,
    result_root: Path,
    q: "queue.Queue[Tuple[str, str, int, int]]",
    force: bool,
    errors_log: Path,
) -> None:
    while True:
        try:
            task = q.get_nowait()
        except queue.Empty:
            return
        try:
            _task_run(repo_root, result_root, task, force, errors_log)
        except Exception as e:
            variant, dataset, seed, gpu = task
            _append_error(
                errors_log,
                f"[EXCEPTION] variant={variant} dataset={dataset} seed={seed} gpu={gpu} error={repr(e)}",
            )
        finally:
            q.task_done()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=str, default=".")
    ap.add_argument("--variants", type=str, default=",".join(REQUIRED_VARIANTS))
    ap.add_argument("--include-optional-virtual-neighbor", action="store_true")
    ap.add_argument("--datasets", type=str, default="books,disney,enron,reddit,weibo")
    ap.add_argument("--seeds", type=str, default="42,0,1,2,3")
    ap.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7")
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    result_root = repo_root / "results" / "ablation"
    result_root.mkdir(parents=True, exist_ok=True)
    errors_log = result_root / "errors.log"

    _sanity_static_source(repo_root)

    variants = _parse_csv_list(args.variants)
    if args.include_optional_virtual_neighbor and "wo_virtual_neighbor" not in variants:
        variants.append("wo_virtual_neighbor")
    for v in variants:
        if v not in VARIANT_OVERRIDES:
            raise ValueError(f"Unknown variant: {v}")

    datasets = _parse_csv_list(args.datasets)
    for d in datasets:
        if d not in DATASET_CONFIG:
            raise ValueError(f"Unknown dataset: {d}")

    seeds = [int(x) for x in _parse_csv_list(args.seeds)]
    gpus = [int(x) for x in _parse_csv_list(args.gpus)]
    if not gpus:
        raise ValueError("No GPUs provided")

    tasks: List[Tuple[str, str, int, int]] = []
    idx = 0
    for variant in variants:
        for dataset in datasets:
            for seed in seeds:
                tasks.append((variant, dataset, seed, gpus[idx % len(gpus)]))
                idx += 1

    q: "queue.Queue[Tuple[str, str, int, int]]" = queue.Queue()
    for t in tasks:
        q.put(t)

    workers = []
    n_workers = max(1, min(args.max_workers, len(tasks)))
    for _ in range(n_workers):
        th = threading.Thread(target=_worker, args=(repo_root, result_root, q, args.force, errors_log), daemon=True)
        th.start()
        workers.append(th)

    for th in workers:
        th.join()

    print(f"Done. Results root: {result_root}")
    print(f"Errors log: {errors_log}")


if __name__ == "__main__":
    main()
