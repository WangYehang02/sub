import os
import sys


def _cuda_visible_devices_from_argv_early() -> None:
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return
    argv = sys.argv
    for i in range(len(argv) - 1):
        if argv[i] == "--device":
            os.environ["CUDA_VISIBLE_DEVICES"] = str(argv[i + 1])
            return


_cuda_visible_devices_from_argv_early()

import json
import time
import argparse
import random

import numpy as np
import torch
import yaml

from model import ResFlowGAD


def get_arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=int, default=0)
    p.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "books.yaml"),
    )
    p.add_argument("--result-file", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_trial", type=int, default=None)
    p.add_argument("--deterministic", action="store_true", help="Enable deterministic PyTorch settings")
    p.add_argument("--profile-efficiency", action="store_true", help="Enable efficiency profiling fields in result json")
    return p.parse_args()


def _set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("PYTHONHASHSEED", str(seed))
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = False

        torch.use_deterministic_algorithms(True, warn_only=True)

    print("Deterministic mode:", deterministic, flush=True)
    print("CUBLAS_WORKSPACE_CONFIG:", os.environ.get("CUBLAS_WORKSPACE_CONFIG"), flush=True)
    print("torch deterministic algorithms:", torch.are_deterministic_algorithms_enabled(), flush=True)


def main():
    args = get_arguments()
    print("Random seed:", args.seed, flush=True)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    dset = cfg["dataset"]

    ae_alpha = cfg.get("ae_alpha", 0.8)
    if ae_alpha == 0.0:
        ae_alpha = 0.9

    _set_seed(args.seed, deterministic=args.deterministic)
    model = ResFlowGAD(
        hid_dim=cfg.get("hid_dim") if cfg.get("hid_dim") else None,
        ae_dropout=cfg["ae_dropout"],
        ae_lr=cfg["ae_lr"],
        ae_alpha=ae_alpha,
        use_proto=bool(cfg.get("use_proto", True)),
        profile_efficiency=bool(args.profile_efficiency),
        proto_alpha=cfg.get("proto_alpha", 0.01),
        weight=cfg.get("weight", 1.0),
        residual_scale=float(cfg.get("residual_scale", 10.0)),
        sample_steps=int(cfg.get("sample_steps", 1)),
        verbose=True,
        use_virtual_neighbors=cfg.get("use_virtual_neighbors", True),
        virtual_degree_threshold=int(cfg.get("virtual_degree_threshold", 5)),
        virtual_k=int(cfg.get("virtual_k", 5)),
        use_score_smoothing=cfg.get("use_score_smoothing", True),
        score_smoothing_alpha=float(cfg.get("score_smoothing_alpha", 0.3)),
        flow_t_sampling=cfg.get("flow_t_sampling", "logit_normal"),
        ensemble_score=cfg.get("ensemble_score", True),
        num_trial=args.num_trial if args.num_trial is not None else int(cfg.get("num_trial", 3)),
        exp_tag=cfg.get("exp_tag", None),
        smoothgnn_polarity=cfg.get("smoothgnn_polarity", False),
        smoothgnn_anchor_k_percent=float(cfg.get("smoothgnn_anchor_k_percent", 0.05)),
        smoothgnn_anchor_margin=float(cfg.get("smoothgnn_anchor_margin", 1.05)),
        smoothgnn_robust_spearman_threshold=float(cfg.get("smoothgnn_robust_spearman_threshold", -0.1)),
        nk_polarity=cfg.get("nk_polarity", False),
        nk_feature_weight=float(cfg.get("nk_feature_weight", 0.8)),
        nk_degree_weight=float(cfg.get("nk_degree_weight", 0.2)),
        nk_min_flip_gain=float(cfg.get("nk_min_flip_gain", 0.02)),
        score_mode=cfg.get("score_mode", "calibrated"),
        score_mode_beta=float(cfg.get("score_mode_beta", 0.05)),
        polarity_reg_weight=float(cfg.get("polarity_reg_weight", 0.0)),
        polarity_reg_target_corr=float(cfg.get("polarity_reg_target_corr", 0.1)),
        polarity_adapter=cfg.get("polarity_adapter", "universal_no_y"),
        polarity_vote_q=float(cfg.get("polarity_vote_q", 0.1)),
        polarity_vote_margin=int(cfg.get("polarity_vote_margin", 1)),
        polarity_min_confidence=float(cfg.get("polarity_min_confidence", 0.2)),
        polarity_lcc_rho_strong=float(cfg.get("polarity_lcc_rho_strong", 0.04)),
        polarity_deg_rho_strong=float(cfg.get("polarity_deg_rho_strong", 0.04)),
        polarity_connectivity_rel_gap=float(cfg.get("polarity_connectivity_rel_gap", 0.02)),
        lcc_spearman_threshold=float(cfg.get("lcc_spearman_threshold", -0.05)),
        polarity_verbose=bool(cfg.get("polarity_verbose", False)),
        polarity_use_local_probe=bool(cfg.get("polarity_use_local_probe", True)),
        polarity_use_nk_probe=bool(cfg.get("polarity_use_nk_probe", True)),
        polarity_use_structural_probe=bool(cfg.get("polarity_use_structural_probe", True)),
        polarity_gate_tau=float(cfg.get("polarity_gate_tau", 0.05)),
        polarity_gate_margin=float(cfg.get("polarity_gate_margin", 0.02)),
        polarity_gate_min_confidence=float(cfg.get("polarity_gate_min_confidence", 0.10)),
        polarity_gate_topk_percent=float(cfg.get("polarity_gate_topk_percent", 0.05)),
        polarity_struct_lcc_threshold=float(cfg.get("polarity_struct_lcc_threshold", 0.04)),
        polarity_struct_deg_threshold=float(cfg.get("polarity_struct_deg_threshold", 0.04)),
        polarity_struct_density_gap=float(cfg.get("polarity_struct_density_gap", 0.02)),
        polarity_autovote_fallback=bool(cfg.get("polarity_autovote_fallback", True)),
        polarity_unsup_proxy_q=float(cfg.get("polarity_unsup_proxy_q", 0.05)),
    )

    print("Running on dataset:", dset, "num_trial:", model.num_trial, flush=True)
    def _sync_cuda():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    _sync_cuda()
    t0 = time.perf_counter()
    _set_seed(args.seed, deterministic=args.deterministic)
    out = model(dset)
    _sync_cuda()
    elapsed = time.perf_counter() - t0
    print("FMGAD_TIME_SEC\t{:.1f}".format(elapsed), flush=True)
    if args.result_file:
        payload = {"dataset": dset, "seed": int(args.seed), "time_sec": elapsed, **out}
        if args.profile_efficiency:
            payload["profile_efficiency"] = True
            payload["total_time_sec"] = float(elapsed)
            payload.setdefault("train_time_sec", None)
            payload.setdefault("inference_time_sec", None)
            payload.setdefault("peak_gpu_mem_mb", None)
            payload.setdefault("peak_gpu_reserved_mb", None)
            payload.setdefault("num_parameters", None)
            payload.setdefault("num_nodes", None)
            payload.setdefault("num_edges", None)
            payload.setdefault("num_features", None)
            payload.setdefault("sample_steps", int(cfg.get("sample_steps", 100)))
        if "auc_mean" in payload:
            payload["auc"] = float(payload["auc_mean"])
        with open(args.result_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    return out


if __name__ == "__main__":
    main()
