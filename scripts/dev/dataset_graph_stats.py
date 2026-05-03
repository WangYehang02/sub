#!/usr/bin/env python3
"""
Graph statistics for the five PyGOD benchmarks (same as FMGAD):
- Mean degree of anomaly nodes
- Mean degree over all nodes
- Mean degree of 1-hop neighbors of anomalies (per-anomaly neighbor-mean, then mean over anomalies)
- Mean cosine similarity between each anomaly and its 1-hop neighbors (same aggregation)

Supports multiprocessing with one GPU per worker (CUDA_VISIBLE_DEVICES) for all five datasets in parallel.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple


DATASETS = ("books", "disney", "enron", "reddit", "weibo")


def compute_node_degree_undirected(edge_index, num_nodes: int):
    import torch

    n = int(num_nodes)
    dev = edge_index.device
    one = torch.ones(edge_index.size(1), device=dev, dtype=torch.float32)
    deg = torch.zeros(n, device=dev, dtype=torch.float32)
    deg.index_add_(0, edge_index[0], one)
    deg.index_add_(0, edge_index[1], one)
    return deg


def analyze_dataset(dataset: str, device: torch.device) -> Dict[str, Any]:
    import warnings

    warnings.filterwarnings("ignore", category=FutureWarning)

    import torch
    import torch.nn.functional as F
    from pygod.utils import load_data

    data = load_data(dataset)
    n = int(data.x.size(0))
    edge_index = data.edge_index.to(device)
    x = data.x.to(device=device, dtype=torch.float32)
    y = data.y
    if y.dtype != torch.bool:
        y = y.bool()
    y = y.to(device)

    deg = compute_node_degree_undirected(edge_index, n)
    mean_deg_all = float(deg.mean().item())

    anom = y
    n_anom = int(anom.sum().item())
    if n_anom == 0:
        raise RuntimeError(f"{dataset}: no anomaly nodes in y")

    mean_deg_anom = float(deg[anom].mean().item())

    src, dst = edge_index[0], edge_index[1]
    # Undirected 1-hop neighbors: each (u,v) contributes v as neighbor of u and u as neighbor of v.
    neigh_deg_sum = torch.zeros(n, device=device, dtype=torch.float32)
    neigh_deg_sum.index_add_(0, src, deg[dst])
    neigh_deg_sum.index_add_(0, dst, deg[src])
    neigh_cnt = torch.zeros(n, device=device, dtype=torch.float32)
    ones = torch.ones(edge_index.size(1), device=device, dtype=torch.float32)
    neigh_cnt.index_add_(0, src, ones)
    neigh_cnt.index_add_(0, dst, ones)

    x_n = F.normalize(x, p=2, dim=1, eps=1e-8)
    cos_e = (x_n[src] * x_n[dst]).sum(dim=1)
    cos_sum = torch.zeros(n, device=device, dtype=torch.float32)
    cos_sum.index_add_(0, src, cos_e)
    cos_sum.index_add_(0, dst, cos_e)

    mask_anom = anom & (neigh_cnt > 0)
    n_anom_with_neigh = int(mask_anom.sum().item())
    n_anom_isolated = n_anom - n_anom_with_neigh

    if n_anom_with_neigh > 0:
        mean_neigh_deg_for_anom = float(
            (neigh_deg_sum[mask_anom] / neigh_cnt[mask_anom]).mean().item()
        )
        mean_cos_anom_neigh = float(
            (cos_sum[mask_anom] / neigh_cnt[mask_anom]).mean().item()
        )
    else:
        mean_neigh_deg_for_anom = float("nan")
        mean_cos_anom_neigh = float("nan")

    return {
        "dataset": dataset,
        "num_nodes": n,
        "num_edges_undirected_storage": int(edge_index.size(1)),
        "num_anomalies": n_anom,
        "anomalies_with_neighbors": n_anom_with_neigh,
        "anomalies_isolated": n_anom_isolated,
        "mean_degree_all_nodes": mean_deg_all,
        "mean_degree_anomaly_nodes": mean_deg_anom,
        "mean_degree_first_neighbors_of_anomalies": mean_neigh_deg_for_anom,
        "mean_cosine_anomaly_vs_first_neighbors": mean_cos_anom_neigh,
    }


def _pick_device(cuda_visible: Optional[str]) -> "torch.device":
    import torch

    if cuda_visible is not None and cuda_visible != "":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def main_single():
    import torch

    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True, choices=DATASETS)
    p.add_argument("--gpu", type=str, default=None, help="Physical GPU id written to CUDA_VISIBLE_DEVICES")
    args = p.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    dev = _pick_device(os.environ.get("CUDA_VISIBLE_DEVICES"))
    out = analyze_dataset(args.dataset, dev)
    # Single-line JSON for the --parallel parent to parse (stats must not go to stderr).
    print(json.dumps(out, ensure_ascii=False, separators=(",", ":")))


def main_parallel():
    import torch

    p = argparse.ArgumentParser(description="Multi-GPU parallel graph stats for five datasets")
    p.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated GPU ids, e.g. 0,1,2,3,4; default 0..min(4,n-1)",
    )
    args = p.parse_args()
    exe = sys.executable
    script = os.path.abspath(__file__)
    n_gpu = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    if args.gpus:
        gpu_list = [g.strip() for g in args.gpus.split(",") if g.strip()]
    else:
        if n_gpu <= 0:
            gpu_list = [""] * len(DATASETS)
        else:
            # One physical GPU per dataset when possible; round-robin if fewer GPUs than datasets.
            gpu_list = [str(i % n_gpu) for i in range(len(DATASETS))]

    rows: List[Dict[str, Any]] = []
    procs: List[Tuple[subprocess.Popen, str]] = []
    env_base = os.environ.copy()
    for i, ds in enumerate(DATASETS):
        g = gpu_list[i] if i < len(gpu_list) else gpu_list[-1]
        env = env_base.copy()
        if g != "":
            env["CUDA_VISIBLE_DEVICES"] = g
        elif "CUDA_VISIBLE_DEVICES" in env:
            del env["CUDA_VISIBLE_DEVICES"]
        cmd = [exe, script, "--dataset", ds]
        if g != "":
            cmd.extend(["--gpu", g])
        procs.append((subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env), ds))

    for proc, ds in procs:
        out_b, err_b = proc.communicate()
        if proc.returncode != 0:
            sys.stderr.write(err_b.decode("utf-8", errors="replace"))
            raise RuntimeError(f"Child failed dataset={ds} rc={proc.returncode}")
        text = out_b.decode("utf-8", errors="replace").strip()
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        parsed = False
        for line in reversed(lines):
            if line.startswith("{") and line.endswith("}"):
                rows.append(json.loads(line))
                parsed = True
                break
        if not parsed:
            raise RuntimeError(f"Could not parse output for {ds}: {text[:500]}")

    keys = [
        "dataset",
        "num_nodes",
        "num_anomalies",
        "mean_degree_anomaly_nodes",
        "mean_degree_all_nodes",
        "mean_degree_first_neighbors_of_anomalies",
        "mean_cosine_anomaly_vs_first_neighbors",
        "anomalies_isolated",
    ]
    print("\n" + "=" * 100)
    print("FMGAD / PyGOD five-dataset graph stats (degree = undirected sum-deg, same as utils.compute_node_degree_tensor)")
    print("=" * 100)
    hdr = (
        f"{'dataset':10} {'|N|':>8} {'|Y|':>6} {'deg(anom)':>12} {'deg(all)':>12} "
        f"{'deg(N1(anom))':>14} {'cos(anom,N1)':>14} {'iso_anom':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['dataset']:10} {r['num_nodes']:8d} {r['num_anomalies']:6d} "
            f"{r['mean_degree_anomaly_nodes']:12.4f} {r['mean_degree_all_nodes']:12.4f} "
            f"{r['mean_degree_first_neighbors_of_anomalies']:14.4f} {r['mean_cosine_anomaly_vs_first_neighbors']:14.4f} "
            f"{r['anomalies_isolated']:8d}"
        )
    print("=" * 100)
    # Also print full JSON
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--parallel":
        sys.argv.pop(1)
        main_parallel()
    else:
        main_single()
