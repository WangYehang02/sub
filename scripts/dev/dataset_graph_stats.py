#!/usr/bin/env python3
"""
五个 PyGOD 基准图（与 FMGAD 一致）的图结构统计：
- 异常节点平均度数
- 全图节点平均度数
- 异常节点的一阶邻居的平均度数（先对每个异常求其邻居度数的均值，再对异常取平均）
- 异常节点与其一阶邻居的平均余弦相似度（先对每个异常求与邻居的余弦均值，再对异常取平均）

支持多进程 + 每进程独占一块 GPU（CUDA_VISIBLE_DEVICES），用于并行跑五个数据集。
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
    # 无向一阶邻居：每条 (u,v) 上 u 把 v 当邻居、v 把 u 当邻居
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
    p.add_argument("--gpu", type=str, default=None, help="物理 GPU id，写入 CUDA_VISIBLE_DEVICES")
    args = p.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    dev = _pick_device(os.environ.get("CUDA_VISIBLE_DEVICES"))
    out = analyze_dataset(args.dataset, dev)
    # 单行 JSON，便于 --parallel 父进程解析（统计信息勿走 stderr）
    print(json.dumps(out, ensure_ascii=False, separators=(",", ":")))


def main_parallel():
    import torch

    p = argparse.ArgumentParser(description="多卡并行跑五个数据集图统计")
    p.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="逗号分隔的 GPU 编号，如 0,1,2,3,4；默认使用 0..min(4,n-1)",
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
            # 五数据集各绑一块物理卡；卡少则轮询，避免多进程抢同一张卡
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
            raise RuntimeError(f"子进程失败 dataset={ds} rc={proc.returncode}")
        text = out_b.decode("utf-8", errors="replace").strip()
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        parsed = False
        for line in reversed(lines):
            if line.startswith("{") and line.endswith("}"):
                rows.append(json.loads(line))
                parsed = True
                break
        if not parsed:
            raise RuntimeError(f"无法解析 {ds} 的输出: {text[:500]}")

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
    print("FMGAD / PyGOD 五数据集图统计（度数=无向合一度，与 utils.compute_node_degree_tensor 一致）")
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
    # 同时打印完整 JSON
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--parallel":
        sys.argv.pop(1)
        main_parallel()
    else:
        main_single()
