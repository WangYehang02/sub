import torch


def compute_residuals(h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """
    AnomalyGFM 风格残差：
      r_i = h_i - mean_{j in N(i)} h_j

    约定 edge_index 为 PyG 格式 [2, E]，消息从 src=edge_index[0] 聚合到 dst=edge_index[1]。
    """
    if h.dim() != 2:
        raise ValueError(f"h should be [N, D], got {tuple(h.shape)}")
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(f"edge_index should be [2, E], got {tuple(edge_index.shape)}")

    src, dst = edge_index[0], edge_index[1]
    n, d = h.size(0), h.size(1)

    # sum_{j->i} h_j
    neigh_sum = torch.zeros((n, d), device=h.device, dtype=h.dtype)
    neigh_sum.index_add_(0, dst, h[src])

    deg = torch.zeros((n,), device=h.device, dtype=h.dtype)
    deg.index_add_(0, dst, torch.ones_like(dst, dtype=h.dtype))
    deg = deg.clamp_min(1.0).unsqueeze(1)  # 避免除0；孤立点视为均值=0

    neigh_mean = neigh_sum / deg
    return h - neigh_mean


def compute_dual_residuals_with_degree(h: torch.Tensor, edge_index: torch.Tensor):
    """
    计算双重残差（全局+局部）并返回节点度数用于自适应门控。
    全局残差：节点与全图均值的差异，适合稀疏/低度节点（稳健）。
    局部残差：节点与邻居均值的差异，适合稠密图上的局部结构异常（敏感）。
    约定 edge_index 为 PyG 格式 [2, E]，消息从 src=edge_index[0] 聚合到 dst=edge_index[1]。

    返回:
        r_global: [N, D] 全局统计残差
        r_local:  [N, D] 局部结构残差
        deg:      [N, 1] 节点度数（用于后续门控）
    """
    if h.dim() != 2:
        raise ValueError(f"h should be [N, D], got {tuple(h.shape)}")
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(f"edge_index should be [2, E], got {tuple(edge_index.shape)}")

    # 1. 全局残差 (Global Residual)：节点相对全图均值的偏差
    global_mean = torch.mean(h, dim=0, keepdim=True)
    r_global = h - global_mean

    # 2. 局部残差 (Local Residual) 与度数
    src, dst = edge_index[0], edge_index[1]
    n, d = h.size(0), h.size(1)
    neigh_sum = torch.zeros((n, d), device=h.device, dtype=h.dtype)
    neigh_sum.index_add_(0, dst, h[src])

    deg_val = torch.zeros((n,), device=h.device, dtype=h.dtype)
    deg_val.index_add_(0, dst, torch.ones_like(dst, dtype=h.dtype))

    deg_clamped = deg_val.clamp_min(1.0).unsqueeze(1)
    neigh_mean = neigh_sum / deg_clamped
    r_local = h - neigh_mean

    return r_global, r_local, deg_val.unsqueeze(1)
