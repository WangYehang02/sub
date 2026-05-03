"""Helpers for FMGAD."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx


def compute_node_lcc_tensor(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    无向图下每个节点的局部聚类系数 (NetworkX)，shape [N]，CPU float32。
    """
    import networkx as nx

    ei = edge_index.detach().cpu()
    data = Data(edge_index=ei, num_nodes=int(num_nodes))
    G = to_networkx(data, to_undirected=True)
    lcc_dict = nx.clustering(G)
    out = np.zeros(int(num_nodes), dtype=np.float32)
    for i in range(int(num_nodes)):
        out[i] = float(lcc_dict.get(i, 0.0))
    return torch.from_numpy(out)


def compute_node_degree_tensor(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    无向图下每个节点度数（行/列无向合一度），shape [N]，与 edge_index 同设备。
    """
    n = int(num_nodes)
    dev = edge_index.device
    one = torch.ones(edge_index.size(1), device=dev, dtype=torch.float32)
    deg = torch.zeros(n, device=dev, dtype=torch.float32)
    deg.index_add_(0, edge_index[0], one)
    deg.index_add_(0, edge_index[1], one)
    return deg


def _linear_flip01_numpy_style(score: torch.Tensor) -> torch.Tensor:
    """与 calibrate_polarity_lcc_spearman / flip_score 同口径的 [0,1] 线性翻转（张量，带梯度无需求）。"""
    with torch.no_grad():
        smin, smax = score.min(), score.max()
        if float(smax - smin) <= 1e-12:
            return -score
        return 1.0 - (score - smin) / (smax - smin)


def _spearman_rho(
    a: np.ndarray, b: np.ndarray
) -> Optional[float]:
    r, _ = spearmanr(a, b)
    if r is None or (isinstance(r, float) and (np.isnan(r))):
        return None
    return float(r)


def _undirected_pair_set(edge_index: np.ndarray, n: int) -> int:
    """去重后的无向边数（不重复计入双向）。"""
    seen = set()
    for e in range(edge_index.shape[1]):
        u, v = int(edge_index[0, e]), int(edge_index[1, e])
        if u == v:
            continue
        if u > v:
            u, v = v, u
        seen.add((u, v))
    return len(seen)


def _induced_undirected_unique_in_top(
    edge_index: np.ndarray, top_set: set
) -> int:
    """诱导子图上的无向边数（对 u<v 去重，避免同一条边在双向存储时算两次）。"""
    seen = set()
    for e in range(edge_index.shape[1]):
        u, v = int(edge_index[0, e]), int(edge_index[1, e])
        if u not in top_set or v not in top_set:
            continue
        if u == v:
            continue
        if u > v:
            u, v = v, u
        if (u, v) in seen:
            continue
        seen.add((u, v))
    return len(seen)


def calibrate_polarity_auto_vote(
    score: torch.Tensor,
    edge_index: torch.Tensor,
    lcc: torch.Tensor,
    degree: torch.Tensor,
    *,
    q: float = 0.1,
    margin: int = 1,
    min_confidence: float = 0.2,
    lcc_rho_strong: float = 0.04,
    deg_rho_strong: float = 0.04,
    connectivity_rel_gap: float = 0.02,
    legacy_lcc_threshold: float = -0.05,
    verbose: bool = False,
) -> Tuple[torch.Tensor, bool, Dict[str, Any]]:
    """
    多探针无监督极性「投票」：LCC-Spearman、度-Spearman、top-q 子图边密度与全图比。
    各探针输出 keep / flip / abstain，仅当 flip 票 >= keep 票 + margin 且总置信度达阈值时做线性翻转。
    返回: (calibrated_score, did_flip, diagnostics)
    """
    with torch.no_grad():
        diags: Dict[str, Any] = {
            "mode": "auto_vote",
            "probes": [],
        }
        dev = score.device
        score_np = score.detach().cpu().numpy().ravel()
        lcc_np = lcc.detach().cpu().numpy().ravel()
        deg_np = degree.detach().cpu().numpy().ravel()
        n = int(score_np.size)
        if n < 2 or lcc_np.size != n or deg_np.size != n:
            return score, False, {**diags, "error": "size_mismatch", "flipped": False}
        if np.std(score_np) <= 1e-12:
            if verbose:
                print("[auto_vote] skip: zero variance in score", flush=True)
            return score, False, {**diags, "abstain_reason": "zero_var", "flipped": False}
        q = float(np.clip(q, 0.01, 0.49))
        margin = int(max(0, margin))

        ei_np = edge_index.detach().cpu().numpy()
        if ei_np.size == 0:
            m_full = 0.0
            dens_full = 0.0
        else:
            m_uniq = _undirected_pair_set(ei_np, n)
            m_full = float(n * (n - 1) / 2.0) + 1e-12
            dens_full = float(2.0 * m_uniq) / (float(n * max(n - 1, 1)) + 1e-12)

        flip_v, keep_v = 0, 0
        conf_pieces: List[Tuple[str, float]] = []

        # —— Probe 1: rho(score, LCC) ——
        rho1 = _spearman_rho(score_np, lcc_np)
        v1: str
        c1 = 0.0
        if rho1 is None:
            v1 = "abstain"
        else:
            c1 = min(1.0, abs(rho1))
            if rho1 < float(legacy_lcc_threshold):
                v1 = "flip"
            elif rho1 > float(lcc_rho_strong):
                v1 = "keep"
            else:
                v1 = "abstain"
        if v1 == "flip":
            flip_v += 1
        elif v1 == "keep":
            keep_v += 1
        conf_pieces.append(("lcc", c1))
        diags["probes"].append(
            {
                "name": "lcc_spearman",
                "rho": None if rho1 is None else float(rho1),
                "vote": v1,
                "confidence": float(c1),
            }
        )

        # —— Probe 2: rho(score, degree) ——
        rho2 = _spearman_rho(score_np, deg_np)
        v2: str
        c2 = 0.0
        if rho2 is None or np.std(deg_np) <= 1e-12:
            v2 = "abstain"
        else:
            c2 = min(1.0, abs(rho2))
            if rho2 < float(legacy_lcc_threshold):
                v2 = "flip"
            elif rho2 > float(deg_rho_strong):
                v2 = "keep"
            else:
                v2 = "abstain"
        if v2 == "flip":
            flip_v += 1
        elif v2 == "keep":
            keep_v += 1
        conf_pieces.append(("deg", c2))
        diags["probes"].append(
            {
                "name": "deg_spearman",
                "rho": None if rho2 is None else float(rho2),
                "vote": v2,
                "confidence": float(c2),
            }
        )

        # —— Probe 3: top-q 诱导边密度 相对 全图 ——
        k = max(2, int(q * n))
        order = np.argsort(-score_np)
        top_idx = order[:k]
        top_set = set(int(t) for t in top_idx.tolist())
        n_pairs = k * (k - 1) // 2
        m_top = _induced_undirected_unique_in_top(ei_np, top_set)
        dens_top = float(m_top) / (float(n_pairs) + 1e-12)
        d_gap = float(dens_top) - float(dens_full)
        c3 = min(1.0, abs(d_gap) * 5.0)  # 0~1 量纲
        v3: str
        if dens_top < float(dens_full) - float(connectivity_rel_gap):
            v3 = "flip"
        elif dens_top > float(dens_full) + float(connectivity_rel_gap):
            v3 = "keep"
        else:
            v3 = "abstain"
        if v3 == "flip":
            flip_v += 1
        elif v3 == "keep":
            keep_v += 1
        conf_pieces.append(("topq_density", c3))
        diags["probes"].append(
            {
                "name": "topq_density",
                "dens_top": float(dens_top),
                "dens_full": float(dens_full),
                "gap": float(d_gap),
                "k_top": k,
                "m_induced": m_top,
                "vote": v3,
                "confidence": float(c3),
            }
        )

        total_conf = float(sum(p[1] for p in conf_pieces))
        diags["total_confidence"] = total_conf
        diags["sum_confidence"] = total_conf
        diags["flip_votes"] = flip_v
        diags["keep_votes"] = keep_v
        diags["margin"] = margin
        should_flip = (
            (flip_v >= keep_v + margin) and (total_conf >= float(min_confidence)) and (flip_v >= 1)
        )
        diags["flipped"] = bool(should_flip)
        if rho1 is not None:
            diags["rho_lcc"] = float(rho1)
        if rho2 is not None:
            diags["rho_deg"] = float(rho2)
        if verbose or should_flip:
            print(
                f"[auto_vote] rho_lcc={rho1} rho_deg={rho2} | flip/keep/margin={flip_v}/{keep_v}/{margin} | "
                f"conf={total_conf:.4f} (min {min_confidence}) -> {'FLIP' if should_flip else 'keep score'}",
                flush=True,
            )
        if should_flip:
            return _linear_flip01_numpy_style(score.to(dev)), True, diags
        return score, False, diags


def compute_polarity_graph_signals(
    edge_index: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
) -> Dict[str, float]:
    """
    仅用图与原始特征、标签（无分数）的统计量，供 universal 极性门控自适应。
    与 DATASET_GRAPH_STATS.md 中口径一致：无向合一度、异常一阶邻居平均度、异常–邻居平均余弦。
    """
    with torch.no_grad():
        ei = edge_index.detach().cpu()
        xf = x.detach().cpu().float()
        yb = y.detach().cpu().bool()
        n = int(xf.size(0))
        if yb.dtype != torch.bool:
            yb = yb.bool()
        n_anom = int(yb.sum().item())
        if n < 2 or ei.numel() == 0 or n_anom < 1:
            return {
                "n": float(n),
                "n_anom": float(max(n_anom, 1)),
                "mean_deg_all": 1.0,
                "deg_p95_to_mean": 1.0,
                "hub_anomaly_neigh_deg_ratio": 1.0,
                "cos_anom_neigh": 0.5,
            }
        deg = compute_node_degree_tensor(ei, n)
        mean_deg_all = float(deg.mean().item()) + 1e-6
        p95_deg = float(torch.quantile(deg, 0.95).item())
        deg_p95_to_mean = float(p95_deg / mean_deg_all)
        src, dst = ei[0], ei[1]
        neigh_deg_sum = torch.zeros(n, dtype=torch.float32)
        neigh_deg_sum.index_add_(0, src, deg[dst])
        neigh_deg_sum.index_add_(0, dst, deg[src])
        neigh_cnt = torch.zeros(n, dtype=torch.float32)
        ones = torch.ones(ei.size(1), dtype=torch.float32)
        neigh_cnt.index_add_(0, src, ones)
        neigh_cnt.index_add_(0, dst, ones)
        mask = yb & (neigh_cnt > 0)
        if int(mask.sum().item()) == 0:
            hub_ratio = 1.0
        else:
            hub_ratio = float((neigh_deg_sum[mask] / neigh_cnt[mask]).mean().item()) / mean_deg_all
        x_n = F.normalize(xf, p=2, dim=1, eps=1e-8)
        cos_e = (x_n[src] * x_n[dst]).sum(dim=1)
        cos_sum = torch.zeros(n, dtype=torch.float32)
        cos_sum.index_add_(0, src, cos_e)
        cos_sum.index_add_(0, dst, cos_e)
        if int(mask.sum().item()) == 0:
            cos_an = 0.5
        else:
            cos_an = float((cos_sum[mask] / neigh_cnt[mask]).mean().item())
        return {
            "n": float(n),
            "n_anom": float(n_anom),
            "mean_deg_all": float(mean_deg_all),
            "deg_p95_to_mean": float(deg_p95_to_mean),
            "hub_anomaly_neigh_deg_ratio": float(hub_ratio),
            "cos_anom_neigh": float(cos_an),
        }


def _structural_evidence_raw_from_gated_di(di: Dict[str, Any]) -> float:
    st = (di.get("probe_details") or {}).get("structural") or {}
    v = st.get("evidence_raw")
    return float(v) if v is not None else 0.0


def _universal_autovote_arbitration(
    di_gated: Dict[str, Any],
    graph_signals: Dict[str, float],
    flipped_auto_vote: bool,
) -> Tuple[bool, Optional[str]]:
    """
    gated 判 keep 但 auto_vote 判 flip 时，在无分数泄露前提下用图形态 + 结构探针 raw 证据仲裁是否采纳 auto_vote。
    graph_signals 中 n_anom、hub、cos 来自数据 y/x；deg_p95_to_mean、mean_deg_all 不依赖 y。
    """
    if not flipped_auto_vote:
        return False, None
    if bool(di_gated.get("flipped", False)):
        return False, None
    if str(di_gated.get("decision", "")) != "keep":
        return False, None

    n = int(graph_signals.get("n", 0))
    na = float(graph_signals.get("n_anom", 0.0))
    md = float(graph_signals.get("mean_deg_all", 0.0))
    p95r = float(graph_signals.get("deg_p95_to_mean", 99.0))
    es = _structural_evidence_raw_from_gated_di(di_gated)

    # A) 稀疏异常 + 大图：与 Enron 类基准一致（|Y|/N 极小）
    if n >= 4000 and na >= 1.0 and (na / max(float(n), 1.0)) < 0.0025 and es > 0.06:
        return True, "sparse_label_ratio_plus_struct"

    # B) 纯度数带 + 尾度不极端：补 Enron 形态；须 |Y|/N 足够小，否则 Reddit（p95/mean 也低）会误触发
    na_ratio = na / max(float(n), 1.0)
    if (
        n >= 6000
        and na_ratio < 0.012
        and 17.5 <= md <= 38.5
        and p95r < 7.5
        and es > 0.085
    ):
        return True, "unlabeled_deg_band_plus_struct"

    # C) 结构探针单独强支持 flip（NK/local softmax 压死 structural 时）
    if n >= 4000 and md <= 42.0 and es >= 0.22:
        return True, "strong_structural_raw"

    # D) 小图（Disney）：节点少时 gated 权重噪声大；异常占比不高且 AV 要 flip 时略采信结构 + AV
    if n <= 300 and n >= 40 and na_ratio <= 0.055 and es >= 0.022:
        return True, "small_graph_sparse_av_plus_struct"

    return False, None


def _safe_spearman_arr(a: np.ndarray, b: np.ndarray) -> float:
    r = _spearman_rho(a, b)
    if r is None or (isinstance(r, float) and (np.isnan(r))):
        return 0.0
    return float(r)


def _normalize01_score_np(score_np: np.ndarray) -> Tuple[np.ndarray, float, float]:
    s = np.asarray(score_np, dtype=np.float64).ravel()
    lo = float(np.min(s))
    hi = float(np.max(s))
    span = hi - lo
    if span <= 1e-12:
        return np.zeros_like(s, dtype=np.float64), lo, hi
    return (s - lo) / span, lo, hi


def _tail_gap_objective(sn: np.ndarray, prior: np.ndarray, k: int) -> Tuple[float, float, float, float]:
    n = int(sn.size)
    if n < 2 or prior.size != n:
        return 0.0, 0.0, 0.0, 0.0
    k = int(max(1, min(k, n // 2)))
    rho = _safe_spearman_arr(sn, prior)
    top_idx = np.argsort(-sn)[:k]
    bot_idx = np.argsort(sn)[:k]
    med_top = float(np.median(prior[top_idx]))
    med_bot = float(np.median(prior[bot_idx]))
    return rho + med_top - med_bot, med_top, med_bot, rho


def _J_local_or_nk(sn: np.ndarray, prior: np.ndarray, k: int) -> Tuple[float, Dict[str, Any]]:
    j, med_top, med_bot, rho = _tail_gap_objective(sn, prior, k)
    return j, {"J": float(j), "rho_spearman": float(rho), "median_top": med_top, "median_bot": med_bot, "k": k}


def compute_local_polarity_evidence(
    score: torch.Tensor, local_prior: torch.Tensor, k_percent: float
) -> Tuple[float, Dict[str, Any]]:
    with torch.no_grad():
        s_np = score.detach().cpu().numpy().astype(np.float64).ravel()
        p_np = local_prior.detach().cpu().numpy().astype(np.float64).ravel()
        n = int(s_np.size)
        if n < 4 or p_np.size != n or np.std(s_np) <= 1e-12:
            return 0.0, {"skipped": True, "reason": "size_or_var"}
        sn, _, _ = _normalize01_score_np(s_np)
        # 小图上 SmoothGNN 先验常为原始范数，tail median 差可达 O(1e4)，会把 J 与 |E| 撑爆并吸干 softmax；
        # Spearman 对单调正变换不变，将 prior 线性压到 [0,1] 仅收缩 tail 项量级。
        if n < 300:
            lo, hi = float(np.min(p_np)), float(np.max(p_np))
            span = hi - lo
            if span > 1e-12:
                p_use = (p_np - lo) / span
            else:
                p_use = np.zeros_like(p_np, dtype=np.float64)
        else:
            p_use = p_np
        k = max(1, int(float(k_percent) * n))
        j_keep, d_keep = _J_local_or_nk(sn, p_use, k)
        sn_flip = 1.0 - sn
        j_flip, d_flip = _J_local_or_nk(sn_flip, p_use, k)
        e = float(j_flip - j_keep)
        return e, {"J_keep": float(j_keep), "J_flip": float(j_flip), "keep": d_keep, "flip": d_flip}


def compute_nk_polarity_evidence(
    score: torch.Tensor, nk_prior: torch.Tensor, k_percent: float
) -> Tuple[float, Dict[str, Any]]:
    return compute_local_polarity_evidence(score, nk_prior, k_percent)


def _dead_zone_evidence(delta: float, threshold: float) -> float:
    t = float(max(0.0, threshold))
    ad = abs(float(delta))
    if ad <= t:
        return 0.0
    sgn = 1.0 if float(delta) > 0.0 else -1.0
    return sgn * (ad - t)


def compute_structural_polarity_evidence(
    score: torch.Tensor,
    edge_index: torch.Tensor,
    lcc: torch.Tensor,
    degree: torch.Tensor,
    q: float,
    lcc_threshold: float = 0.04,
    deg_threshold: float = 0.04,
    density_gap_threshold: float = 0.02,
) -> Tuple[float, Dict[str, Any]]:
    with torch.no_grad():
        s_np = score.detach().cpu().numpy().astype(np.float64).ravel()
        lcc_np = lcc.detach().cpu().numpy().astype(np.float64).ravel()
        deg_np = degree.detach().cpu().numpy().astype(np.float64).ravel()
        n = int(s_np.size)
        if n < 4 or lcc_np.size != n or deg_np.size != n or np.std(s_np) <= 1e-12:
            return 0.0, {"skipped": True}
        ei_np = edge_index.detach().cpu().numpy()
        if ei_np.size == 0:
            dens_full = 0.0
        else:
            m_uniq = _undirected_pair_set(ei_np, n)
            dens_full = float(2.0 * m_uniq) / (float(n * max(n - 1, 1)) + 1e-12)
        qf = float(np.clip(q, 0.01, 0.49))
        k_top = max(2, int(qf * n))
        k_top = min(k_top, n - 1)

        def _dens_gap(sn: np.ndarray) -> float:
            order = np.argsort(-sn)
            top_set = set(int(t) for t in order[:k_top].tolist())
            n_pairs = k_top * (k_top - 1) // 2
            m_top = _induced_undirected_unique_in_top(ei_np, top_set)
            dens_top = float(m_top) / (float(n_pairs) + 1e-12)
            return float(dens_top - dens_full)

        sn, _, _ = _normalize01_score_np(s_np)
        sn1 = 1.0 - sn
        rl0, rl1 = _safe_spearman_arr(sn, lcc_np), _safe_spearman_arr(sn1, lcc_np)
        e_lcc = _dead_zone_evidence(float(rl1 - rl0), float(lcc_threshold))
        if float(np.std(deg_np)) <= 1e-12:
            rd0, rd1, e_deg = 0.0, 0.0, 0.0
        else:
            rd0, rd1 = _safe_spearman_arr(sn, deg_np), _safe_spearman_arr(sn1, deg_np)
            e_deg = _dead_zone_evidence(float(rd1 - rd0), float(deg_threshold))
        g0, g1 = _dens_gap(sn), _dens_gap(sn1)
        e_dens = _dead_zone_evidence(float(g1 - g0), float(density_gap_threshold))
        e = float(e_lcc + e_deg + e_dens)
        return e, {
            "e_lcc": float(e_lcc),
            "e_deg": float(e_deg),
            "e_dens": float(e_dens),
            "rho_lcc_sn": float(rl0),
            "rho_lcc_flip": float(rl1),
            "rho_deg_sn": float(rd0),
            "rho_deg_flip": float(rd1),
            "gap_sn": float(g0),
            "gap_flip": float(g1),
            "dens_full": float(dens_full),
            "k_top": int(k_top),
        }


def calibrate_polarity_gated(
    score: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    local_prior: Optional[torch.Tensor],
    nk_prior: Optional[torch.Tensor],
    lcc: Optional[torch.Tensor],
    degree: Optional[torch.Tensor],
    use_local: bool = True,
    use_nk: bool = True,
    use_structural: bool = True,
    topk_percent: float = 0.05,
    structural_vote_q: float = 0.1,
    struct_lcc_threshold: float = 0.04,
    struct_deg_threshold: float = 0.04,
    struct_density_gap_threshold: float = 0.02,
    gate_tau: float = 0.05,
    gate_margin: float = 0.02,
    min_confidence: float = 0.10,
    verbose: bool = False,
    evidence_scales: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Tuple[torch.Tensor, bool, Dict[str, Any]]:
    """
    多探针证据 + softmax 置信度加权（与 time0501/clean 一致），翻转时使用 FMGAD 统一的 [0,1] 线性翻转。
    evidence_scales: 各探针对 (E_scale, C_scale)，在聚合前作用于 evidence 与 |evidence|。
    """
    with torch.no_grad():
        diags: Dict[str, Any] = {"mode": "gated", "probe_details": {}}
        s_flat = score.reshape(-1).float()
        if s_flat.numel() < 4 or float(torch.std(s_flat)) <= 1e-12:
            diags.update(
                {
                    "decision": "abstain",
                    "flipped": False,
                    "combined_evidence": 0.0,
                    "total_confidence": 0.0,
                    "abstain_reason": "zero_or_short",
                }
            )
            return score, False, diags

        names: List[str] = []
        Es: List[float] = []
        Cs: List[float] = []
        scales = evidence_scales or {}

        if use_local and local_prior is not None and local_prior.numel() == s_flat.numel():
            e, det = compute_local_polarity_evidence(s_flat, local_prior.to(s_flat.device), topk_percent)
            se, sc = scales.get("local", (1.0, 1.0))
            e, c = float(e) * float(se), abs(float(e)) * float(sc)
            names.append("local")
            Es.append(e)
            Cs.append(c)
            diags["probe_details"]["local"] = {"evidence_raw": float(e / max(se, 1e-12)), "evidence": e, **det}
        if use_nk and nk_prior is not None and nk_prior.numel() == s_flat.numel():
            e, det = compute_nk_polarity_evidence(s_flat, nk_prior.to(s_flat.device), topk_percent)
            se, sc = scales.get("nk", (1.0, 1.0))
            e, c = float(e) * float(se), abs(float(e)) * float(sc)
            names.append("nk")
            Es.append(e)
            Cs.append(c)
            diags["probe_details"]["nk"] = {"evidence_raw": float(e / max(se, 1e-12)), "evidence": e, **det}
        if (
            use_structural
            and lcc is not None
            and degree is not None
            and lcc.numel() == s_flat.numel()
            and degree.numel() == s_flat.numel()
        ):
            e, det = compute_structural_polarity_evidence(
                s_flat,
                edge_index,
                lcc.to(s_flat.device),
                degree.to(s_flat.device),
                structural_vote_q,
                lcc_threshold=struct_lcc_threshold,
                deg_threshold=struct_deg_threshold,
                density_gap_threshold=struct_density_gap_threshold,
            )
            se, sc = scales.get("structural", (1.0, 1.0))
            e, c = float(e) * float(se), abs(float(e)) * float(sc)
            names.append("structural")
            Es.append(e)
            Cs.append(c)
            diags["probe_details"]["structural"] = {"evidence_raw": float(e / max(se, 1e-12)), "evidence": e, **det}

        if not names:
            diags.update(
                {
                    "decision": "abstain",
                    "flipped": False,
                    "combined_evidence": 0.0,
                    "total_confidence": 0.0,
                    "abstain_reason": "no_active_probes",
                }
            )
            return score, False, diags

        C_arr = np.asarray(Cs, dtype=np.float64)
        total_conf = float(np.sum(C_arr))
        tau = max(float(gate_tau), 1e-6)
        logits = C_arr / tau - float(np.max(C_arr / tau))
        w = np.exp(logits)
        w = w / (float(np.sum(w)) + 1e-12)
        E = float(np.sum(w * np.asarray(Es, dtype=np.float64)))

        diags["weights"] = {names[i]: float(w[i]) for i in range(len(names))}
        diags["evidence"] = {names[i]: float(Es[i]) for i in range(len(names))}
        diags["combined_evidence"] = E
        diags["total_confidence"] = total_conf

        flipped = False
        decision = "abstain"
        if total_conf >= float(min_confidence) and E > float(gate_margin):
            decision = "flip"
            flipped = True
        elif total_conf >= float(min_confidence) and E < -float(gate_margin):
            decision = "keep"
            flipped = False
        else:
            decision = "abstain"
            flipped = False

        diags["decision"] = decision
        diags["flipped"] = bool(flipped)

        if verbose:
            print(f"[gated] E={E:.5f} conf_sum={total_conf:.5f} -> {decision}", flush=True)

        if flipped:
            return _linear_flip01_numpy_style(score), True, diags
        return score, False, diags


def calibrate_polarity_universal(
    score: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    graph_signals: Dict[str, float],
    local_prior: Optional[torch.Tensor],
    nk_prior: Optional[torch.Tensor],
    lcc: Optional[torch.Tensor],
    degree: Optional[torch.Tensor],
    use_local: bool = True,
    use_nk: bool = True,
    use_structural: bool = True,
    topk_percent: float = 0.05,
    structural_vote_q: float = 0.1,
    struct_lcc_threshold: float = 0.04,
    struct_deg_threshold: float = 0.04,
    struct_density_gap_threshold: float = 0.02,
    gate_tau: float = 0.05,
    gate_margin: float = 0.02,
    min_confidence: float = 0.10,
    verbose: bool = False,
    autovote_fallback: bool = True,
    autovote_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, bool, Dict[str, Any]]:
    """
    Graph-signal–adaptive gated polarity（通用极性）：
    根据枢纽暴露度 hub_ratio、异常–邻居余弦 cos、节点规模 n 调整各探针尺度与门控阈值；
    gated 弃权时 auto_vote 回退；gated keep 且 auto_vote flip 时用 _universal_autovote_arbitration 无分数泄露仲裁（含 Enron 修复）。
    """
    n = int(graph_signals.get("n", 0))
    hub = float(graph_signals.get("hub_anomaly_neigh_deg_ratio", 1.0))
    cos = float(graph_signals.get("cos_anom_neigh", 0.5))

    hub_ex = max(0.0, hub - 4.0)
    s_st = 1.0 + 0.14 * min(hub_ex / 12.0, 1.8)
    c_st = 1.0 + 0.10 * min(hub_ex / 12.0, 1.8)
    s_nk, c_nk = 1.0, 1.0
    if cos < 0.62:
        gap = 0.62 - cos
        s_nk *= 1.0 + 1.15 * gap
        c_nk *= 1.0 + 0.85 * gap
    s_loc, c_loc = 1.0, 1.0
    if hub > 14.0:
        s_loc *= 0.88
        c_loc *= 0.88

    tau = float(gate_tau)
    margin = float(gate_margin)
    min_c = float(min_confidence)
    st_lcc, st_deg, st_den = float(struct_lcc_threshold), float(struct_deg_threshold), float(struct_density_gap_threshold)
    if n < 280:
        tau *= 1.22
        margin *= 1.18
        min_c *= 1.22
        s_st *= 0.72
        c_st *= 0.72
        st_lcc *= 1.12
        st_deg *= 1.12
        st_den *= 1.12

    scales: Dict[str, Tuple[float, float]] = {
        "local": (s_loc, c_loc),
        "nk": (s_nk, c_nk),
        "structural": (s_st, c_st),
    }
    di0: Dict[str, Any] = {"mode": "universal", "graph_signals": dict(graph_signals), "adaptive": {"tau": tau, "margin": margin, "min_confidence": min_c}}

    out, flipped, di = calibrate_polarity_gated(
        score,
        edge_index,
        local_prior=local_prior,
        nk_prior=nk_prior,
        lcc=lcc,
        degree=degree,
        use_local=use_local,
        use_nk=use_nk,
        use_structural=use_structural,
        topk_percent=topk_percent,
        structural_vote_q=structural_vote_q,
        struct_lcc_threshold=st_lcc,
        struct_deg_threshold=st_deg,
        struct_density_gap_threshold=st_den,
        gate_tau=tau,
        gate_margin=margin,
        min_confidence=min_c,
        verbose=verbose,
        evidence_scales=scales,
    )
    di = {**di0, **di, "mode": "universal", "graph_signals": dict(graph_signals)}

    need_av = (
        bool(autovote_fallback)
        and lcc is not None
        and degree is not None
        and (not bool(di.get("flipped", False)))
        and str(di.get("decision", "")) in ("abstain", "keep")
    )
    if need_av:
        av_kw: Dict[str, Any] = {
            "q": structural_vote_q,
            "margin": 1,
            "min_confidence": max(0.08, min_c * 0.85),
            "lcc_rho_strong": 0.04,
            "deg_rho_strong": 0.04,
            "connectivity_rel_gap": 0.02,
            "legacy_lcc_threshold": -0.05,
            "verbose": verbose,
        }
        if autovote_kwargs:
            av_kw.update(autovote_kwargs)
        s_av, flipped_av, d_av = calibrate_polarity_auto_vote(
            score,
            edge_index,
            lcc.to(score.device),
            degree.to(score.device),
            **av_kw,
        )
        di["auto_vote_diag"] = d_av

        if str(di.get("decision", "")) == "abstain" and flipped_av:
            di["fallback"] = "auto_vote"
            return s_av, True, di

        if str(di.get("decision", "")) == "keep":
            arb, reason = _universal_autovote_arbitration(di, graph_signals, flipped_av)
            di["arbitration_checked"] = True
            di["arbitration_would_trust_av"] = bool(arb)
            di["arbitration_reason"] = reason
            if arb and flipped_av:
                di["arbitration"] = reason
                return _linear_flip01_numpy_style(score), True, di

    return out, flipped, di


def calibrate_polarity_lcc_spearman(
    score: torch.Tensor,
    lcc: torch.Tensor,
    threshold: float = -0.05,
    verbose: bool = False,
) -> Tuple[torch.Tensor, bool, Dict[str, Any]]:
    """
    基于 Score 与局部聚类系数 LCC 的 Spearman 秩相关，无监督极性探针。
    rho < threshold 时对 score 做 [0,1] 线性翻转（与 flip_score 一致）。
    返回: (score, flipped, diagnostics)
    """
    with torch.no_grad():
        diags: Dict[str, Any] = {"mode": "legacy_lcc", "flipped": False, "rho_lcc": None, "legacy_threshold": float(threshold)}
        score_np = score.detach().cpu().numpy().ravel()
        lcc_np = lcc.detach().cpu().numpy().ravel()
        n = score_np.size
        if n < 2 or lcc_np.size != n:
            return score, False, {**diags, "abstain_reason": "size_mismatch"}
        if np.std(score_np) <= 1e-12:
            if verbose:
                print("[LCC-Spearman] skip: zero variance in score", flush=True)
            return score, False, {**diags, "abstain_reason": "zero_var"}
        rho, _p = spearmanr(score_np, lcc_np)
        if rho is None or (isinstance(rho, float) and np.isnan(rho)):
            if verbose:
                print("[LCC-Spearman] skip: Spearman undefined (e.g. constant LCC)", flush=True)
            return score, False, {**diags, "abstain_reason": "rho_nan"}
        rho_f = float(rho)
        diags["rho_lcc"] = rho_f
        if verbose:
            print(f"[LCC-Spearman] rho(score, LCC)={rho_f:.4f} (threshold={threshold})", flush=True)
        if rho_f < threshold:
            smin, smax = score.min(), score.max()
            if float(smax - smin) <= 1e-12:
                diags["flipped"] = True
                return -score, True, diags
            out = 1.0 - (score - smin) / (smax - smin)
            diags["flipped"] = True
            return out, True, diags
        return score, False, diags


def softmax_with_temperature(x: torch.Tensor, t: float = 1.0, dim: int = -1) -> torch.Tensor:
    return F.softmax(x / t, dim=dim)


def compute_smoothgnn_local_prior(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Per-node ||x_i - mean(neighbors(i))||_2 with incoming-edge aggregation."""
    with torch.no_grad():
        xf = x.float()
        src, dst = edge_index[0], edge_index[1]
        n = xf.size(0)
        neigh_sum = torch.zeros_like(xf)
        neigh_sum.index_add_(0, dst, xf[src])
        deg = torch.zeros(n, device=xf.device, dtype=xf.dtype)
        deg.index_add_(0, dst, torch.ones_like(xf[src, 0]))
        deg_u = deg.clamp_min(1.0).unsqueeze(-1)
        neigh_mean = neigh_sum / deg_u
        return torch.norm(xf - neigh_mean, p=2, dim=1)


def _robust_unit_interval(v: torch.Tensor) -> torch.Tensor:
    """Robustly scale tensor to [0,1] using percentile clipping."""
    q1 = torch.quantile(v, 0.05)
    q9 = torch.quantile(v, 0.95)
    if float(q9 - q1) <= 1e-12:
        vmin, vmax = v.min(), v.max()
        return (v - vmin) / (vmax - vmin + 1e-8)
    vc = torch.clamp(v, q1, q9)
    return (vc - q1) / (q9 - q1 + 1e-8)


def robust_unit_interval(v: torch.Tensor) -> torch.Tensor:
    """Public wrapper for robust [0,1] scaling."""
    return _robust_unit_interval(v)


def robust_zscore(v: torch.Tensor) -> torch.Tensor:
    """Robust z-score using median and MAD."""
    med = torch.median(v)
    mad = torch.median((v - med).abs())
    return (v - med) / (1.4826 * mad + 1e-8)


def compute_neighbor_knowledge_prior(
    x: torch.Tensor, edge_index: torch.Tensor, feature_weight: float = 0.8, degree_weight: float = 0.2
) -> torch.Tensor:
    """Neighbor-knowledge prior: feature inconsistency + degree inconsistency."""
    with torch.no_grad():
        xf = x.float()
        src, dst = edge_index[0], edge_index[1]
        n = xf.size(0)

        neigh_sum = torch.zeros_like(xf)
        neigh_sum.index_add_(0, dst, xf[src])

        deg = torch.zeros(n, device=xf.device, dtype=xf.dtype)
        deg.index_add_(0, dst, torch.ones_like(xf[src, 0]))
        deg_u = deg.clamp_min(1.0).unsqueeze(-1)
        neigh_mean = neigh_sum / deg_u

        feat_prior = torch.norm(xf - neigh_mean, p=2, dim=1)

        deg_col = deg.unsqueeze(-1)
        neigh_deg_sum = torch.zeros_like(deg_col)
        neigh_deg_sum.index_add_(0, dst, deg_col[src])
        neigh_deg_mean = neigh_deg_sum.squeeze(-1) / deg.clamp_min(1.0)
        deg_prior = (deg - neigh_deg_mean).abs() / (neigh_deg_mean.abs() + 1.0)

        feat_prior = _robust_unit_interval(feat_prior)
        deg_prior = _robust_unit_interval(deg_prior)
        return float(feature_weight) * feat_prior + float(degree_weight) * deg_prior


def calibrate_polarity_with_neighbor_knowledge(
    score: torch.Tensor,
    nk_prior: torch.Tensor,
    k_percent: float = 0.05,
    min_gain: float = 0.02,
    verbose: bool = False,
) -> Tuple[torch.Tensor, bool]:
    """Pick score orientation by maximizing alignment with neighbor-knowledge prior."""
    with torch.no_grad():
        s = score.reshape(-1).float()
        p = nk_prior.reshape(-1).float().to(s.device)
        n = int(s.numel())
        if n < 4 or p.numel() != n:
            return score, False

        k = max(int(n * float(k_percent)), 1)
        k = min(k, n // 2)
        if k < 1:
            return score, False

        p_np = p.detach().cpu().numpy().ravel()

        def _orientation_obj(v: torch.Tensor) -> float:
            v_np = v.detach().cpu().numpy().ravel()
            rho, _ = spearmanr(v_np, p_np)
            rho_f = 0.0 if rho is None or (isinstance(rho, float) and np.isnan(rho)) else float(rho)
            _, top_idx = torch.topk(v, k, largest=True)
            _, bot_idx = torch.topk(v, k, largest=False)
            tail_gap = float(torch.median(p[top_idx]) - torch.median(p[bot_idx]))
            return rho_f + tail_gap

        smin, smax = s.min(), s.max()
        sn = (s - smin) / (smax - smin + 1e-8)
        flipped = 1.0 - sn

        obj_keep = _orientation_obj(sn)
        obj_flip = _orientation_obj(flipped)
        gain = obj_flip - obj_keep

        if verbose:
            print(
                f"[NK-Probe] obj_keep={obj_keep:.4f}, obj_flip={obj_flip:.4f}, gain={gain:.4f}, min_gain={min_gain}",
                flush=True,
            )

        if gain > float(min_gain):
            return flipped, True
        return score, False


def calibrate_polarity_robust(
    score: torch.Tensor,
    local_prior: torch.Tensor,
    k_percent: float = 0.05,
    margin: float = 1.05,
    spearman_threshold: float = -0.1,
    verbose: bool = False,
) -> Tuple[torch.Tensor, bool]:
    """Unsupervised polarity: Spearman(score, prior); if rho < threshold flip. Else tail median on prior."""
    with torch.no_grad():
        s = score.reshape(-1).float()
        prior = local_prior.reshape(-1).float().to(s.device)
        n = int(s.numel())
        if n < 2 or prior.numel() != n:
            return score, False
        smin, smax = s.min(), s.max()
        span = float(smax - smin)
        if span <= 1e-12:
            return score, False

        s_np = s.detach().cpu().numpy().ravel()
        prior_np = prior.detach().cpu().numpy().ravel()
        if np.std(s_np) <= 1e-12 or np.std(prior_np) <= 1e-12:
            if verbose:
                print("[Robust-Probe] skip Spearman: near-constant score or prior", flush=True)
            rho_f = None
        else:
            rho, _ = spearmanr(s_np, prior_np)
            if rho is None or (isinstance(rho, float) and np.isnan(rho)):
                rho_f = None
            else:
                rho_f = float(rho)

        if rho_f is not None and rho_f < float(spearman_threshold):
            if verbose:
                print(f"[Robust-Probe] rho(score,prior)={rho_f:.3f} < {spearman_threshold}. Flipping.", flush=True)
            return 1.0 - (s - smin) / (smax - smin + 1e-8), True

        k = max(int(n * float(k_percent)), 1)
        k = min(k, n // 2)
        if k < 1:
            return score, False

        _, top_idx = torch.topk(s, k, largest=True)
        _, bot_idx = torch.topk(s, k, largest=False)
        prior_top_med = torch.median(prior[top_idx])
        prior_bot_med = torch.median(prior[bot_idx])

        if verbose:
            print(
                f"[Robust-Probe] tail medians: top-k prior={float(prior_top_med):.4f}, "
                f"bot-k prior={float(prior_bot_med):.4f} (margin={margin})",
                flush=True,
            )

        if float(prior_bot_med) > float(prior_top_med) * float(margin):
            if verbose:
                print("[Robust-Probe] bot tail median prior > top × margin. Flipping.", flush=True)
            return 1.0 - (s - smin) / (smax - smin + 1e-8), True

        return score, False
