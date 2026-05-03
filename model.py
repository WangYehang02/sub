import os
import sys
import math
import time
import tqdm
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any, List

from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
from sklearn.metrics import auc, precision_recall_curve
from pygod.metric.metric import (
    eval_roc_auc,
    eval_average_precision,
    eval_recall_at_k,
    eval_precision_at_k,
)

from pygod.utils import load_data

FMGAD_ROOT = os.path.dirname(os.path.abspath(__file__))
if FMGAD_ROOT not in sys.path:
    sys.path.insert(0, FMGAD_ROOT)

from auto_encoder import GraphAE
from utils import (
    softmax_with_temperature,
    compute_smoothgnn_local_prior,
    compute_neighbor_knowledge_prior,
    compute_node_lcc_tensor,
    compute_node_degree_tensor,
    compute_polarity_graph_signals_unsup,
    robust_zscore,
    robust_unit_interval,
    calibrate_polarity_universal,
)
from flow_matching_model import MLPFlowMatching, FlowMatchingModel, sample_flow_matching, sample_flow_matching_free
from FMloss import flow_matching_loss, conditional_flow_matching_loss

from encoder import compute_dual_residuals_with_degree


def _robust_minmax_norm(
    score: torch.Tensor,
    eps: float = 1e-8,
    q_low: float = 0.01,
    q_high: float = 0.99,
) -> torch.Tensor:
    """Quantile-clipped min-max normalization into [0, 1]."""
    if score.dim() != 1:
        score = score.reshape(-1)
    score_f = score.float()
    low = torch.quantile(score_f, q_low)
    high = torch.quantile(score_f, q_high)
    clipped = torch.clamp(score_f, min=low, max=high)
    vmin = torch.min(clipped)
    vmax = torch.max(clipped)
    span = vmax - vmin
    if torch.abs(span) < eps:
        return torch.zeros_like(score_f)
    return (clipped - vmin) / (span + eps)


def _smooth_scores_by_graph(
    score: torch.Tensor, edge_index: torch.Tensor, alpha: float, device: torch.device
) -> torch.Tensor:
    """score_smoothed = (1-alpha)*score + alpha*mean(score[neighbors])."""
    if alpha <= 0.0 or edge_index.numel() == 0:
        return score
    src, dst = edge_index[0], edge_index[1]
    n = score.size(0)
    neigh_sum = torch.zeros(n, device=device, dtype=score.dtype)
    neigh_sum.index_add_(0, dst, score[src])
    deg = torch.zeros(n, device=device, dtype=score.dtype)
    deg.index_add_(0, dst, torch.ones_like(score[src]))
    deg = deg.clamp_min(1.0)
    neigh_mean = neigh_sum / deg
    return (1.0 - alpha) * score + alpha * neigh_mean


def _add_virtual_knn_edges(
    edge_index: torch.Tensor,
    h: torch.Tensor,
    degree_threshold: int,
    k: int,
    device: torch.device,
) -> torch.Tensor:
    """Append kNN edges in embedding space for low-degree nodes. Skipped when n > 50000 (O(n^2) sim matrix)."""
    n = h.size(0)
    if n > 50000:
        return edge_index
    with torch.no_grad():
        deg = torch.zeros(n, device=device, dtype=torch.long)
        deg.scatter_add_(0, edge_index[1], torch.ones(edge_index.size(1), device=device, dtype=torch.long))
        low_deg_mask = (deg < degree_threshold) & (deg >= 0)
        if low_deg_mask.sum() == 0:
            return edge_index
        h_norm = torch.nn.functional.normalize(h, p=2, dim=1)
        sim = torch.mm(h_norm, h_norm.t())
        sim.fill_diagonal_(-1e9)
        _, idx = sim.topk(min(k, n - 1), dim=1)
        new_edges = []
        for i in range(n):
            if not low_deg_mask[i]:
                continue
            for j in idx[i].tolist():
                if j != i:
                    new_edges.append([i, j])
        if not new_edges:
            return edge_index
        new_edges = torch.tensor(new_edges, device=device, dtype=edge_index.dtype).t()
        combined = torch.cat([edge_index, new_edges], dim=1)
        combined = torch.unique(combined, dim=1)
        return combined


class _GateParams(nn.Module):
    """Learnable gate for fusing local vs global residual by node degree."""

    def __init__(self, bias: float = 2.0, sharpness: float = 1.0):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(bias, dtype=torch.float32))
        self._raw_sharpness = nn.Parameter(torch.tensor(sharpness, dtype=torch.float32))

    @property
    def sharpness(self):
        return torch.nn.functional.softplus(self._raw_sharpness)


class ResFlowGAD(BaseTransform):
    """Graph anomaly detection via AE latent, dual residual, flow matching, reconstruction error."""

    def __init__(
        self,
        name: str = "FMGAD",
        hid_dim: Optional[int] = None,
        ae_epochs: int = 300,
        diff_epochs: int = 800,
        patience: int = 100,
        lr: float = 0.005,
        wd: float = 0.0,
        weight: float = 1.0,
        sample_steps: int = 50,
        ae_dropout: float = 0.3,
        ae_lr: float = 0.01,
        ae_alpha: float = 0.8,
        use_proto: bool = True,
        profile_efficiency: bool = False,
        proto_alpha: float = 0.01,
        residual_scale: float = 10.0,
        gate_bias: float = 2.0,
        gate_sharpness: float = 1.0,
        verbose: bool = True,
        use_virtual_neighbors: bool = True,
        virtual_degree_threshold: int = 5,
        virtual_k: int = 5,
        use_score_smoothing: bool = True,
        score_smoothing_alpha: float = 0.3,
        flow_t_sampling: str = "logit_normal",
        ensemble_score: bool = True,
        num_trial: int = 3,
        exp_tag: Optional[str] = None,
        smoothgnn_polarity: bool = False,
        smoothgnn_anchor_k_percent: float = 0.05,
        smoothgnn_anchor_margin: float = 1.05,
        smoothgnn_robust_spearman_threshold: float = -0.1,
        nk_polarity: bool = False,
        nk_feature_weight: float = 0.8,
        nk_degree_weight: float = 0.2,
        nk_min_flip_gain: float = 0.02,
        score_mode: str = "calibrated",
        score_mode_beta: float = 0.05,
        polarity_reg_weight: float = 0.0,
        polarity_reg_target_corr: float = 0.1,
        polarity_adapter: str = "universal_no_y",
        polarity_vote_q: float = 0.1,
        polarity_vote_margin: int = 1,
        polarity_min_confidence: float = 0.2,
        polarity_lcc_rho_strong: float = 0.04,
        polarity_deg_rho_strong: float = 0.04,
        polarity_connectivity_rel_gap: float = 0.02,
        lcc_spearman_threshold: float = -0.05,
        polarity_verbose: bool = False,
        polarity_use_local_probe: bool = True,
        polarity_use_nk_probe: bool = True,
        polarity_use_structural_probe: bool = True,
        polarity_gate_tau: float = 0.05,
        polarity_gate_margin: float = 0.02,
        polarity_gate_min_confidence: float = 0.10,
        polarity_gate_topk_percent: float = 0.05,
        polarity_struct_lcc_threshold: float = 0.04,
        polarity_struct_deg_threshold: float = 0.04,
        polarity_struct_density_gap: float = 0.02,
        polarity_autovote_fallback: bool = True,
        polarity_unsup_proxy_q: float = 0.05,
    ):
        self.name = name
        self.num_trial = num_trial
        self.hid_dim = hid_dim
        self.ae_epochs = ae_epochs
        self.diff_epochs = diff_epochs
        self.patience = patience
        self.lr = lr
        self.wd = wd
        self.weight = weight
        self.sample_steps = sample_steps
        self.verbose = verbose
        self.use_proto = bool(use_proto)
        self.profile_efficiency = bool(profile_efficiency)
        self.proto_alpha = proto_alpha
        self.residual_scale = residual_scale
        self.gate_module = _GateParams(bias=gate_bias, sharpness=gate_sharpness)
        self.use_virtual_neighbors = use_virtual_neighbors
        self.virtual_degree_threshold = virtual_degree_threshold
        self.virtual_k = virtual_k
        self.use_score_smoothing = use_score_smoothing
        self.score_smoothing_alpha = score_smoothing_alpha
        self.flow_t_sampling = flow_t_sampling
        self.ensemble_score = ensemble_score
        self.exp_tag = exp_tag
        self.smoothgnn_polarity = bool(smoothgnn_polarity)
        self.smoothgnn_anchor_k_percent = float(smoothgnn_anchor_k_percent)
        self.smoothgnn_anchor_margin = float(smoothgnn_anchor_margin)
        self.smoothgnn_robust_spearman_threshold = float(smoothgnn_robust_spearman_threshold)
        self.nk_polarity = bool(nk_polarity)
        self.nk_feature_weight = float(nk_feature_weight)
        self.nk_degree_weight = float(nk_degree_weight)
        self.nk_min_flip_gain = float(nk_min_flip_gain)
        self.score_mode = str(score_mode)
        self.score_mode_beta = float(score_mode_beta)
        self.polarity_reg_weight = float(polarity_reg_weight)
        self.polarity_reg_target_corr = float(polarity_reg_target_corr)
        _pad = str(polarity_adapter).strip()
        if _pad == "universal":
            raise ValueError(
                'polarity_adapter "universal" is no longer supported; use "universal_no_y" (strict label-free).'
            )
        if _pad not in ("universal_no_y", "none"):
            raise ValueError(
                f'polarity_adapter must be "universal_no_y" or "none", got {_pad!r}. '
                'Legacy values "nk", "auto_vote", and "universal" are disabled in this submission build.'
            )
        self.polarity_adapter = _pad
        self.polarity_vote_q = float(polarity_vote_q)
        self.polarity_vote_margin = int(polarity_vote_margin)
        self.polarity_min_confidence = float(polarity_min_confidence)
        self.polarity_lcc_rho_strong = float(polarity_lcc_rho_strong)
        self.polarity_deg_rho_strong = float(polarity_deg_rho_strong)
        self.polarity_connectivity_rel_gap = float(polarity_connectivity_rel_gap)
        self.lcc_spearman_threshold = float(lcc_spearman_threshold)
        self.polarity_verbose = bool(polarity_verbose)
        self.polarity_use_local_probe = bool(polarity_use_local_probe)
        self.polarity_use_nk_probe = bool(polarity_use_nk_probe)
        self.polarity_use_structural_probe = bool(polarity_use_structural_probe)
        self.polarity_gate_tau = float(polarity_gate_tau)
        self.polarity_gate_margin = float(polarity_gate_margin)
        self.polarity_gate_min_confidence = float(polarity_gate_min_confidence)
        self.polarity_gate_topk_percent = float(polarity_gate_topk_percent)
        self.polarity_struct_lcc_threshold = float(polarity_struct_lcc_threshold)
        self.polarity_struct_deg_threshold = float(polarity_struct_deg_threshold)
        self.polarity_struct_density_gap = float(polarity_struct_density_gap)
        self.polarity_autovote_fallback = bool(polarity_autovote_fallback)
        self.polarity_unsup_proxy_q = float(polarity_unsup_proxy_q)
        self._smoothgnn_prior = None  # type: Optional[torch.Tensor]
        self._nk_prior = None  # type: Optional[torch.Tensor]
        self._polarity_anchor = None  # type: Optional[torch.Tensor]
        self._node_lcc = None
        self._node_degree = None
        self._last_auto_vote_diag = None
        self._polarity_graph_signals = None  # type: Optional[Dict[str, Any]]
        self._gated_local_prior = None  # type: Optional[torch.Tensor]
        self._gated_nk_prior = None  # type: Optional[torch.Tensor]
        self._last_universal_diag = None

        self.ae_dropout = ae_dropout
        self.ae_lr = ae_lr
        self.ae_alpha = ae_alpha

        self.ae = None  # type: Optional[GraphAE]
        self.dm = None  # type: Optional[FlowMatchingModel]
        self.dm_proto = None  # type: Optional[FlowMatchingModel]
        self.proto = None  # type: Optional[torch.Tensor]

        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.timesteps = 100

    def _apply_universal_polarity(self, score: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if getattr(self, "polarity_adapter", "universal_no_y") != "universal_no_y":
            return score
        if (
            self._node_lcc is None
            or self._node_degree is None
            or self._polarity_graph_signals is None
        ):
            if self.polarity_verbose:
                print("[universal] missing cache or graph signals, keep score.", flush=True)
            return score
        lp = self._gated_local_prior if getattr(self, "polarity_use_local_probe", True) else None
        nk = self._gated_nk_prior if getattr(self, "polarity_use_nk_probe", True) else None
        score2, _flipped, diag = calibrate_polarity_universal(
            score,
            edge_index,
            graph_signals=self._polarity_graph_signals,
            local_prior=lp,
            nk_prior=nk,
            lcc=self._node_lcc.to(score.device),
            degree=self._node_degree.to(score.device),
            use_local=bool(getattr(self, "polarity_use_local_probe", True)),
            use_nk=bool(getattr(self, "polarity_use_nk_probe", True)),
            use_structural=bool(getattr(self, "polarity_use_structural_probe", True)),
            topk_percent=float(getattr(self, "polarity_gate_topk_percent", 0.05)),
            structural_vote_q=float(self.polarity_vote_q),
            struct_lcc_threshold=float(self.polarity_struct_lcc_threshold),
            struct_deg_threshold=float(self.polarity_struct_deg_threshold),
            struct_density_gap_threshold=float(self.polarity_struct_density_gap),
            gate_tau=float(self.polarity_gate_tau),
            gate_margin=float(self.polarity_gate_margin),
            min_confidence=float(self.polarity_gate_min_confidence),
            verbose=bool(self.polarity_verbose),
            autovote_fallback=bool(getattr(self, "polarity_autovote_fallback", True)),
            autovote_kwargs={
                "q": float(self.polarity_vote_q),
                "margin": int(self.polarity_vote_margin),
                "min_confidence": float(self.polarity_min_confidence),
                "lcc_rho_strong": float(self.polarity_lcc_rho_strong),
                "deg_rho_strong": float(self.polarity_deg_rho_strong),
                "connectivity_rel_gap": float(self.polarity_connectivity_rel_gap),
                "legacy_lcc_threshold": float(self.lcc_spearman_threshold),
                "verbose": bool(self.polarity_verbose),
            },
        )
        self._last_universal_diag = diag
        return score2

    def _apply_score_polarity_adapter(self, score: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        pa = getattr(self, "polarity_adapter", "universal_no_y")
        if pa == "none":
            return score
        if pa == "universal_no_y":
            return self._apply_universal_polarity(score, edge_index)
        raise ValueError(
            f'Unsupported polarity_adapter {pa!r}; submission build allows only "universal_no_y" or "none".'
        )

    def _load_dataset(self, dset: str):
        """PyGOD 内置图异常检测数据集：books / disney / enron / reddit / weibo。"""
        return load_data(dset)

    def _ensure_save_dir(self, dset: str):
        # 默认保存到 cwd/models；可通过环境变量 FMGAD_MODEL_ROOT 重定向到大容量磁盘
        model_root = os.environ.get("FMGAD_MODEL_ROOT", os.path.join(os.getcwd(), "models"))
        save_dir = os.path.join(model_root, dset, "full_batch")
        os.makedirs(save_dir, exist_ok=True)
        return save_dir

    def _build_z(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return z = [h; scaled fused residual], h, and fused residual."""
        h = self.ae.encode(x, edge_index)
        dev = h.device
        if self.use_virtual_neighbors and getattr(self, "virtual_degree_threshold", 5) is not None:
            edge_index = _add_virtual_knn_edges(
                edge_index, h,
                self.virtual_degree_threshold,
                getattr(self, "virtual_k", 5),
                dev,
            )

        r_global, r_local, deg = compute_dual_residuals_with_degree(h, edge_index)
        bias = self.gate_module.bias.to(dev)
        sharpness = self.gate_module.sharpness.to(dev)
        alpha = torch.sigmoid((deg - bias) * sharpness)
        r_fused = alpha * r_local + (1.0 - alpha) * r_global
        r_final = r_fused * self.residual_scale
        z = torch.cat([h, r_final], dim=1)
        return z, h, r_final

    def forward(self, dset: str):
        def _sync_cuda() -> None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        self.dataset = dset
        data = self._load_dataset(dset)
        num_nodes = int(getattr(data, "num_nodes", data.x.size(0)))
        num_edges = int(data.edge_index.size(1)) if hasattr(data, "edge_index") else 0
        num_features = int(data.x.size(1)) if hasattr(data, "x") and data.x.dim() == 2 else 0
        train_time_sec = None
        inference_time_sec = None
        peak_gpu_mem_mb = None
        peak_gpu_reserved_mb = None
        num_parameters = None
        if bool(getattr(self, "profile_efficiency", False)):
            if torch.cuda.is_available():
                _sync_cuda()
                torch.cuda.reset_peak_memory_stats()
            train_time_sec = 0.0
            inference_time_sec = 0.0
        self._last_auto_vote_diag = None
        self._smoothgnn_prior = None
        self._nk_prior = None
        self._last_universal_diag = None
        _pa = getattr(self, "polarity_adapter", "universal_no_y")
        if _pa == "universal_no_y":
            self._node_lcc = compute_node_lcc_tensor(data.edge_index.detach().cpu(), data.x.size(0))
            self._node_degree = compute_node_degree_tensor(data.edge_index.detach().cpu(), data.x.size(0))
            self._polarity_graph_signals = compute_polarity_graph_signals_unsup(
                data.edge_index.detach().cpu(),
                data.x.detach().cpu(),
                q=float(getattr(self, "polarity_unsup_proxy_q", 0.05)),
            )
            if bool(getattr(self, "polarity_use_local_probe", True)):
                if self.verbose:
                    print("Precomputing universal local prior (SmoothGNN-style)...", flush=True)
                self._gated_local_prior = compute_smoothgnn_local_prior(data.x.cpu(), data.edge_index.cpu())
            else:
                self._gated_local_prior = None
            if bool(getattr(self, "polarity_use_nk_probe", True)):
                if self.verbose:
                    print("Precomputing universal neighbor-knowledge prior...", flush=True)
                self._gated_nk_prior = compute_neighbor_knowledge_prior(
                    data.x.cpu(),
                    data.edge_index.cpu(),
                    feature_weight=float(getattr(self, "nk_feature_weight", 0.8)),
                    degree_weight=float(getattr(self, "nk_degree_weight", 0.2)),
                )
            else:
                self._gated_nk_prior = None
        else:
            self._node_lcc = None
            self._node_degree = None
            self._polarity_graph_signals = None
            self._gated_local_prior = None
            self._gated_nk_prior = None
        # Legacy YAML keys smoothgnn_polarity / nk_polarity are ignored on the submission path (no mid-path flip).
        self._smoothgnn_prior = None
        self._nk_prior = None
        if self.hid_dim is None:
            self.hid_dim = 2 ** int(math.log2(data.x.size(1)) - 1)

        # AE
        self.ae = GraphAE(in_dim=data.num_node_features, hid_dim=self.hid_dim, dropout=self.ae_dropout).cuda()
        save_dir = self._ensure_save_dir(dset)
        ae_path = os.path.join(
            save_dir,
            f"ae_drop{self.ae_dropout}_lr{self.ae_lr}_alpha{self.ae_alpha}_hid{self.hid_dim}",
        )
        # 并发运行时避免多个进程写入同一 checkpoint 文件
        run_tag = self.exp_tag if self.exp_tag else f"run_{os.getpid()}_{int(time.time() * 1000)}"
        _tag_suffix = os.environ.get("FMGAD_RUN_TAG_SUFFIX", "").strip()
        if _tag_suffix:
            run_tag = f"{run_tag}_{_tag_suffix}"
        ae_path = os.path.join(ae_path, run_tag)
        os.makedirs(ae_path, exist_ok=True)

        if bool(getattr(self, "profile_efficiency", False)):
            _sync_cuda()
            _t_train = time.perf_counter()
        ae_ckpt = self._train_ae_once(data, ae_path)
        if bool(getattr(self, "profile_efficiency", False)):
            _sync_cuda()
            train_time_sec += time.perf_counter() - _t_train
        if self.verbose:
            print(f"loading AE checkpoint: {ae_ckpt:04d}")
        ae_dict = torch.load(os.path.join(ae_path, f"{ae_ckpt}.pt"))
        self.ae.load_state_dict(ae_dict["state_dict"])
        self.gate_module = self.gate_module.to(next(self.ae.parameters()).device)

        # 2) trials
        num_trial = getattr(self, "num_trial", 3)
        dm_auc, dm_ap, dm_rec, dm_auprc, dm_f1 = [], [], [], [], []

        for _ in tqdm.tqdm(range(num_trial)):
            # z_dim = 2*hid_dim
            z_dim = 2 * self.hid_dim

            # free model: cond_dim=None => 用全0 context
            velocity_free = MLPFlowMatching(d_in=z_dim, dim_t=512, cond_dim=None).cuda()
            self.dm = FlowMatchingModel(velocity_fn=velocity_free, hid_dim=z_dim).cuda()
            if bool(getattr(self, "profile_efficiency", False)):
                _sync_cuda()
                _t_train = time.perf_counter()
            proto_h = self._train_dm_free(data, ae_path)
            if bool(getattr(self, "profile_efficiency", False)):
                _sync_cuda()
                train_time_sec += time.perf_counter() - _t_train

            dm_dict = torch.load(os.path.join(ae_path, "dm_self.pt"))
            self.dm.load_state_dict(dm_dict["state_dict"])
            if "gate_state" in dm_dict:
                self.gate_module.load_state_dict(dm_dict["gate_state"])
            self.proto = dm_dict["prototype"]  # [hid_dim]

            if bool(getattr(self, "use_proto", True)):
                # proto model: cond_dim = hid_dim（只条件在 h 的原型上）
                velocity_proto = MLPFlowMatching(d_in=z_dim, dim_t=512, cond_dim=self.hid_dim).cuda()
                self.dm_proto = FlowMatchingModel(velocity_fn=velocity_proto, hid_dim=z_dim).cuda()
                if bool(getattr(self, "profile_efficiency", False)):
                    _sync_cuda()
                    _t_train = time.perf_counter()
                self._train_dm_proto(data, ae_path)
                if bool(getattr(self, "profile_efficiency", False)):
                    _sync_cuda()
                    train_time_sec += time.perf_counter() - _t_train
                dm_proto_dict = torch.load(os.path.join(ae_path, "proto_dm_self.pt"))
                self.dm_proto.load_state_dict(dm_proto_dict["state_dict"])
                if bool(getattr(self, "profile_efficiency", False)):
                    _sync_cuda()
                    _t_infer = time.perf_counter()
                ret = self.sample(self.dm_proto, self.dm, data)
                if bool(getattr(self, "profile_efficiency", False)):
                    _sync_cuda()
                    inference_time_sec += time.perf_counter() - _t_infer
            else:
                # Hard no-proto: no prototype-conditioned branch in training/inference.
                self.dm_proto = None
                self.proto = None
                if bool(getattr(self, "profile_efficiency", False)):
                    _sync_cuda()
                    _t_infer = time.perf_counter()
                ret = self.sample(None, self.dm, data)
                if bool(getattr(self, "profile_efficiency", False)):
                    _sync_cuda()
                    inference_time_sec += time.perf_counter() - _t_infer
            if len(ret) == 6:
                auc_this, ap_this, rec_this, auprc_this, f1_this, scores = ret
                if not hasattr(self, "_ensemble_scores"):
                    self._ensemble_scores = []
                self._ensemble_scores.append(scores)
            else:
                auc_this, ap_this, rec_this, auprc_this, f1_this = ret
            dm_auc.append(auc_this)
            dm_ap.append(ap_this)
            dm_rec.append(rec_this)
            dm_auprc.append(auprc_this)
            dm_f1.append(f1_this)

        if getattr(self, "ensemble_score", False) and hasattr(self, "_ensemble_scores") and len(self._ensemble_scores) > 0:
            # 多 trial 分数取平均，再按平均分数计算一次指标
            stacked = torch.stack(self._ensemble_scores)  # [num_trial, N]
            mean_scores = stacked.mean(dim=0)  # [N]
            if torch.isnan(mean_scores).any() or torch.isinf(mean_scores).any():
                mean_scores = torch.nan_to_num(mean_scores, nan=0.0, posinf=0.0, neginf=0.0)
            mean_scores = self._apply_score_polarity_adapter(mean_scores, data.edge_index)

            y_eval = data.y  # evaluation labels only (after all score / polarity steps)

            pyg_auc = eval_roc_auc(y_eval, mean_scores)
            pyg_ap = eval_average_precision(y_eval, mean_scores)
            pyg_rec = eval_recall_at_k(y_eval, mean_scores, int(y_eval.sum()))
            pyg_prec = eval_precision_at_k(y_eval, mean_scores, int(y_eval.sum()))

            y_np = y_eval.cpu().numpy()
            p, r, _ = precision_recall_curve(y_np, mean_scores.cpu().numpy())
            pyg_auprc = auc(r, p)
            pyg_f1 = 2 * pyg_prec * pyg_rec / (pyg_prec + pyg_rec) if (pyg_prec + pyg_rec) > 0 else 0.0
            dm_auc = torch.tensor([float(pyg_auc)])
            dm_ap = torch.tensor([float(pyg_ap)])
            dm_rec = torch.tensor([float(pyg_rec)])
            dm_auprc = torch.tensor([float(pyg_auprc)])
            dm_f1 = torch.tensor([float(pyg_f1)])
            del self._ensemble_scores
        else:
            dm_auc = torch.tensor(dm_auc)
            dm_ap = torch.tensor(dm_ap)
            dm_rec = torch.tensor(dm_rec)
            dm_auprc = torch.tensor(dm_auprc)
            dm_f1 = torch.tensor(dm_f1)

        print(
            "Final AUC: {:.4f}±{:.4f} ({:.4f})\t"
            "Final AP: {:.4f}±{:.4f} ({:.4f})\t"
            "Final Recall: {:.4f}±{:.4f} ({:.4f})\t"
            "Final AUPRC: {:.4f}±{:.4f} ({:.4f})\t"
            "Final F1@k: {:.4f}±{:.4f} ({:.4f})".format(
                torch.mean(dm_auc),
                torch.std(dm_auc),
                torch.max(dm_auc),
                torch.mean(dm_ap),
                torch.std(dm_ap),
                torch.max(dm_ap),
                torch.mean(dm_rec),
                torch.std(dm_rec),
                torch.max(dm_rec),
                torch.mean(dm_auprc),
                torch.std(dm_auprc),
                torch.max(dm_auprc),
                torch.mean(dm_f1),
                torch.std(dm_f1),
                torch.max(dm_f1),
            )
        )

        if bool(getattr(self, "profile_efficiency", False)):
            _sync_cuda()
            if torch.cuda.is_available():
                peak_gpu_mem_mb = float(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)
                peak_gpu_reserved_mb = float(torch.cuda.max_memory_reserved() / 1024.0 / 1024.0)
            param_count = 0
            modules_for_count = [self.ae, self.dm, self.gate_module]
            if getattr(self, "dm_proto", None) is not None:
                modules_for_count.append(self.dm_proto)
            for mod in modules_for_count:
                if mod is not None:
                    param_count += int(sum(p.numel() for p in mod.parameters()))
            num_parameters = int(param_count)

        out = {
            "auc_mean": float(torch.mean(dm_auc)),
            "auc_std": float(torch.std(dm_auc)),
            "ap_mean": float(torch.mean(dm_ap)),
            "ap_std": float(torch.std(dm_ap)),
            "rec_mean": float(torch.mean(dm_rec)),
            "rec_std": float(torch.std(dm_rec)),
            "auprc_mean": float(torch.mean(dm_auprc)),
            "auprc_std": float(torch.std(dm_auprc)),
            "f1_mean": float(torch.mean(dm_f1)),
            "f1_std": float(torch.std(dm_f1)),
            "polarity_adapter": self.polarity_adapter,
            "auto_vote_diagnostics": self._last_auto_vote_diag,
            "universal_polarity_diagnostics": self._last_universal_diag,
            "polarity_graph_signals": self._polarity_graph_signals,
        }
        if bool(getattr(self, "profile_efficiency", False)):
            out.update(
                {
                    "profile_efficiency": True,
                    "train_time_sec": float(train_time_sec) if train_time_sec is not None else None,
                    "inference_time_sec": float(inference_time_sec) if inference_time_sec is not None else None,
                    "peak_gpu_mem_mb": peak_gpu_mem_mb,
                    "peak_gpu_reserved_mb": peak_gpu_reserved_mb,
                    "num_parameters": num_parameters,
                    "num_nodes": num_nodes,
                    "num_edges": num_edges,
                    "num_features": num_features,
                    "sample_steps": int(self.sample_steps),
                }
            )
        return out

    def _train_ae_once(self, data, ae_path: str) -> int:
        if self.verbose:
            print("Training autoencoder...")

        optimizer = torch.optim.Adam(self.ae.parameters(), lr=self.ae_lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

        best_loss = float("inf")
        best_epoch = 0
        patience = 0

        x = data.x.cuda().to(torch.float32)
        edge_index = data.edge_index.cuda()
        s = to_dense_adj(edge_index)[0].cuda()

        for epoch in range(1, self.ae_epochs + 1):
            self.ae.train()
            optimizer.zero_grad()

            x_, s_, _ = self.ae(x, edge_index)
            score = self.ae.loss_func(x, x_, s, s_, self.ae_alpha)
            loss = torch.mean(score)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_epoch = epoch
                patience = 0
                torch.save({"state_dict": self.ae.state_dict()}, os.path.join(ae_path, f"{best_epoch}.pt"))
            else:
                patience += 1
                if patience >= self.patience:
                    if self.verbose:
                        print("AE early stopping")
                    break

            if self.verbose and epoch % 50 == 0:
                print(f"AE Epoch {epoch:04d} loss={loss.item():.6f}")

        return best_epoch

    def _normalize_clip(self, inputs: torch.Tensor) -> torch.Tensor:
        x, _, _ = self._normalize_clip_with_stats(inputs)
        return x

    def _normalize_clip_with_stats(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = inputs.mean(dim=0, keepdim=True)
        std = inputs.std(dim=0, keepdim=True) + 1e-8
        x = (inputs - mean) / std
        x = torch.clamp(x, -10.0, 10.0)
        return x, mean, std

    def _denormalize(self, normed: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return normed * std + mean

    def _train_dm_free(self, data, ae_path: str) -> torch.Tensor:
        from flow_matching_model import sample_flow_matching

        if self.verbose:
            print("Training FM free model...")

        fm_lr = self.lr * 0.5
        params = list(self.dm.parameters()) + list(self.gate_module.parameters())
        optimizer = torch.optim.Adam(params, lr=fm_lr, weight_decay=self.wd)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

        best_loss = float("inf")
        patience = 0
        proto_h = None
        with torch.no_grad():
            x0 = data.x.cuda().to(torch.float32)
            e0 = data.edge_index.cuda()
            _, h0, _ = self._build_z(x0, e0)
            proto_h_init = torch.mean(h0, dim=0).detach()

        for epoch in range(self.diff_epochs):
            x = data.x.cuda().to(torch.float32)
            edge_index = data.edge_index.cuda()
            z, h, r_final = self._build_z(x, edge_index)

            z = self._normalize_clip(z)
            if torch.isnan(z).any() or torch.isinf(z).any():
                continue

            graph_context = torch.zeros(1, z.shape[1], device=z.device)
            loss = flow_matching_loss(self.dm.velocity_fn, z, graph_context, reduction="mean")
            if torch.isnan(loss) or torch.isinf(loss):
                continue

            with torch.no_grad():
                noise = torch.randn_like(z)
                reconstructed = sample_flow_matching(self.dm.velocity_fn, noise, num_steps=10, proto=None, proto_alpha=None)
                if torch.isnan(reconstructed).any() or torch.isinf(reconstructed).any():
                    reconstructed = z.clone()
                recon_h = reconstructed[:, : self.hid_dim]

            if epoch == 0:
                proto_h = torch.mean(h, dim=0)  # [hid_dim]
            else:
                proto_expanded = proto_h.unsqueeze(0)
                s_v = self.cos(proto_expanded, recon_h)
                weight = softmax_with_temperature(s_v, t=5).reshape(1, -1)
                proto_h = torch.mm(weight, recon_h).squeeze(0).detach()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 0.5)
            optimizer.step()
            scheduler.step()

            if self.verbose and epoch % 20 == 0:
                print(f"FM-free Epoch {epoch:04d} loss={loss.item():.6f}")

            if loss.item() < best_loss:
                best_loss = loss.item()
                patience = 0
                save_dict = {
                    "state_dict": self.dm.state_dict(),
                    "prototype": proto_h,
                    "gate_state": self.gate_module.state_dict(),
                }
                torch.save(save_dict, os.path.join(ae_path, "dm_self.pt"))
            else:
                patience += 1
                if patience >= self.patience:
                    if self.verbose:
                        print("FM-free early stopping")
                    break

        dm_path = os.path.join(ae_path, "dm_self.pt")
        if not os.path.exists(dm_path):
            proto_fallback = proto_h if proto_h is not None else proto_h_init
            save_dict = {
                "state_dict": self.dm.state_dict(),
                "prototype": proto_fallback,
                "gate_state": self.gate_module.state_dict(),
            }
            torch.save(save_dict, dm_path)
            if self.verbose:
                print("FM-free: fallback save")

        return proto_h

    def _train_dm_proto(self, data, ae_path: str):
        if self.verbose:
            print("Training FM proto model...")

        fm_lr = self.lr * 0.5
        params_proto = list(self.dm_proto.parameters()) + list(self.gate_module.parameters())
        optimizer = torch.optim.Adam(params_proto, lr=fm_lr, weight_decay=self.wd)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

        best_loss = float("inf")
        patience = 0

        for epoch in range(self.diff_epochs):
            x = data.x.cuda().to(torch.float32)
            edge_index = data.edge_index.cuda()
            z, _, _ = self._build_z(x, edge_index)
            z = self._normalize_clip(z)
            if torch.isnan(z).any() or torch.isinf(z).any():
                continue

            proto_context = self.proto.unsqueeze(0) if self.proto.dim() == 1 else self.proto.mean(dim=0, keepdim=True)
            loss = conditional_flow_matching_loss(
                self.dm_proto.velocity_fn,
                z,
                proto_context,
                t_sampling=self.flow_t_sampling,
                reduction="mean",
            )

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params_proto, 0.5)
            optimizer.step()
            scheduler.step()

            if self.verbose and epoch % 20 == 0:
                print(f"FM-proto Epoch {epoch:04d} loss={loss.item():.6f}")

            if loss.item() < best_loss:
                best_loss = loss.item()
                patience = 0
                torch.save({"state_dict": self.dm_proto.state_dict()}, os.path.join(ae_path, "proto_dm_self.pt"))
            else:
                patience += 1
                if patience >= self.patience:
                    if self.verbose:
                        print("FM-proto early stopping")
                    break

        proto_path = os.path.join(ae_path, "proto_dm_self.pt")
        if not os.path.exists(proto_path):
            torch.save({"state_dict": self.dm_proto.state_dict()}, proto_path)
            if self.verbose:
                print("FM-proto: fallback save")

    def sample(self, proto_model, free_model, data):
        self.ae.eval()
        if proto_model is not None:
            proto_model.eval()
        free_model.eval()

        proto_net = proto_model.velocity_fn if proto_model is not None else None
        free_net = free_model.velocity_fn

        x = data.x.cuda().to(torch.float32)
        edge_index = data.edge_index.cuda()

        z0, _, _ = self._build_z(x, edge_index)
        z0 = self._normalize_clip(z0)
        noise = torch.randn_like(z0)

        proto_context = None
        if bool(getattr(self, "use_proto", True)) and self.proto is not None:
            proto_context = self.proto.unsqueeze(0) if self.proto.dim() == 1 else self.proto.mean(dim=0, keepdim=True)
            if proto_context.dim() == 1:
                proto_context = proto_context.unsqueeze(0)

        s = to_dense_adj(edge_index)[0].cuda()

        # Fix step to 1 and avoid label-based step selection.
        num_steps = 1
        if bool(getattr(self, "use_proto", True)) and proto_net is not None and proto_context is not None:
            reconstructed = sample_flow_matching_free(
                proto_net,
                free_net,
                noise,
                num_steps,
                proto=proto_context,
                proto_alpha=self.proto_alpha,
                weight=self.weight,
            )
        else:
            # Hard no-proto: free-only velocity branch for inference.
            reconstructed = sample_flow_matching(
                free_net,
                noise,
                num_steps=num_steps,
                proto=None,
                proto_alpha=None,
            )

        h_hat = reconstructed[:, : self.hid_dim]
        x_, s_ = self.ae.decode(h_hat, edge_index)
        score_recon = self.ae.loss_func(x, x_, s, s_, self.ae_alpha)

        raw_score = score_recon
        score = raw_score

        if getattr(self, "use_score_smoothing", False) and edge_index.numel() > 0:
            score = _smooth_scores_by_graph(score, edge_index, self.score_smoothing_alpha, score.device)

        score_mode = os.environ.get("FMGAD_SCORE_MODE", getattr(self, "score_mode", "calibrated"))
        beta = float(os.environ.get("FMGAD_SCORE_MODE_BETA", str(getattr(self, "score_mode_beta", 0.05))))
        # Legacy mid-path priors (_smoothgnn_prior / _nk_prior) are disabled; anchor uses reconstruction score only.
        anchor = robust_unit_interval(raw_score.detach())

        raw_term = robust_zscore(raw_score.detach())
        if score_mode == "raw":
            score = raw_score
        elif score_mode == "neg_raw":
            score = -raw_score
        elif score_mode == "anchor":
            score = anchor
        elif score_mode == "anchor_plus_raw_005":
            score = anchor + beta * raw_term
        elif score_mode == "anchor_plus_negraw_005":
            score = anchor - beta * raw_term
        elif score_mode == "polarity_safe":
            score = anchor + beta * raw_term.abs()

        if not bool(getattr(self, "ensemble_score", False)):
            score = self._apply_score_polarity_adapter(score, edge_index)

        # Ground-truth mask: evaluation / optional debug only (must not affect score path above).
        y_eval = data.y.bool()

        if os.environ.get("FMGAD_POLARITY_DEBUG", "0") == "1":
            raw_cpu = raw_score.detach().cpu()
            final_cpu = score.detach().cpu()
            if torch.isnan(raw_cpu).any() or torch.isinf(raw_cpu).any():
                raw_cpu = torch.nan_to_num(raw_cpu, nan=0.0, posinf=0.0, neginf=0.0)
            if torch.isnan(final_cpu).any() or torch.isinf(final_cpu).any():
                final_cpu = torch.nan_to_num(final_cpu, nan=0.0, posinf=0.0, neginf=0.0)

            fmin, fmax = final_cpu.min(), final_cpu.max()
            final_norm = (final_cpu - fmin) / (fmax - fmin + 1e-8)
            one_minus = 1.0 - final_norm

            print(
                "[PolarityDebug] dset={} steps={} "
                "score[min,max,mean]=[{:.6f},{:.6f},{:.6f}] "
                "mode={} auc_raw={:.6f} auc_final={:.6f} auc_neg={:.6f} auc_one_minus={:.6f}".format(
                    getattr(self, "dataset", "unknown"),
                    num_steps,
                    float(final_cpu.min()),
                    float(final_cpu.max()),
                    float(final_cpu.mean()),
                    score_mode,
                    float(eval_roc_auc(y_eval, raw_cpu)),
                    float(eval_roc_auc(y_eval, final_cpu)),
                    float(eval_roc_auc(y_eval, -final_cpu)),
                    float(eval_roc_auc(y_eval, one_minus)),
                ),
                flush=True,
            )

        scores_cpu = score.detach().cpu()
        if torch.isnan(scores_cpu).any() or torch.isinf(scores_cpu).any():
            scores_cpu = torch.nan_to_num(scores_cpu, nan=0.0, posinf=0.0, neginf=0.0)
        pyg_auc = eval_roc_auc(y_eval, scores_cpu)
        pyg_ap = eval_average_precision(y_eval, scores_cpu)
        pyg_rec = eval_recall_at_k(y_eval, scores_cpu, int(y_eval.sum()))
        pyg_prec = eval_precision_at_k(y_eval, scores_cpu, int(y_eval.sum()))
        p, r, _ = precision_recall_curve(y_eval.numpy(), scores_cpu.numpy())
        pyg_auprc = auc(r, p)

        if (pyg_prec + pyg_rec) > 0:
            f1_at_k = 2 * pyg_prec * pyg_rec / (pyg_prec + pyg_rec)
        else:
            f1_at_k = 0.0

        if self.verbose:
            print(
                "steps:{},pyg_AUC: {:.4f}, pyg_AP: {:.4f}, pyg_Recall: {:.4f}, F1@k: {:.4f}, AUPRC: {:.4f}".format(
                    num_steps, pyg_auc, pyg_ap, pyg_rec, f1_at_k, pyg_auprc
                )
            )

        if getattr(self, "ensemble_score", False):
            return float(pyg_auc), float(pyg_ap), float(pyg_rec), float(pyg_auprc), float(f1_at_k), scores_cpu.clone()
        return float(pyg_auc), float(pyg_ap), float(pyg_rec), float(pyg_auprc), float(f1_at_k)