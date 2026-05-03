# Polarity 严格无标签改造 — 审计报告

本文档满足「训练 / 推理 / score calibration / polarity determination 全程不读 ground-truth anomaly mask」的代码审计与复现实验记录。

---

## 1. 修改前：`data.y` 在何处参与极性（已删除）

| 位置 | 行为 |
|------|------|
| `model.py` `forward()` | `polarity_adapter in ("universal",)` 时调用 `compute_polarity_graph_signals(edge_index, x, data.y)`，用真异常掩码计算 `n_anom`、`hub_anomaly_neigh_deg_ratio`、`cos_anom_neigh`。 |
| `utils.py` `calibrate_polarity_universal` | 用上述 **label-derived** 量调节门控尺度（hub / cos）。 |
| `utils.py` `_universal_autovote_arbitration` | 规则 A/B/D 使用 `n_anom`、`na/n` 等 **标签比例**。 |

**未改动（本就不参与极性）：** `sample()` / 集成路径末尾的 `eval_roc_auc`、`eval_average_precision` 等 — 仅用于 **最终指标**，合法。

---

## 2. 修改后：`data.y` 仅出现位置

| 文件 | 用途 |
|------|------|
| `model.py` | `sample()`：在 **分数与极性适配器全部执行完毕** 后，`y_eval = data.y.bool()`，仅用于 `eval_*` 与 PR 曲线。 |
| `model.py` | `ensemble_score` 分支：在 `mean_scores = _apply_score_polarity_adapter(...)` **之后** `y_eval = data.y`，仅用于聚合指标。 |
| `scripts/dataset_graph_stats.py` | **离线**数据集说明脚本，不参与 `main_train` 训练/推理管线。 |

仓库内 `grep`（`*.py`）：`data.y` / `y_true` / `y_eval` 仅上述训练代码路径 + `dataset_graph_stats.py`。

**`eval_roc_auc` / `eval_average_precision`：** 仅在 `model.py` 指标段与可选 `FMGAD_POLARITY_DEBUG` 调试打印中出现；**未**用于 checkpoint 选择、超参搜索或极性分支（`run_fmgad_ablation.py` 的 `max(auc,...)` 仅为静态字符串黑名单，非运行时代码）。

---

## 3. 新的无标签图信号与极性依赖

### 3.1 `compute_polarity_graph_signals_unsup(edge_index, x, q=0.05)`

- **输入：** 仅 `edge_index`、`x`、`q`（默认 `0.05`，YAML：`polarity_unsup_proxy_q`）。
- **Proxy 可疑度：**  
  \(\pi_i = \mathrm{robust\_z}(\|x_i-\bar x_{\mathcal N(i)}\|_2) + \mathrm{robust\_z}(|\deg_i-\overline{\deg}_{\mathcal N(i)}|/(\cdot)) + \mathrm{robust\_z}(1-\mathrm{LCC}_i)\)  
  （实现见 `utils.py`，与 `robust_zscore` + clamp 一致。）
- **集合 M：** \(\pi\) 的 **Top-⌈q·N⌉** 节点。
- **输出字段：**  
  `n`, `n_proxy`, `mean_deg_all`, `deg_p95_to_mean`,  
  `proxy_neigh_deg_ratio`, `proxy_neigh_feature_cos`,  
  `graph_density`, `lcc_mean`, `lcc_p10`, `lcc_p90`。

### 3.2 `calibrate_polarity_universal` 门控自适应

- 仍用 **gated softmax** 融合 local / NK / structural 探针；**尺度** 由 `proxy_neigh_deg_ratio` 与 `proxy_neigh_feature_cos` 驱动（替代原 hub / cos_anom）。

### 3.3 `_universal_autovote_arbitration`（无 `n_anom` / 无标签比例）

仅使用：`n`, `mean_deg_all`, `deg_p95_to_mean`, `proxy_neigh_deg_ratio`, `proxy_neigh_feature_cos`, 以及 gated 诊断中的 **structural evidence raw**。

规则代号：  
`large_graph_proxyhub_plus_struct` / `mid_deg_band_plus_struct` / `strong_structural_raw` / `small_graph_proxyhetero_plus_struct`。

---

## 4. 五个数据集配置的 `polarity_adapter`

| 数据集 | `configs/{dataset}_best.yaml` |
|--------|-------------------------------|
| books / disney / enron / reddit / weibo | **`polarity_adapter: universal_no_y`**（与 `universal` 代码路径相同，均为严格无标签 graph signals） |

**说明：** 未按 AUC 为各数据集选择不同 adapter；`polarity_unsup_proxy_q: 0.05` 为 **全局统一** 默认值（非由验证/测试标签搜索得到）。

---

## 5. 复现实验：五数据集 × seeds `42,0,1,2,3`

- **旧版（label-derived graph signals + sweep 合并的 universal 超参）：**  
  `results/universal_5x5_sweep/summary_20260503_051346.json`
- **新版（`universal_no_y` + `compute_polarity_graph_signals_unsup` + 同上 sweep 默认超参）：**  
  `results/universal_5x5_sweep/summary_20260503_140854.json`

### 5.1 数据集平均 AUROC / AP（5 seeds）

| Dataset | 旧 AUROC mean | 新 AUROC mean | 旧 AP mean | 新 AP mean |
|---------|---------------|---------------|------------|------------|
| books | 0.596 | **0.617** | 0.030 | 0.030 |
| disney | 0.420 | **0.472** | 0.048 | **0.068** |
| enron | 0.835 | **0.836** | 0.004 | 0.003 |
| reddit | 0.548 | 0.546 | 0.037 | 0.037 |
| weibo | 0.942 | 0.942 | 0.344 | 0.344 |

### 5.2 25-run 全局平均 AUROC

| 版本 | macro mean AUROC |
|------|------------------|
| 旧 | 0.668 |
| 新 | **0.683** |

> 数值随 seed 与训练随机性波动；本对比仅反映 **同一代码框架、同一组 sweep 默认 YAML 覆盖** 下，**去掉 y 图信号后的单次复跑**。

---

## 6. 相关文件清单

- `utils.py`：`compute_polarity_graph_signals_unsup`；删除基于 `y` 的 `compute_polarity_graph_signals`；`calibrate_polarity_universal`、`_universal_autovote_arbitration` 更新。
- `model.py`：`universal_no_y` / `universal` 共用路径；`polarity_unsup_proxy_q`；`y_eval` 仅指标段。
- `main_train.py`：`polarity_unsup_proxy_q` 传入模型。
- `configs/*_best.yaml`、`configs/universal_five_benchmark.yaml`：`polarity_adapter: universal_no_y`。
- `scripts/run_universal_5x5_sweep.py`、`scripts/run_universal_param_tune.py`、`scripts/run_fmgad_ablation.py`：适配 `universal_no_y`。

---

*生成时间：与 `summary_20260503_140854` 复跑同一批次。*
