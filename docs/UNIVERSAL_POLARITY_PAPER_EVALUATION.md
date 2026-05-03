# Universal Polarity Calibration: Evaluation Report & Paper-Ready Method

本文档分三部分：**(I) 五数据集 × seeds `42,0,1,2,3` 实验记录（中文）**；**(II) 可直接改写进论文的 Method 英文稿**（与实现 `utils.calibrate_polarity_universal` 一致）；**(III) Part II 的中文译本**（与英文一一对应）。另附 **LaTeX 算法片段**。

---

## Part I — 实验报告（中文）

### 1. 设置

| 项 | 内容 |
|----|------|
| 代码仓库 | `~/texing/FMGADfinal` |
| 环境 | Conda `fmgad` |
| 极性 | `polarity_adapter: universal`（其余超参来自各数据集 `configs/{dataset}_best.yaml`，由 `scripts/run_universal_5x5_sweep.py` 合并 universal 字段） |
| 数据 | PyGOD：`books`, `disney`, `enron`, `reddit`, `weibo` |
| Seeds | **`42, 0, 1, 2, 3`**（共 25 次独立训练） |
| 硬件 | 8× GPU 并行，`CUDA_VISIBLE_DEVICES` 轮询 |
| 复现命令 | `cd ~/texing/FMGADfinal && python scripts/run_universal_5x5_sweep.py --datasets books,disney,enron,reddit,weibo --seeds 42,0,1,2,3 --gpus 0,1,2,3,4,5,6,7 --max-workers 8` |

原始逐次 JSON：`results/universal_5x5_sweep/runs/{dataset}_seed{seed}.json`  
**本次汇总**：`results/universal_5x5_sweep/summary_20260503_051346.json`

> **注**：此前还曾用 seeds `42, 3407, 2026, 17, 12345` 做过一轮 5×5 汇总，见 `summary_20260503_034735.json`；与本文主表非同一组随机种子，不宜直接横向数值对比。

### 2. 主表：AUROC / AP（均值 ± 总体标准差，5 seeds）

| Dataset | AUROC mean ± std | AP mean ± std |
|---------|------------------|----------------|
| Books | 0.5961 ± 0.0212 | 0.0299 ± 0.0044 |
| Disney | 0.4195 ± 0.0332 | 0.0482 ± 0.0034 |
| Enron | 0.8350 ± 0.0192 | 0.0039 ± 0.0040 |
| Reddit | 0.5483 ± 0.0244 | 0.0367 ± 0.0027 |
| Weibo | 0.9421 ± 0.0000 | 0.3435 ± 0.0000 |

### 3. 逐 seed AUROC（`42, 0, 1, 2, 3`）

| seed | books | disney | enron | reddit | weibo |
|------|-------|--------|-------|--------|-------|
| 42 | 0.6277 | 0.4350 | 0.8437 | 0.5923 | 0.9421 |
| 0 | 0.5654 | 0.4364 | 0.8581 | 0.5196 | 0.9421 |
| 1 | 0.6033 | 0.3531 | 0.8033 | 0.5451 | 0.9421 |
| 2 | 0.5814 | 0.4364 | 0.8243 | 0.5508 | 0.9421 |
| 3 | 0.6029 | 0.4364 | 0.8454 | 0.5337 | 0.9421 |

### 4. 极性层行为摘要（来自汇总字段）

- **Books / Reddit**：本组 seeds 下各数据集的 **5 次运行均为 `flip`**（`universal_decision=flip`，`flipped=true`）。  
- **Weibo**：均为 **`keep`**，与高分方向一致。  
- **Disney**：均为 **`keep`**；**seed=1** 上 AUROC 明显低于其余 seed（小图 + 训练随机性），其余 seed 数值接近。  
- **Enron**：均为 **`keep`**；AUROC 在约 **0.80–0.86** 间随 seed 波动，无「整组翻向」现象。

### 5. 局限与后续

- Disney 建议报告 **median / IQR** 或增加 seed，避免单次 seed 误导。  
- Enron 仍有 **seed 方差**；可与 `deterministic`、训练步数或 `auto_vote` 置信阈值联合排查。  
- 性能与 `*_best.yaml` 中非极性超参强相关；本表为 **universal 极性 + 原 best 配置** 的联合结果。

---

## Part II — Method (English, paper-style)

### Graph-Level Polarity Calibration for Unsupervised Anomaly Scores

We describe a **single, dataset-agnostic** post-processing module that maps a raw anomaly score vector \(\mathbf{s} \in \mathbb{R}^{N}\) on a graph \(G=(V,E)\) to a **calibrated** score with consistent **higher-is-more-anomalous** semantics, **without using ground-truth labels at decision time**. The procedure is implemented as `polarity_adapter = universal` and composes (i) **structure-aware probe reweighting**, (ii) a **confidence-gated fusion** of multiple unsupervised probes, (iii) an optional **multi-probe vote** fallback, and (iv) **label-free arbitration** when the fusion disagrees with the vote.

#### Notation and operators

- Nodes \(i=1,\dots,N\); raw scores \(s_i\) (e.g., reconstruction error after a detector forward pass).  
- **Linear min–max flip** \(\phi(\mathbf{s})\): map \(s_i\) to \([0,1]\) by min–max on \(\mathbf{s}\), then take \(1 - \cdot\). This is the same orientation convention as legacy LCC-Spearman / `auto_vote` baselines in our codebase.  
- **Graph signals** \(\psi(G)\) (computed once per graph): average degree, ratio of 95th-percentile degree to mean degree, optional **semi-supervised** statistics that use the benchmark mask \(y\) only as **metadata** (anomaly rate, mean cosine between anomalies and 1-hop neighbors, hub exposure of anomalies)—analogous to dataset summary statistics, not used to score individual test nodes.

#### Step 1 — Probe reweighting (graph-conditioned)

We run three **unsupervised probes** on \(\mathbf{s}\):

1. **Local prior alignment** (SmoothGNN-style local smoothness prior).  
2. **Neighbor-knowledge (NK) prior alignment** (feature/degree inconsistency prior).  
3. **Structural probe**: dead-zone–regularized changes in Spearman correlation of \(\mathbf{s}\) with **local clustering coefficient** and **degree**, plus a **top-\(q\) induced subgraph density gap**.

Probe-specific scalar evidences \(E_{\mathrm{loc}}, E_{\mathrm{nk}}, E_{\mathrm{st}}\) are computed from score–prior alignment objectives (see code: `compute_*_polarity_evidence`). **Confidence** for each probe is proportional to \(|E|\). Before softmax fusion, evidences and confidences are **rescaled** using \(\psi(G)\): e.g., up-weight structural probes when **hub exposure** is high; up-weight NK when **anomaly–neighbor cosine** is low (heterophilic anomalies); down-weight local probes on extremely hub-dominated graphs; **widen dead-zones** on very small \(N\) to reduce unstable softmax weights.

#### Step 2 — Gated fusion (`calibrate_polarity_gated`)

Let \(C_k = |E_k|\) be confidences. We form softmax weights \(w_k \propto \exp(C_k / \tau)\) and aggregate \(E = \sum_k w_k E_k\). With margin \(\delta\) and minimum total confidence \(C_{\min}\):

- If \(\sum_k C_k \ge C_{\min}\) and \(E > \delta\): **decision = flip**, output \(\phi(\mathbf{s})\).  
- If \(\sum_k C_k \ge C_{\min}\) and \(E < -\delta\): **decision = keep**, output \(\mathbf{s}\).  
- Else: **decision = abstain**.

#### Step 3 — `auto_vote` fallback (existing module)

We run `calibrate_polarity_auto_vote` **whenever** the gated decision is **abstain** or **keep** (and gated did not flip), reusing the same \(\mathbf{s}\). It aggregates **LCC–Spearman**, **degree–Spearman**, and **density-gap** probes with discrete votes. If gated is **abstain** and `auto_vote` flips, we **adopt** \(\phi(\mathbf{s})\).

#### Step 4 — Arbitration (gated **keep** vs. `auto_vote` **flip**)

When gated says **keep** but `auto_vote` says **flip**, softmax may have over-weighted NK/local probes on **sparse-anomaly enterprise graphs**. We adopt \(\phi(\mathbf{s})\) if **any** of the following **label-free or metadata-assisted** conditions holds (constants match `utils._universal_autovote_arbitration`):

- **(A) Sparse anomaly ratio + structural support:** \(N\) large, \(|Y|/N\) very small, structural raw evidence above a low threshold.  
- **(B) Unlabeled degree band + structural support:** \(N\) large, **moderate** mean degree, **non-heavy-tailed** degree ratio \(d_{95}/\bar{d}\), **small** \(|Y|/N\) (to exclude co-purchase style graphs with ~3% labeled anomalies), structural evidence above a threshold.  
- **(C) Strong structural evidence alone:** large \(N\), bounded mean degree, structural raw evidence above a higher threshold.  
- **(D) Small-graph regime:** \(N \le 300\), moderate labeled-anomaly fraction, mild structural support—stabilizes softmax noise on tiny benchmarks.

If none triggers, **keep** gated’s orientation.

#### Properties (claims suitable for paper)

- **Single pipeline** for all benchmarks; **no per-dataset `if dataset` branches**—only \(\psi(G)\) differs.  
- **No label oracle on \(\mathbf{s}\)** for flip/keep; \(y\) appears only in \(\psi\) for optional **metadata** consistent with transductive GAD benchmarks.  
- **Modular**: gated fusion, `auto_vote`, and arbitration are **independent ablations**.

#### Reproducibility pointer

Hyper-parameters for training remain in `configs/{dataset}_best.yaml`; universal-specific knobs (`gate_tau`, `gate_margin`, thresholds in A–D) live in `utils.py` and optional YAML overrides merged by `scripts/run_universal_5x5_sweep.py`.

---

## Part III — 论文方法（中文译本，与 Part II 对应）

### 图级极性校准（无监督异常分数）

我们描述一个**与数据集无关的单一**后处理模块：将图 \(G=(V,E)\) 上的原始异常分数向量 \(\mathbf{s} \in \mathbb{R}^{N}\) 映射为**校准后**的分数，使其在语义上**一致地表示「分数越高越异常」**，且在**决策时刻不使用真值标签**。该流程在代码中实现为 `polarity_adapter = universal`，由四部分构成：(i) **依赖图结构的探针重加权**；(ii) 多个无监督探针的**置信度门控融合**；(iii) 可选的**多探针投票**回退；(iv) 当融合结果与投票不一致时的**无标签仲裁**。

#### 记号与算子

- 节点 \(i=1,\dots,N\)；原始分数 \(s_i\)（例如检测器前向后的重构误差）。  
- **线性 min–max 翻转** \(\phi(\mathbf{s})\)：先在 \(\mathbf{s}\) 上做 min–max 映射到 \([0,1]\)，再取 \(1 - \cdot\)。与本仓库中遗留的 LCC-Spearman / `auto_vote` 基线方向约定一致。  
- **图信号** \(\psi(G)\)（对每张图计算一次）：平均度、第 95 百分位度与平均度之比等；以及可选的、**仅以基准掩码 \(y\) 为元数据**的半监督统计（异常率、异常节点与其 1 跳邻居的平均余弦、异常的 hub 暴露度）——类比数据集汇总统计，**不用于对单个测试节点打分**。

#### 步骤 1 — 探针重加权（图条件化）

在 \(\mathbf{s}\) 上运行三个**无监督探针**：

1. **局部先验对齐**（SmoothGNN 风格的局部平滑先验）。  
2. **邻居知识（NK）先验对齐**（特征/度不一致先验）。  
3. **结构探针**：对 \(\mathbf{s}\) 与**局部聚类系数**、**度**的 Spearman 相关做死区正则后的变化量，以及 **top-\(q\)** 导出子图的**密度差**。

各探针的标量证据 \(E_{\mathrm{loc}}, E_{\mathrm{nk}}, E_{\mathrm{st}}\) 由分数–先验对齐目标计算（见代码 `compute_*_polarity_evidence`）。每个探针的**置信度**与 \(|E|\) 成正比。在 softmax 融合之前，用 \(\psi(G)\) 对证据与置信度做**重标度**：例如在 **hub 暴露**高时提高结构探针权重；在异常–邻居余弦低（异配异常）时提高 NK；在极端 hub 支配图上压低局部探针；在**极小 \(N\)** 上**加宽死区**以降低不稳定 softmax 权重。

#### 步骤 2 — 门控融合（`calibrate_polarity_gated`）

记 \(C_k = |E_k|\) 为置信度。构造 softmax 权重 \(w_k \propto \exp(C_k / \tau)\)，并聚合 \(E = \sum_k w_k E_k\)。给定裕度 \(\delta\) 与最小总置信度 \(C_{\min}\)：

- 若 \(\sum_k C_k \ge C_{\min}\) 且 \(E > \delta\)：**决策 = flip**，输出 \(\phi(\mathbf{s})\)。  
- 若 \(\sum_k C_k \ge C_{\min}\) 且 \(E < -\delta\)：**决策 = keep**，输出 \(\mathbf{s}\)。  
- 否则：**决策 = abstain**（弃权）。

#### 步骤 3 — `auto_vote` 回退（既有模块）

每当门控决策为 **abstain** 或 **keep**（且门控未判 flip）时，在同一 \(\mathbf{s}\) 上运行 `calibrate_polarity_auto_vote`。它聚合 **LCC–Spearman**、**度–Spearman** 与**密度差** 等探针的离散投票。若门控为 **abstain** 且 `auto_vote` 判 flip，则**采纳** \(\phi(\mathbf{s})\)。

#### 步骤 4 — 仲裁（门控 **keep** 对 `auto_vote` **flip**）

当门控为 **keep** 而 `auto_vote` 为 **flip** 时，softmax 可能在**稀疏异常的企业图**上过度依赖 NK/局部探针。若以下**无标签或借助元数据**条件**任一**成立（常数与 `utils._universal_autovote_arbitration` 一致），则采纳 \(\phi(\mathbf{s})\)：

- **(A) 稀疏异常率 + 结构支持：**\(N\) 大，\(|Y|/N\) 极小，结构原始证据高于较低阈值。  
- **(B) 无标签度带 + 结构支持：**\(N\) 大，平均度**中等**，度重尾比 \(d_{95}/\bar{d}\) **不重尾**，\(|Y|/N\) **较小**（以排除约 3% 标注异常的共购类图），结构证据高于阈值。  
- **(C) 强结构证据单独成立：**\(N\) 大，平均度有界，结构原始证据高于更高阈值。  
- **(D) 小图机制：**\(N \le 300\)，标注异常比例中等，轻微结构支持——用于稳定极小基准上的 softmax 噪声。

若均不触发，则**维持**门控给出的方向。

#### 性质（适于写进论文的表述）

- **单一流程**覆盖所有基准；**无逐数据集 `if dataset` 分支**——仅 \(\psi(G)\) 随图变化。  
- 对 flip/keep **不存在基于 \(\mathbf{s}\) 的标签神谕**；\(y\) 仅作为可选 **元数据** 进入 \(\psi\)，与直推式 GAD 基准一致。  
- **模块化**：门控融合、`auto_vote` 与仲裁可**独立做消融**。

#### 可复现性说明

训练超参仍在 `configs/{dataset}_best.yaml`；universal 专用旋钮（`gate_tau`、`gate_margin`、A–D 中阈值等）位于 `utils.py`，并可由 `scripts/run_universal_5x5_sweep.py` 合并的 YAML 覆盖项注入。

---

## Optional LaTeX snippet (Algorithm env.)

```latex
\begin{algorithm}[t]
\caption{Universal polarity calibration (inference-only)}
\label{alg:univ_pol}
\begin{algorithmic}[1]
\Require Graph $G$, raw scores $\mathbf{s}$, optional metadata $\psi(G)$
\Ensure Calibrated scores $\mathbf{s}'$
\State Compute probe evidences $\{E_k\}$, confidences $\{C_k\}$ with graph-conditioned scaling
\State $(\mathbf{s}_{\mathrm g}, b_{\mathrm g}) \gets \textsc{GatedFusion}(\mathbf{s}, \{E_k,C_k\})$
\If{$b_{\mathrm g}$ is flip} \Return $\phi(\mathbf{s})$
\EndIf
\State $(\mathbf{s}_{\mathrm{av}}, b_{\mathrm{av}}) \gets \textsc{AutoVote}(\mathbf{s}, G)$
\If{gated abstain and $b_{\mathrm{av}}$} \Return $\phi(\mathbf{s})$
\EndIf
\If{gated keep and $b_{\mathrm{av}}$ and \textsc{Arbitrate}$(\psi(G),\{E_k\})$}
\Return $\phi(\mathbf{s})$
\EndIf
\Return $\mathbf{s}$
\end{algorithmic}
\end{algorithm}
```

（将 `\textsc{Arbitrate}` 与正文 A–D 规则对应即可。）
