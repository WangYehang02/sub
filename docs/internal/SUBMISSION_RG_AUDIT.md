# Submission-clean 仓库检索审计（`data.y` / 指标 / 标签）

审计日期：2026-05-03。在**本仓库根目录**（历史版本中曾存在嵌套目录副本；结论与单根树扫描一致）对以下模式做了全文检索：

- `data\.y`
- `y_true`
- `\blabel\b`（Python 源码，结果多为 LaTeX `\label` 或注释中的 “label-free”）
- `eval_roc_auc`、`eval_average_precision`
- `max\(.*AUC`（大小写不敏感语义由人工核对）

## 1. `data.y` 出现位置（Python）

| 文件 | 用途 |
|------|------|
| `model.py` | `ensemble_score` 分支：在 `_apply_score_polarity_adapter` **之后** `y_eval = data.y`，仅用于 `eval_*`、PR 曲线与 F1。 |
| `model.py` | `sample()`：在极性与分数路径完成后 `y_eval = data.y.bool()`，用于 `eval_*`、调试分支 `FMGAD_POLARITY_DEBUG`、以及最终指标。 |
| `scripts/dev/dataset_graph_stats.py` | 离线统计脚本读取 `data.y`，不参与训练/推理主路径。 |

**结论**：训练、推理、分数校准与 `polarity_adapter` 选择路径中**不**读取 `data.y`；`data.y` 仅在上述 **metric computation**（及离线脚本）中出现。

## 2. `y_true`

Python 源码中**无** `y_true` 标识符（指标函数在 `utils` 等处以局部变量接收标签张量，但不使用该全局名）。

## 3. `eval_roc_auc` / `eval_average_precision`

仅出现在 `model.py`（导入与 `sample()` / `ensemble_score` 指标块）及可能的指标定义文件；均发生在分数与极性适配器执行**之后**。

## 4. `max(.*AUC` 类模式

- `model.py` 中 `torch.max(dm_auc)` 等为 **多 trial 聚合上的最大值**，非 `max(AUC, 1-AUC)` 式极性作弊。
- `scripts/run_ablation.py` 中相关字符串为 **静态禁止模式**（sanity），非运行时代码。

## 5. 主路径约束（与本次 submission-clean 一致）

- `polarity_adapter` 仅允许 `universal_no_y` 与 `none`；`universal` 在 `ResFlowGAD.__init__` 中 `ValueError`。
- `universal_no_y` 在 `forward` 中仅通过 `compute_polarity_graph_signals_unsup(edge_index, x, q=...)` 构建图信号，**不**传入 `data.y`。
- `calibrate_polarity_auto_vote` 仅由 `calibrate_polarity_universal` 内部作为 fallback 调用，不作为顶层 adapter。

## 6. Smoke 复现

在 `conda` 环境 `fmgad` 下执行：

```bash
python scripts/smoke_submission_adapters.py --device 0 --seed 42 --num-trial 1
```

2026-05-03 运行结果：五数据集 × (`universal_no_y` | `none`) 共 10 次，`exit_code: 0`。
