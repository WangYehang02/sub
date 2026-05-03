# FMGAD: label usage and evaluation protocol

This document summarizes how **ground-truth anomaly labels** are used in the released code.

## Training, inference, and score calibration

- The model is trained as an **unsupervised** reconstruction / flow-matching objective on node features and graph structure.
- The **polarity adapter** supported for reported runs is `universal_no_y` (strict label-free graph/attribute proxy signals) or `none` for ablations. **Labels are not read** for checkpoint selection, timestep selection, polarity decisions, or adapter selection.
- Labels **do not** enter the loss used for optimization.

## Where labels appear

- **Evaluation only:** after scores are finalized, metrics such as AUROC and Average Precision are computed with `eval_*` functions that require a binary label vector (standard unsupervised anomaly detection evaluation on benchmark datasets with held-out labels).

## Hyperparameters

- Per-dataset YAML files under `configs/` (`books.yaml`, `disney.yaml`, …) contain **fixed** hyperparameters used for the five-seed evaluation protocol in the paper supplement. They are **not** tuned by automated search inside this artifact.

## Ablation scripts

- `scripts/run_ablation.py` implements paper ablations by overriding a small set of YAML fields; it does not use labels for decisions, only for the same final metric computation as `main_train.py`.
