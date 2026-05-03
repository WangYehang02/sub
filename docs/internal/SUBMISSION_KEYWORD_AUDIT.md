# Submission keyword audit (post clean)

Generated for `WangYehang02/sub` submission-clean pass. Patterns searched under repository root (excluding large binary trees; primary hits in `.py`, `.yaml`, `.md`, `.tex`).

## 1. `trial`

**Hits (representative):** `num_trial` / training loop “trials” in `model.py`, `main_train.py`; dev tooling `scripts/dev/tune_disney_nontune.py`, `run_universal_param_tune.py`; internal docs `docs/internal/SUBMISSION_RG_AUDIT.md`.

**Note:** No remaining **Disney YAML** line referencing “trial 17” or tune IDs after `configs/disney.yaml` comment update.

## 2. `mean AUROC`

**Hits:** `scripts/dev/tune_disney_nontune.py` (dev tuning), `scripts/dev/run_universal_param_tune.py`, `scripts/dev/summarize_fmgad_ablation.py` (LaTeX caption), `docs/internal/POLARITY_LABEL_FREE_AUDIT.md` (“macro mean AUROC” table header).

**Configs:** none.

## 3. `tune_`

**Hits:** `scripts/dev/tune_disney_nontune.py`, `scripts/dev/run_universal_param_tune.py` (`univ_tune_…` exp tags).

**Configs / `model.py` / `main_train.py`:** none.

## 4. `best auc` (case-insensitive)

**Hits:** none in tracked sources scanned.

## 5. `data.y`

**Python (training path):**

| File | Role |
|------|------|
| `model.py` | `y_eval = data.y` / `data.y.bool()` **after** `_apply_score_polarity_adapter` / score path; used only for `eval_*`, PR, F1, optional `FMGAD_POLARITY_DEBUG` prints. |
| `scripts/dev/dataset_graph_stats.py` | Offline stats only. |

**Docs:** `docs/internal/POLARITY_LABEL_FREE_AUDIT.md`, `docs/internal/SUBMISSION_RG_AUDIT.md`, `scripts/dev/summarize_fmgad_ablation.py` (audit text).

## 6. `y_true`

**Python:** no identifier `y_true` in `.py` sources (per prior audit; metrics use other local names).

**Docs:** mentioned in `docs/internal/POLARITY_LABEL_FREE_AUDIT.md` / `SUBMISSION_RG_AUDIT.md` as “not used as global name”.

## 7. `polarity_adapter: nk` / `polarity_adapter: auto_vote`

**Hits:** none in configs or code string literals matching these adapter names.

## 8. `polarity_adapter: universal`

**Hits:** `polarity_adapter: universal_no_y` in all five `configs/*.yaml` and `universal_template.yaml`; docs references in `POLARITY_LABEL_FREE_AUDIT.md`, `summarize_fmgad_ablation.py`.  
**Legacy doc line** in `docs/internal/UNIVERSAL_POLARITY_PAPER_EVALUATION.md` updated to `universal_no_y` for consistency.

---

## `data.y` usage conclusion (requirement 6)

- **Training / AE / flow / gate:** no `data.y`.
- **Inference `sample()`:** scores and `_apply_score_polarity_adapter` do **not** use `data.y`; `y_eval` is read **only after** final scores for metrics (and optional debug when `FMGAD_POLARITY_DEBUG=1`).
- **Polarity determination (`universal_no_y`):** uses `edge_index`, `x`, and precomputed unsupervised graph signals — not labels.
- **Offline tooling:** `scripts/dev/dataset_graph_stats.py` reads labels for statistics only.
