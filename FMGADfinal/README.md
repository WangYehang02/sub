# FMGAD

Official PyTorch implementation accompanying our **NeurIPS** submission on **flow matching for graph anomaly detection**. This repository provides training, evaluation, and auxiliary scripts used to produce the empirical results reported in the paper.

## Method at a glance

FMGAD combines a graph autoencoder with a **flow-matching** generative model over latent representations, together with optional prototype conditioning and score calibration. The core implementation lives in `model.py` (`ResFlowGAD`), with supporting modules for the autoencoder, encoder, flow-matching dynamics, and losses.

## Scope of this release

The artifact targets **five standard PyGOD benchmarks**: `books`, `disney`, `enron`, `reddit`, and `weibo`. Data are obtained through **PyGOD** (`pygod.utils.load_data`) on first use; no separate preprocessing pipeline is required for these datasets.

## Requirements

See `requirements.txt` for pinned dependencies. In brief: Python 3.8+ (tested with 3.8), PyTorch with CUDA, **PyTorch Geometric**, **PyGOD**, and common scientific Python stack. Install into a clean environment before running experiments.

## Training and evaluation

From the repository root:

```bash
python main_train.py --config configs/<dataset>_best.yaml --device 0 --seed 42
```

Replace `<dataset>` with one of `books`, `disney`, `enron`, `reddit`, or `weibo`. Each `configs/*_best.yaml` file holds hyperparameters aligned with our reported runs. Optional flags include `--num_trial`, `--result-file`, `--deterministic`, and `--profile-efficiency`; see `main_train.py` for details.

Metrics (e.g., ROC-AUC, AP) and timing are printed to stdout; `--result-file` writes a JSON summary suitable for aggregation.

## Configuration layout

- `configs/` — per-dataset YAML files for FMGAD (`*_best.yaml`).
- `configs/README.md` — short index of configuration conventions.

## Supplementary scripts

Under `scripts/` you will find utilities used for **ablation studies**, **strict runtime comparisons**, and **figure generation** (e.g., speedup and ablation plots). Invoke each script with `--help` for usage; paths are relative to the repository root unless noted otherwise.

## Documentation and reproducibility

Additional materials for release hygiene and reproducibility checks live in `docs/` (e.g., command inventories and checklists). These files are provided to support reviewers and future open-source release; they do not alter training behavior.

## Citation

If this code is useful, please cite the accompanying paper (bibtex will be added upon publication). Until then, you may reference this repository using the metadata in `CITATION.cff`.

## License

See `LICENSE` for terms of use.
