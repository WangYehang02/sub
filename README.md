# FMGAD

PyTorch implementation of **FMGAD** (flow matching for graph anomaly detection) for the NeurIPS supplementary code release.

## What this repository contains

- **Training and evaluation** entrypoint: `main_train.py` (wrapper CLI: `scripts/run_single.py`).
- **Core model:** `model.py` (`ResFlowGAD`), with `auto_encoder.py`, `encoder.py`, `flow_matching_model.py`, `FMloss.py`, `utils.py`.
- **Fixed configs:** `configs/{books,disney,enron,reddit,weibo}.yaml` for the five PyGOD benchmarks used in the paper.

## Install

1. Install **PyTorch** for your CUDA/CPU setup: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
2. Install **PyTorch Geometric** matching your Torch/CUDA: [https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
3. Install Python dependencies:

```bash
pip install -r requirements.txt
```

Python **3.8+** is supported (CI-style testing used 3.8).

## Run (exact commands)

**One dataset, one seed**

```bash
python scripts/run_single.py --dataset books --seed 42 --device 0
```

**Five datasets × five seeds (parallel)**

```bash
python scripts/run_all_5seeds.py --datasets books,disney,enron,reddit,weibo --seeds 42,0,1,2,3 --gpus 0,1,2,3,4,5,6,7 --max-workers 8 --output-dir results/main_runs
```

**Aggregate metrics to CSV**

```bash
python scripts/aggregate_results.py --input results/main_runs --output results/main_table.csv
```

**Optional ablations**

```bash
python scripts/run_ablation.py --help
```

Full step-by-step notes: [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Label usage / fairness

See [`docs/FAIRNESS.md`](docs/FAIRNESS.md): labels are used **only for final evaluation metrics** (AUROC, AP, etc.), not for training objectives, polarity calibration, or adapter selection.

## Layout

| Path | Role |
|------|------|
| `configs/*.yaml` | Fixed per-dataset hyperparameters. |
| `scripts/run_single.py` | Recommended CLI for a single experiment. |
| `scripts/run_all_5seeds.py` | Batch driver for multi-seed tables. |
| `scripts/aggregate_results.py` | CSV aggregation. |
| `scripts/run_ablation.py` | Ablation protocol. |
| `scripts/dev/` | Non-essential development utilities (tuning, plots, smoke tests). |
| `docs/internal/` | Historical notes; **not** required for reproduction. |

## Citation

See `CITATION.cff`. BibTeX for the camera-ready paper will be added upon publication.

## License

See `LICENSE`.
