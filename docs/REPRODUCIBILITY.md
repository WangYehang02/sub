# Reproducibility (exact commands)

Environment: **Python 3.8+**, **CUDA** recommended. Install **PyTorch** for your platform first ([pytorch.org](https://pytorch.org/get-started/locally/)), then install PyTorch Geometric wheels matching your Torch/CUDA build ([pyg.org](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)), then:

```bash
cd /path/to/fmgad-repo
pip install -r requirements.txt
```

## Single dataset / single seed

```bash
python scripts/run_single.py --dataset books --seed 42 --device 0
```

Optional JSON output:

```bash
python scripts/run_single.py --dataset books --seed 42 --device 0 --result-file results/example_books_seed42.json
```

## Five datasets × five seeds (main table)

Uses `configs/{dataset}.yaml` and parallel workers (adjust GPUs to your machine):

```bash
python scripts/run_all_5seeds.py --datasets books,disney,enron,reddit,weibo --seeds 42,0,1,2,3 --gpus 0,1,2,3,4,5,6,7 --max-workers 8 --output-dir results/main_runs
```

## Aggregate CSV

```bash
python scripts/aggregate_results.py --input results/main_runs --output results/main_table.csv
```

This writes `results/main_table.csv` and `results/main_table_summary_by_dataset.csv` (mean ± sample stdev of AUROC/AP **across seeds** per dataset).

## Ablations (optional)

```bash
python scripts/run_ablation.py --help
```

## Determinism (optional)

```bash
python scripts/run_single.py --dataset enron --seed 42 --device 0 --deterministic
```
