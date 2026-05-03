# Repository Reorganization Proposal

## 1) Separate library code and experiment scripts

Current: `rule1_code/` and `rule2_code/` both mix model definitions, training entrypoints, and utilities.

Target:

- `src/rulegad/models/`
- `src/rulegad/data/`
- `src/rulegad/train/`
- `scripts/` for experiment launchers only

## 2) Standardize configuration system

Current configs are under multiple subfolders and script-specific naming.

Target:

- single `configs/` with grouped folders: `dataset/`, `model/`, `train/`, `ablation/`
- reproducible run IDs should be auto-generated and logged

## 3) Remove paper-specific ad-hoc scripts from core path

Keep scripts such as one-off report merging in `scripts/reports/` so `src/` remains reusable.

## 4) Add tests for deterministic utilities

At minimum:

- seed setup
- config parsing
- metric aggregation and report merging

## 5) Add packaging metadata

Add `pyproject.toml` and installable package name `rulegad` to support:

- `pip install -e .`
- `python -m rulegad.train.main --config configs/train/default.yaml`

## 6) Documentation expected by reviewers

- Problem definition and assumptions
- Dataset processing steps
- Exact training/eval commands for tables in paper
- Known limitations and failure modes
