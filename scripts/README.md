# Scripts

## Reviewer-facing

| Script | Purpose |
|--------|---------|
| `run_single.py` | One dataset, one seed → trains and evaluates FMGAD. |
| `run_all_5seeds.py` | Parallel driver for five datasets × multiple seeds; writes `results/main_runs/*.json`. |
| `aggregate_results.py` | Collates run JSON files into CSV tables. |
| `run_ablation.py` | Paper ablation variants (optional). |

## Development (not needed for main tables)

See `scripts/dev/` — tuning sweeps, smoke tests, plotting, and runtime comparison helpers.
