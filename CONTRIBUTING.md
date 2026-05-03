# Contributing

## Development workflow

1. Create a branch from `main`.
2. Add or update tests for all behavior changes.
3. Run formatting/linting/tests before opening a PR.
4. Update docs (`README`, reproducibility notes) if scripts or configs change.

## Commit style

Use concise conventional style when possible, e.g.:

- `feat: add unified training entrypoint`
- `fix: stabilize seed handling in evaluator`
- `docs: update reproducibility instructions`

## Reproducibility requirements

For experiment-related changes, include in PR description:

- dataset and split
- seed(s)
- exact command
- expected metrics file path
