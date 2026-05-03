.PHONY: help check-structure test-wrapper-targets smoke

help:
	@echo "Targets:"
	@echo " make check-structure # verify non-breaking OSS scaffold files"
	@echo " make smoke # run lightweight wrapper/import smoke checks"

check-structure:
	test -f README.md
	test -f pyproject.toml
	test -f src/rulegad/__init__.py
	test -f scripts/rule1/train.py
	test -f scripts/rule2/train.py
	test -f scripts/rule2/eval_best.py
	test -f configs/README.md

test-wrapper-targets:
	python - <<'PY'
from pathlib import Path
root = Path('.')
checks = [
    (root / 'scripts/rule1/train.py', root / 'rule1_code/main_train.py'),
    (root / 'scripts/rule2/train.py', root / 'rule2_code/main_train.py'),
    (root / 'scripts/rule2/eval_best.py', root / 'rule2_code/run_best_eval.py'),
]
for wrapper, target in checks:
    if not wrapper.exists() or not target.exists():
        raise AssertionError(f"missing pair: {wrapper}, {target}")
print('wrapper target checks passed')
PY

smoke: check-structure test-wrapper-targets
	python -m py_compile scripts/rule1/train.py scripts/rule2/train.py scripts/rule2/eval_best.py src/rulegad/__init__.py
