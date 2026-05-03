#!/usr/bin/env python3
"""Compatibility entrypoint for best-config evaluation (rule2)."""

from pathlib import Path
from runpy import run_path

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    run_path(str(root / "rule2_code" / "run_best_eval.py"), run_name="__main__")
