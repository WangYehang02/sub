#!/usr/bin/env python3
"""Compatibility entrypoint for rule1 training.

Behavior is intentionally preserved by delegating to the original script.
"""

from pathlib import Path
from runpy import run_path

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    run_path(str(root / "rule1_code" / "main_train.py"), run_name="__main__")
