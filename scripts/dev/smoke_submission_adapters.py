#!/usr/bin/env python3
"""Smoke: five datasets × polarity_adapter universal_no_y | none, seed=42."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

DATASETS = ("books", "disney", "enron", "reddit", "weibo")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--num-trial", type=int, default=1)
    args = ap.parse_args()
    root: Path = args.repo_root
    cfg_dir = root / "configs"
    ok = True
    for d in DATASETS:
        base = cfg_dir / f"{d}.yaml"
        if not base.exists():
            print(f"[FAIL] missing {base}", flush=True)
            ok = False
            continue
        for adapter in ("universal_no_y", "none"):
            with open(base, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            cfg["polarity_adapter"] = adapter
            tmp = cfg_dir / f"_smoke_{adapter}_{d}.yaml"
            with open(tmp, "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)
            cmd = [
                sys.executable,
                str(root / "main_train.py"),
                "--config",
                str(tmp),
                "--seed",
                str(args.seed),
                "--num_trial",
                str(args.num_trial),
                "--device",
                str(args.device),
            ]
            print(f"[RUN] {' '.join(cmd)}", flush=True)
            r = subprocess.run(cmd, cwd=str(root))
            if r.returncode != 0:
                print(f"[FAIL] dataset={d} adapter={adapter} rc={r.returncode}", flush=True)
                ok = False
            else:
                print(f"[OK] dataset={d} adapter={adapter}", flush=True)
            try:
                tmp.unlink()
            except OSError:
                pass
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
