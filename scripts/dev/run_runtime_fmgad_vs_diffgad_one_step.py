#!/usr/bin/env python3
"""
Sequential runtime: FMGAD (texing/FMGADfinal) vs DiffGAD, seed=42, 3 repeats each,
same GPU, wall-clock via /usr/bin/time (stderr WALL_TIME_SEC=...).
"""
from __future__ import annotations

import csv
import json
import os
import shlex
import statistics
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

FMGAD_ROOT = Path(__file__).resolve().parents[2]
DIFFGAD_ROOT = Path("/home/yehang/time0501/DiffGAD")
OUT_ROOT = Path("/mnt/yehang/FMGADfinal_runs/runtime_fmgad_vs_diffgad_one_step")
PYTHON = Path(os.environ.get("FMGAD_PYTHON", "/home/yehang/miniconda3/envs/fmgad/bin/python"))

GPU_ID = "0"
SEED = 42
RUN_IDS = [1, 2, 3]
DATASETS = ["books", "disney", "enron", "reddit", "weibo"]


@dataclass
class RunSpec:
    method: str
    dataset: str
    run_id: int
    cwd: Path
    command: str


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def run_cmd_capture(cmd: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        shell=True,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def get_gpu_name(gpu_id: str) -> str:
    cp = run_cmd_capture(
        f"nvidia-smi --query-gpu=name --format=csv,noheader,nounits -i {shlex.quote(gpu_id)}",
        cwd=FMGAD_ROOT,
    )
    if cp.returncode != 0:
        return "UNKNOWN_GPU"
    return cp.stdout.strip().splitlines()[0].strip()


def parse_wall_time(stderr_text: str) -> float:
    for line in stderr_text.splitlines():
        if line.startswith("WALL_TIME_SEC="):
            return float(line.split("=", 1)[1].strip())
    raise ValueError("WALL_TIME_SEC not found in stderr")


def build_specs() -> List[RunSpec]:
    specs: List[RunSpec] = []
    py = shlex.quote(str(PYTHON))
    for method in ["fmgad", "diffgad"]:
        for dataset in DATASETS:
            for run_id in RUN_IDS:
                cfg = f"configs/{dataset}.yaml"
                if method == "fmgad":
                    cmd = (
                        f"CUDA_VISIBLE_DEVICES={GPU_ID} "
                        f"/usr/bin/time -f \"WALL_TIME_SEC=%e\" "
                        f"{py} main_train.py "
                        f"--device 0 --config {cfg} --seed {SEED}"
                    )
                    cwd = FMGAD_ROOT
                else:
                    cmd = (
                        f"CUDA_VISIBLE_DEVICES={GPU_ID} "
                        f"/usr/bin/time -f \"WALL_TIME_SEC=%e\" "
                        f"{py} main.py "
                        f"--device 0 --config {cfg} --seed {SEED}"
                    )
                    cwd = DIFFGAD_ROOT
                specs.append(RunSpec(method=method, dataset=dataset, run_id=run_id, cwd=cwd, command=cmd))
    return specs


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def summarize(all_payloads: List[Dict], out_root: Path) -> None:
    rows = []
    grouped: Dict[str, Dict[str, List[Dict]]] = {"fmgad": {}, "diffgad": {}}
    for m in grouped:
        for d in DATASETS:
            grouped[m][d] = []
    for p in all_payloads:
        grouped[p["method"]][p["dataset"]].append(p)

    for method in ["fmgad", "diffgad"]:
        for dataset in DATASETS:
            runs = sorted(grouped[method][dataset], key=lambda x: x["run_id"])
            times = [r["wall_time_sec"] for r in runs if r["return_code"] == 0 and r["wall_time_sec"] is not None]
            if len(times) != 3:
                continue
            mean_t = statistics.mean(times)
            std_t = statistics.stdev(times)
            rows.append(
                {
                    "method": method,
                    "dataset": dataset,
                    "seed": SEED,
                    "num_runs": 3,
                    "wall_time_mean_sec": mean_t,
                    "wall_time_std_sec": std_t,
                    "all_wall_times_sec": ",".join(f"{x:.4f}" for x in times),
                    "gpu_name": runs[0]["gpu_name"],
                    "command": runs[0]["command"],
                }
            )

    csv_path = out_root / "runtime_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "dataset",
                "seed",
                "num_runs",
                "wall_time_mean_sec",
                "wall_time_std_sec",
                "all_wall_times_sec",
                "gpu_name",
                "command",
            ],
        )
        writer.writeheader()
        for r in rows:
            rr = r.copy()
            rr["wall_time_mean_sec"] = f"{rr['wall_time_mean_sec']:.4f}"
            rr["wall_time_std_sec"] = f"{rr['wall_time_std_sec']:.4f}"
            writer.writerow(rr)

    stat = {(r["method"], r["dataset"]): r for r in rows}
    latex_path = out_root / "runtime_latex.tex"
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write(
            "\\caption{Wall-clock runtime (FMGAD fixed one-step configs vs DiffGAD). "
            "Seed 42; three sequential repeats per setting on one GPU; "
            "/usr/bin/time outer measurement includes load, train, infer, eval.}\n"
        )
        f.write("\\label{tab:runtime_fmgad_diffgad_onestep}\n")
        f.write("\\small\n")
        f.write("\\resizebox{\\linewidth}{!}{%\n")
        f.write("\\begin{tabular}{lcccccc}\n")
        f.write("\\toprule\n")
        f.write("Method & Books & Disney & Enron & Reddit & Weibo & Avg \\\\\n")
        f.write("\\midrule\n")
        for method_name, method_key in [("FMGAD", "fmgad"), ("DiffGAD", "diffgad")]:
            means = []
            cells = []
            for d in DATASETS:
                r = stat[(method_key, d)]
                m = float(r["wall_time_mean_sec"])
                s = float(r["wall_time_std_sec"])
                means.append(m)
                cells.append(f"${m:.2f} \\\\pm {s:.2f}$")
            avg = statistics.mean(means)
            f.write(f"{method_name} & " + " & ".join(cells) + f" & ${avg:.2f}$ \\\\\n")
        ratio_cells = []
        ratio_means = []
        for d in DATASETS:
            rf = float(stat[("fmgad", d)]["wall_time_mean_sec"])
            rd = float(stat[("diffgad", d)]["wall_time_mean_sec"])
            ratio = rd / rf if rf > 0 else float("nan")
            ratio_means.append(ratio)
            ratio_cells.append(f"${ratio:.2f}\\times$")
        ratio_avg = statistics.mean(ratio_means)
        f.write("DiffGAD/FMGAD & " + " & ".join(ratio_cells) + f" & ${ratio_avg:.2f}\\times$ \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}%\n")
        f.write("}\n")
        f.write("\\end{table}\n")

    print(f"Wrote {csv_path}")
    print(f"Wrote {latex_path}")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=str, default=str(OUT_ROOT))
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Skip specs whose JSON already exists with return_code==0 and wall_time_sec set.",
    )
    args = ap.parse_args()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    logs_dir = out_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    gpu_name = get_gpu_name(GPU_ID)
    all_payloads: List[Dict] = []
    errors_log = out_root / "errors.log"
    if errors_log.exists():
        errors_log.unlink()

    specs = build_specs()
    env_base = os.environ.copy()
    env_base["FMGAD_MODEL_ROOT"] = env_base.get("FMGAD_MODEL_ROOT", "/mnt/yehang/FMGADfinal_runs/models")

    for i, spec in enumerate(specs, 1):
        out_stem = f"{spec.method}_{spec.dataset}_seed{SEED}_run{spec.run_id}"
        stdout_path = logs_dir / f"{out_stem}.stdout.log"
        stderr_path = logs_dir / f"{out_stem}.stderr.log"
        json_path = out_root / f"{out_stem}.json"

        if args.resume and json_path.is_file():
            try:
                prev = json.loads(json_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                prev = None
            if (
                isinstance(prev, dict)
                and int(prev.get("return_code", -1)) == 0
                and prev.get("wall_time_sec") is not None
            ):
                all_payloads.append(prev)
                print(f"[{i}/{len(specs)}] {out_stem} SKIP(resume) wall={prev.get('wall_time_sec')}")
                continue

        env_run = env_base.copy()
        env_run["CUDA_VISIBLE_DEVICES"] = GPU_ID
        env_run["FMGAD_RUN_TAG_SUFFIX"] = f"runtime_{spec.method}_{spec.dataset}_r{spec.run_id}"

        before_cp = run_cmd_capture(
            f"nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv,noheader -i {GPU_ID}",
            cwd=spec.cwd,
        )
        start_time = now_iso()
        with open(stdout_path, "w", encoding="utf-8") as fo, open(stderr_path, "w", encoding="utf-8") as fe:
            cp = subprocess.run(
                spec.command,
                shell=True,
                cwd=str(spec.cwd),
                stdout=fo,
                stderr=fe,
                env=env_run,
                check=False,
            )
        end_time = now_iso()
        after_cp = run_cmd_capture(
            f"nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv,noheader -i {GPU_ID}",
            cwd=spec.cwd,
        )

        stderr_text = stderr_path.read_text(encoding="utf-8", errors="ignore")
        wall_time_sec = None
        notes = ""
        if cp.returncode == 0:
            try:
                wall_time_sec = parse_wall_time(stderr_text)
            except Exception as e:
                notes = f"failed to parse wall time: {repr(e)}"
                cp = subprocess.CompletedProcess(args=[], returncode=2)
        else:
            notes = "command failed, see stderr log"

        payload = {
            "method": spec.method,
            "dataset": spec.dataset,
            "seed": SEED,
            "run_id": spec.run_id,
            "command": spec.command,
            "cuda_visible_devices": GPU_ID,
            "gpu_name": gpu_name,
            "wall_time_sec": wall_time_sec,
            "start_time": start_time,
            "end_time": end_time,
            "return_code": cp.returncode,
            "nvidia_smi_before": before_cp.stdout.strip(),
            "nvidia_smi_after": after_cp.stdout.strip(),
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "notes": notes,
        }
        write_json(json_path, payload)
        all_payloads.append(payload)

        if cp.returncode != 0:
            with open(errors_log, "a", encoding="utf-8") as f:
                f.write(
                    f"[FAIL] method={spec.method} dataset={spec.dataset} run_id={spec.run_id} return_code={cp.returncode}\n"
                )
        print(f"[{i}/{len(specs)}] {out_stem} rc={cp.returncode} wall={wall_time_sec}")

    ok = True
    for method in ["fmgad", "diffgad"]:
        for d in DATASETS:
            hit = [p for p in all_payloads if p["method"] == method and p["dataset"] == d and p["return_code"] == 0]
            if len(hit) != 3:
                ok = False
    if ok:
        summarize(all_payloads, out_root)
        print(f"Summary written to {out_root}")
    else:
        print("Incomplete runs detected; summary not generated.")


if __name__ == "__main__":
    main()
