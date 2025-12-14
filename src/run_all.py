#!/usr/bin/env python
"""
run_all.py

Combined reproducibility pipeline for the collapse_thresholds project.

This script serves as the single entry point for reviewers.
It orchestrates the canonical analysis steps used in the manuscript.

Stages:
  1) Threshold estimation
  2) Table generation
  3) Figure generation
  4) Robustness analyses
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def run_script(script_path: Path) -> None:
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    cmd = [sys.executable, str(script_path)]
    print("\n" + "=" * 72)
    print(f"Running: {' '.join(cmd)}")
    print("=" * 72 + "\n")

    subprocess.check_call(cmd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combined reproducibility pipeline for collapse_thresholds."
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=["thresholds", "tables", "figures", "robustness"],
        help="Limit execution to selected pipeline stages.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    stages: List[Tuple[str, Path]] = [
        ("thresholds", SRC / "compute_thresholds.py"),
        ("tables", SRC / "tables" / "make_all_tables.py"),
        ("figures", SRC / "figures" / "make_all_figures.py"),
        ("robustness", SRC / "robustness" / "bootstrap_spc1_threshold_w100.py"),
        ("robustness", SRC / "robustness" / "permutation_threshold_alignment.py"),
    ]

    if args.only is None:
        stages_to_run = {name for name, _ in stages}
    else:
        stages_to_run = set(args.only)

    print(f"Project root: {ROOT}")
    print(f"Stages to run: {', '.join(sorted(stages_to_run))}")

    for stage_name, script in stages:
        if stage_name not in stages_to_run:
            continue
        run_script(script)

    print("\nAll requested stages completed successfully.")


if __name__ == "__main__":
    main()
