#!/usr/bin/env python3
"""
Run all manuscript figure scripts (Figures 1–5).

Assumes scripts are located in:
  src/figures/manuscript/

Outputs are written by each script to:
  figures/manuscript/

Usage (from repo root):
  python src/figures/run_all_manuscript_figures.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    figs_dir = repo_root / "src" / "figures" / "manuscript"

    scripts = [
        "fig1a_spc1_distribution.py",
        "fig1b_seshat_eta_distribution.py",
        "fig2_logistic_overload_curves.py",
        "fig3_percentile_convergence.py",
        "fig4_threshold_stability_overload_measures.py",
        "fig5_conceptual_overload_mechanism.py",
    ]

    print("Running manuscript figure scripts...\n")

    for s in scripts:
        script_path = figs_dir / s
        if not script_path.exists():
            raise FileNotFoundError(f"Missing script: {script_path}")

        print(f"→ {s}")
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=repo_root,
            check=False
        )

        if result.returncode != 0:
            raise RuntimeError(f"Script failed: {s}")

    print("\n[OK] All manuscript figures generated successfully.")


if __name__ == "__main__":
    main()
