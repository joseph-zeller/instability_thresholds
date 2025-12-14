#!/usr/bin/env python3
"""
Run all Supplementary Information (SI) figure scripts.

Assumes SI figure scripts are located in:
  src/figures/si/

Outputs are written by each script to:
  figures/si/

Usage (from repo root):
  python src/figures/run_all_si_figures.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    si_dir = repo_root / "src" / "figures" / "si"

    # EDIT THIS LIST ONLY if filenames change
    scripts = [
        "figS1_pca_biplot.py",
        "figS2_window_robustness.py",
        "figS3_transformation_robustness.py",
        "figS4_threshold_distribution.py",
        "figS5_population_stratification.py",
        "figS6_crosspredictor_alignment.py",
    ]

    print("Running SI figure scripts...\n")

    for s in scripts:
        script_path = si_dir / s
        if not script_path.exists():
            raise FileNotFoundError(f"Missing SI figure script: {script_path}")

        print(f"→ {s}")
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=repo_root,
            check=False
        )

        if result.returncode != 0:
            raise RuntimeError(f"SI figure script failed: {s}")

    print("\n[OK] All SI figures generated successfully.")


if __name__ == "__main__":
    main()
