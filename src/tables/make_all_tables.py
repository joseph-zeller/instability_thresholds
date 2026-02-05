"""
Run all table-generation scripts for the collapse_thresholds repository.

This orchestrator executes each make_table_*.py script in a fixed order,
allowing full regeneration of all manuscript and SI tables with one command:

    python src/tables/make_all_tables.py
"""

from __future__ import annotations

import sys
import time
import runpy
import pathlib


def main() -> None:
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    tables_dir = pathlib.Path(__file__).resolve().parent

    scripts = [
        "make_table_S1_pca_complexity.py",
        "make_table_S2_core_thresholds.py",
        "make_table_S3_eta_exclusions.py",
        "make_table_S4_SPC1_horizons.py",
        "make_table_S5_eta_horizons.py",
        "make_table_S6_population_strata.py",
        "make_table_S7_cross_predictor.py",
    ]

    print("[make_all_tables] Repo root:", repo_root)
    print("[make_all_tables] Tables directory:", tables_dir)
    print("[make_all_tables] Running scripts:")
    for s in scripts:
        print("  -", s)

    # Ensure repo root is on sys.path
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    t0 = time.time()

    for script in scripts:
        path = tables_dir / script
        if not path.exists():
            raise FileNotFoundError(f"Missing table script: {path}")

        print("\n" + "=" * 72)
        print(f"[make_all_tables] Running {path.relative_to(repo_root)}")
        print("=" * 72)

        runpy.run_path(str(path), run_name="__main__")

    elapsed = time.time() - t0
    print("\n" + "-" * 72)
    print(f"[make_all_tables] Completed {len(scripts)} tables in {elapsed:.1f}s")
    print("-" * 72)


if __name__ == "__main__":
    main()
