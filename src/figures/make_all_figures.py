#!/usr/bin/env python
"""
make_all_figures.py

Runs all figure-generation scripts for the repository.

This script is called by src/run_all.py.

It executes each figure script as a module (runpy), so figures are generated
in the same Python process and inherit the repo-root working directory.

Edit FIGURE_SCRIPTS if you add/remove figure scripts.
"""

from pathlib import Path
import runpy


REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = REPO_ROOT / "src" / "figures"

# Update this list to match the scripts you actually keep in src/figures/
FIGURE_SCRIPTS = [
    # Manuscript figures (examples — adjust names to your actual files)
    "fig1a_spc1_distribution.py",
    "fig1b_eta_distribution.py",
    "fig2_logistic_fits.py",
    "fig3_percentile_convergence.py",
    "fig4_threshold_instability_band.py",
    "fig5_overload_trajectories_examples.py",
    # SI figures (examples — adjust)
    "figS1_something.py",
    "figS2_something.py",
]


def main() -> None:
    print(f"[make_all_figures] Repo root: {REPO_ROOT}")
    print(f"[make_all_figures] Figures directory: {FIG_DIR}")

    missing = []
    present = []

    for name in FIGURE_SCRIPTS:
        p = FIG_DIR / name
        if p.exists():
            present.append(p)
        else:
            missing.append(p)

    print("[make_all_figures] Scripts configured:")
    for p in present:
        print(f"  - {p.relative_to(REPO_ROOT)}")

    if missing:
        print("\n[make_all_figures] WARNING: Missing figure scripts (skipping):")
        for p in missing:
            print(f"  - {p.relative_to(REPO_ROOT)}")
        print("\n[make_all_figures] Tip: Update FIGURE_SCRIPTS in this file to match your repo.")
        # Do NOT fail hard — allow run_all to proceed with other stages.
        # If you prefer strict mode, raise FileNotFoundError here instead.

    for p in present:
        print("\n" + "=" * 72)
        print(f"[make_all_figures] Running {p.relative_to(REPO_ROOT)}")
        print("=" * 72)
        runpy.run_path(str(p), run_name="__main__")

    print("\n[make_all_figures] Done.")


if __name__ == "__main__":
    main()
