"""
Supplementary Figure S1:
"Temporal Coverage of Civilizational Windows in the Seshat and SPC1 Panels"

Visualises the distribution of polity-year windows across time for both datasets.
Output:
    figures/Figure_S1_temporal_coverage.pdf
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
DATA_DIR = ROOT / "data" / "final"
FIG_DIR = ROOT / "figures"

SESHAT_FILE = DATA_DIR / "seshat_EI_collapse_panel_w100.csv"
SPC1_FILE   = DATA_DIR / "SPC1_collapse_panel_w100_horizons.csv"


def main():
    print(f"[LOAD] Seshat: {SESHAT_FILE}")
    print(f"[LOAD] SPC1:   {SPC1_FILE}")

    seshat = pd.read_csv(SESHAT_FILE)
    spc1   = pd.read_csv(SPC1_FILE)

    seshat_years = seshat["year"].dropna()
    spc1_years   = spc1["year"].dropna()

    FIG_DIR.mkdir(exist_ok=True, parents=True)
    outpath = FIG_DIR / "Figure_S1_temporal_coverage.pdf"

    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.linewidth": 1.2,
        "axes.labelsize": 11,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    fig, ax = plt.subplots(figsize=(7, 3.7))

    ax.hist(
        [seshat_years, spc1_years],
        bins=40,
        label=["Seshat (EI Panel)", "SPC1 Panel"],
        alpha=0.7,
    )

    ax.set_xlabel("Calendar year")
    ax.set_ylabel("Number of polity-year windows")
    ax.legend(frameon=False)
    ax.set_title("Figure S1. Temporal Coverage of Civilizational Data", fontweight="bold")

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    print(f"[SAVE] Figure S1 written to:\n  {outpath}")


if __name__ == "__main__":
    main()
