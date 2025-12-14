"""
figS3_transformation_robustness.py
--------------------------------------------------------------
Supplementary Figure S3:
"Robustness of η* Threshold Across Transformations"

Uses the output of build_table_S3_spec_robustness.py:

    data/final/table_S3_spec_robustness.csv

and plots the percentile position of η* for each model variant.

Output:
    figures/Figure_S3_transformation_robustness.pdf
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
DATA_DIR = ROOT / "data" / "final"
FIG_DIR = ROOT / "figures"

S3_FILE = DATA_DIR / "table_S3_spec_robustness.csv"


def main():
    print(f"[ROOT] {ROOT}")
    print(f"[LOAD] S3 robustness table: {S3_FILE}")

    if not S3_FILE.exists():
        raise FileNotFoundError(
            f"Table S3 data not found at {S3_FILE}. "
            "Run build_table_S3_spec_robustness.py first."
        )

    df = pd.read_csv(S3_FILE)

    # Expect columns: Model Variant, n, eta_star, percentile, auc
    if "Model Variant" not in df.columns or "percentile" not in df.columns:
        raise ValueError("Table S3 file missing required columns.")

    # Convert percentile to %
    df["percentile_pct"] = df["percentile"] * 100.0

    variants = df["Model Variant"].tolist()
    pct_vals = df["percentile_pct"].tolist()

    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.linewidth": 1.2,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
    })

    FIG_DIR.mkdir(exist_ok=True, parents=True)
    outpath = FIG_DIR / "Figure_S3_transformation_robustness.pdf"

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(
        variants,
        pct_vals,
        marker="o",
        linewidth=2.0,
    )

    ax.set_ylabel("Percentile of η* threshold")
    ax.set_ylim(0, 105)
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(variants, rotation=20, ha="right")

    # Smaller, harmonised title
    ax.set_title(
        "Figure S3. Robustness of η* Threshold Across Transformations",
        fontweight="bold",
        fontsize=10,
        pad=10,
    )

    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    print(f"\n[SAVE] Figure S3 written to:\n  {outpath}")


if __name__ == "__main__":
    main()
