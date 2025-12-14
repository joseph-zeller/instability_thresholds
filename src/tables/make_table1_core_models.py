#!/usr/bin/env python
"""
make_table1_core_models.py

Build Table 1 (core models) summarising collapse threshold estimates for:

  - Seshat energetic–informational overshoot (eta_ratio)
  - SPC1 structural-demographic stress index (SPC1)

Input:
  results/thresholds/threshold_cv_summary_w100.csv

Output:
  tables/table1_core_models.csv

Notes:
- Uses cross-validated thresholds (reported values).
- Percentiles are on the 0–100 scale.
- Markdown output is intentionally not generated to avoid optional dependencies
  (e.g., pandas.to_markdown requires 'tabulate').
"""

from pathlib import Path
import pandas as pd


# ---------------------
# Paths
# ---------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "thresholds"
OUT_DIR = REPO_ROOT / "tables"

IN_FILE = RESULTS_DIR / "threshold_cv_summary_w100.csv"
OUT_CSV = OUT_DIR / "table1_core_models.csv"


# ---------------------
# Helpers
# ---------------------

def ensure_outdir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {IN_FILE.name}: {missing}")


# ---------------------
# Main
# ---------------------

def main() -> None:
    ensure_outdir()

    if not IN_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {IN_FILE}")

    df = pd.read_csv(IN_FILE)

    # Expected schema from src/compute_thresholds.py
    require_columns(
        df,
        [
            "dataset",
            "predictor",
            "n",
            "pos",
            "neg",
            "theta_mean",
            "theta_sd",
            "threshold_percentile",
            "auc_mean",
            "auc_sd",
        ],
    )

    rows = []

    for _, r in df.iterrows():
        if r["dataset"] == "Seshat_eta_ratio_w100":
            label = "Energetic–informational overshoot (ηᵢ/ηₑ)"
        elif r["dataset"] == "SPC1_w100":
            label = "Structural–demographic stress (SPC1)"
        else:
            # Ignore unknown datasets defensively
            continue

        rows.append(
            {
                "Indicator": label,
                "n": int(r["n"]),
                "Collapse cases": int(r["pos"]),
                "Non-collapse cases": int(r["neg"]),
                "Threshold θ (mean)": f"{float(r['theta_mean']):.2f}",
                "Threshold θ (SD)": f"{float(r['theta_sd']):.2f}",
                "Threshold percentile": f"{float(r['threshold_percentile']):.1f}",
                "AUC (mean)": f"{float(r['auc_mean']):.2f}",
                "AUC (SD)": f"{float(r['auc_sd']):.2f}",
            }
        )

    out = pd.DataFrame(rows)

    # Paper-friendly order
    out = out[
        [
            "Indicator",
            "n",
            "Collapse cases",
            "Non-collapse cases",
            "Threshold θ (mean)",
            "Threshold θ (SD)",
            "Threshold percentile",
            "AUC (mean)",
            "AUC (SD)",
        ]
    ]

    out.to_csv(OUT_CSV, index=False)

    print("[make_table1_core_models] Wrote:")
    print(f"  {OUT_CSV.relative_to(REPO_ROOT)}")
    print("[make_table1_core_models] Note: Markdown output disabled (no 'tabulate' dependency).")


if __name__ == "__main__":
    main()
