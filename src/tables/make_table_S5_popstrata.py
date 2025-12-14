"""
Generate Table S5: Threshold Robustness Across Polity Population Strata
--------------------------------------------------------------------------

This script:
- Loads the Seshat EI collapse panel (w100)
- Splits polities into 3 equal-sized strata by log_Pop:
    * Small (lower 33.3%)
    * Medium (middle 33.3%)
    * Large (upper 33.3%)
- Fits logistic regression η-ratio → collapse_next_100y for each stratum
- Computes:
    * n
    * θ* threshold (p=0.5)
    * Percentile of θ* in η-ratio distribution
    * AUC
- Saves:
    * CSV: table_S5_popstrata.csv
    * Markdown: table_S5_popstrata.md

"""

import pathlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data" / "final" / "seshat_EI_collapse_panel_w100.csv"

OUTPUT_CSV = REPO_ROOT / "output" / "table_S5_popstrata.csv"
OUTPUT_MD  = REPO_ROOT / "output" / "table_S5_popstrata.md"

PREDICTOR = "eta_ratio"
TARGET    = "collapse_next_100y"
POP_COL   = "log_Pop"


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------

def fit_threshold(df, predictor_col, target_col):
    """Fit logistic regression and compute θ*, percentile, AUC."""
    x = df[predictor_col].to_numpy().reshape(-1, 1)
    y = df[target_col].to_numpy()

    clf = LogisticRegression(solver="lbfgs")
    clf.fit(x, y)

    intercept = clf.intercept_[0]
    coef = clf.coef_[0][0]
    theta = -intercept / coef if coef != 0 else np.nan

    y_prob = clf.predict_proba(x)[:, 1]
    auc = roc_auc_score(y, y_prob)

    vals = df[predictor_col].to_numpy()
    pct = 100 * np.mean(vals <= theta)

    return len(df), theta, pct, auc


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

def main():
    print("[INFO] Loading:", DATA_PATH)
    df = pd.read_csv(DATA_PATH)

    # Drop NA rows for predictor or target
    df = df.dropna(subset=[PREDICTOR, TARGET, POP_COL])

    # Determine population strata using quantiles
    q1 = df[POP_COL].quantile(0.3333)
    q2 = df[POP_COL].quantile(0.6667)

    strata_defs = [
        ("Small polities",  df[df[POP_COL] <= q1]),
        ("Medium polities", df[(df[POP_COL] > q1) & (df[POP_COL] <= q2)]),
        ("Large polities",  df[df[POP_COL] > q2]),
    ]

    rows = []

    for label, subset in strata_defs:
        if subset.empty or subset[TARGET].nunique() < 2:
            print(f"[WARN] Insufficient data for: {label}; skipping.")
            continue

        print(f"[INFO] Processing: {label}")

        n, theta, pct, auc = fit_threshold(subset, PREDICTOR, TARGET)

        rows.append({
            "Population Stratum": label,
            "n": n,
            "theta": theta,
            "percentile": pct,
            "auc": auc
        })

    # Save CSV
    out_df = pd.DataFrame(rows)
    out_df_round = out_df.copy()
    out_df_round["theta"] = out_df_round["theta"].round(2)
    out_df_round["percentile"] = out_df_round["percentile"].round(1)
    out_df_round["auc"] = out_df_round["auc"].round(2)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df_round.to_csv(OUTPUT_CSV, index=False)
    print("[OK] Wrote CSV →", OUTPUT_CSV)

    # Markdown SI table
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("Table S5. Threshold Robustness Across Polity Population Strata\n")
        f.write("-------------------------------------------------------------------------------\n")
        f.write("Population Stratum                 |  n  |   θ*   | Percentile |   AUC\n")
        f.write("-------------------------------------------------------------------------------\n")

        for _, row in out_df_round.iterrows():
            f.write(
                f"{row['Population Stratum']:<32} | "
                f"{int(row['n']):>3} | "
                f"{row['theta']:>6.2f} | "
                f"{row['percentile']:>10.1f}% | "
                f"{row['auc']:>5.2f}\n"
            )

        f.write("-------------------------------------------------------------------------------\n")
        f.write("Notes:\n")
        f.write("- θ* is the logistic decision threshold where P(collapse)=0.5.\n")
        f.write("- Population strata are quantile-based (lower/middle/upper thirds).\n")
        f.write("- Stability of thresholds across strata indicates that η-ratio is not a size\n")
        f.write("  artefact but reflects internal systemic stress.\n")

    print("[OK] Wrote Markdown →", OUTPUT_MD)


if __name__ == "__main__":
    main()
