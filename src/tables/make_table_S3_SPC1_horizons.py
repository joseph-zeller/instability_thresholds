"""
Generate Table S3: Robustness of SPC1 Thresholds Across Temporal Horizons
-------------------------------------------------------------------------------

This script:
- Searches data/final/ for an SPC1 panel (CSV)
- Loads the SPC1 EI collapse panel
- Fits logistic models for:
    * collapse_next_50y
    * collapse_next_100y
    * collapse_next_150y
- Computes for each horizon:
    * n
    * θ* (decision threshold where P(collapse) = 0.5)
    * Percentile of θ* in the SPC1 distribution
    * AUC
- Saves:
    * CSV: table_S3_SPC1_horizons.csv
    * Markdown: table_S3_SPC1_horizons.md (SI formatted)
"""

import pathlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "final"

# Horizons + target labels
HORIZONS = [
    ("SPC1 (50-year horizon)", "collapse_next_50y"),
    ("SPC1 (100-year horizon)", "collapse_next_100y"),
    ("SPC1 (150-year horizon)", "collapse_next_150y"),
]

CSV_OUT = REPO_ROOT / "output" / "table_S3_SPC1_horizons.csv"
MD_OUT = REPO_ROOT / "output" / "table_S3_SPC1_horizons.md"


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------

def find_spc1_file():
    """
    Try to locate an SPC1 panel in data/final/.
    Prefer filenames containing 'SPC1' and 'w100'; fall back to any 'SPC1*.csv'.
    """
    candidates = list(DATA_DIR.glob("SPC1*100*.csv"))
    if not candidates:
        candidates = list(DATA_DIR.glob("SPC1*.csv"))

    if not candidates:
        raise FileNotFoundError(
            f"No SPC1 CSV found in {DATA_DIR}. "
            f"Expected something like 'SPC1_panel_w100.csv'."
        )

    if len(candidates) > 1:
        print("[WARN] Multiple SPC1 CSV files found; using the first one:")
        for c in candidates:
            print("       -", c.name)

    chosen = candidates[0]
    print(f"[INFO] Using SPC1 file: {chosen}")
    return chosen


def find_spc1_predictor_column(df: pd.DataFrame) -> str:
    """
    Determine which column to use as the SPC1 predictor.
    Prefer exact 'SPC1'; otherwise any column containing 'SPC1' (case-insensitive).
    """
    if "SPC1" in df.columns:
        return "SPC1"

    # Try partial match
    for col in df.columns:
        if "spc1" in col.lower():
            print(f"[INFO] Using predictor column inferred from name: {col}")
            return col

    raise KeyError(
        "Could not find an SPC1 predictor column. "
        "Expected 'SPC1' or something containing 'SPC1' in its name."
    )


def fit_threshold(df, predictor_col, target_col):
    """Fit logistic regression and compute n, θ*, percentile, AUC."""
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
    spc1_path = find_spc1_file()

    print(f"[INFO] Loading SPC1 panel:", spc1_path)
    df = pd.read_csv(spc1_path)

    predictor_col = find_spc1_predictor_column(df)
    print(f"[INFO] Using predictor column: {predictor_col}")

    rows = []

    for label, target_col in HORIZONS:
        if target_col not in df.columns:
            print(f"[WARN] Target column '{target_col}' not found; skipping {label}")
            continue

        print(f"[INFO] Processing horizon: {label} (target={target_col})")

        sub = df.dropna(subset=[predictor_col, target_col]).copy()
        if sub.empty:
            print(f"[WARN] No valid rows for {label}; skipping.")
            continue

        n, theta, pct, auc = fit_threshold(sub, predictor_col, target_col)

        rows.append(
            {
                "Model (Horizon)": label,
                "n": n,
                "theta": theta,
                "percentile": pct,
                "auc": auc,
            }
        )

    if not rows:
        raise RuntimeError("No horizons could be processed; check SPC1 file and columns.")

    out_df = pd.DataFrame(rows)
    out_df_rounded = out_df.copy()
    out_df_rounded["theta"] = out_df_rounded["theta"].round(2)
    out_df_rounded["percentile"] = out_df_rounded["percentile"].round(1)
    out_df_rounded["auc"] = out_df_rounded["auc"].round(2)

    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    out_df_rounded.to_csv(CSV_OUT, index=False)
    print("[OK] Wrote CSV to:", CSV_OUT)

    # SI-formatted Markdown
    with open(MD_OUT, "w", encoding="utf-8") as f:
        f.write("Table S3. Robustness of SPC1 Thresholds Across Temporal Horizons\n")
        f.write("-------------------------------------------------------------------------------\n")
        f.write("Model (Horizon)                        |  n   |   θ*   | Percentile |   AUC\n")
        f.write("-------------------------------------------------------------------------------\n")

        for _, row in out_df_rounded.iterrows():
            f.write(
                f"{row['Model (Horizon)']:<36} | "
                f"{int(row['n']):>3}  | "
                f"{row['theta']:>6.2f} | "
                f"{row['percentile']:>10.1f}% | "
                f"{row['auc']:>5.2f}\n"
            )

        f.write("-------------------------------------------------------------------------------\n")
        f.write("Notes:\n")
        f.write("- θ* = logistic decision threshold where P(collapse) = 0.5.\n")
        f.write("- Percentiles computed relative to the empirical SPC1 distribution.\n")
        f.write("- AUC reflects full-sample discrimination performance.\n")

    print("[OK] Wrote Markdown to:", MD_OUT)


if __name__ == "__main__":
    main()
