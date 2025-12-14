"""
Generate Table S4: Robustness of η-ratio Thresholds Across Temporal Horizons
-------------------------------------------------------------------------------

This script:
- Loads the Seshat EI collapse panel (w100)
- Fits logistic regression models for:
    * collapse_next_50y
    * collapse_next_100y
    * collapse_next_150y
- Computes:
    * n
    * θ* (decision threshold where P(collapse)=0.5)
    * Percentile of θ* in η-ratio distribution
    * AUC
- Saves CSV + SI-formatted Markdown.

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

OUTPUT_CSV = REPO_ROOT / "output" / "table_S4_eta_horizons.csv"
OUTPUT_MD  = REPO_ROOT / "output" / "table_S4_eta_horizons.md"

PREDICTOR = "eta_ratio"

HORIZONS = [
    ("η-ratio (50-year horizon)",  "collapse_next_50y"),
    ("η-ratio (100-year horizon)", "collapse_next_100y"),
    ("η-ratio (150-year horizon)", "collapse_next_150y"),
]


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------

def fit_model(df, predictor, target):
    """Fit logistic regression and compute θ*, percentile, AUC."""
    x = df[predictor].to_numpy().reshape(-1, 1)
    y = df[target].to_numpy()

    clf = LogisticRegression(solver="lbfgs")
    clf.fit(x, y)

    intercept = clf.intercept_[0]
    coef = clf.coef_[0][0]

    theta = -intercept / coef if coef != 0 else np.nan

    preds = clf.predict_proba(x)[:, 1]
    auc = roc_auc_score(y, preds)

    vals = df[predictor].to_numpy()
    pct = 100 * np.mean(vals <= theta)

    return len(df), theta, pct, auc


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

def main():
    print("[INFO] Loading:", DATA_PATH)
    df = pd.read_csv(DATA_PATH)

    rows = []

    for label, target in HORIZONS:
        if target not in df.columns:
            print(f"[WARN] Missing target column: {target}, skipping.")
            continue

        sub = df.dropna(subset=[PREDICTOR, target])
        if sub.empty:
            print(f"[WARN] No rows for {label}, skipping.")
            continue

        print(f"[INFO] Processing horizon: {label}")

        n, theta, pct, auc = fit_model(sub, PREDICTOR, target)

        rows.append({
            "Horizon (Model)": label,
            "n": n,
            "theta": theta,
            "percentile": pct,
            "auc": auc
        })

    out = pd.DataFrame(rows)
    out_rounded = out.copy()
    out_rounded["theta"] = out_rounded["theta"].round(2)
    out_rounded["percentile"] = out_rounded["percentile"].round(1)
    out_rounded["auc"] = out_rounded["auc"].round(2)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_rounded.to_csv(OUTPUT_CSV, index=False)
    print("[OK] Wrote CSV →", OUTPUT_CSV)

    # ------------------ Markdown (SI-formatted) ------------------
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("Table S4. Robustness of η-ratio Thresholds Across Temporal Horizons\n")
        f.write("-------------------------------------------------------------------------------\n")
        f.write("Horizon (Model)                   |  n  |   θ*   | Percentile |   AUC\n")
        f.write("-------------------------------------------------------------------------------\n")

        for _, row in out_rounded.iterrows():
            f.write(
                f"{row['Horizon (Model)']:<32} | "
                f"{int(row['n']):>3} | "
                f"{row['theta']:>6.2f} | "
                f"{row['percentile']:>10.1f}% | "
                f"{row['auc']:>5.2f}\n"
            )

        f.write("-------------------------------------------------------------------------------\n")
        f.write("Notes:\n")
        f.write("- θ* is the logistic decision threshold where P(collapse)=0.5.\n")
        f.write("- Percentiles computed relative to the empirical η-ratio distribution.\n")
        f.write("- Stability of θ* across temporal windows indicates that the\n")
        f.write("  high-percentile instability band is not dependent on horizon length.\n")

    print("[OK] Wrote Markdown →", OUTPUT_MD)


if __name__ == "__main__":
    main()
