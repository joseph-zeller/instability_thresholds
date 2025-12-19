"""
Generate Table S4: Robustness of η-ratio Thresholds Across Temporal Horizons
-------------------------------------------------------------------------------

This script:
- Uses the canonical cross-validated η-ratio threshold for the 100-year horizon
  (from results/thresholds/threshold_cv_summary_w100.csv, as in Table 1).
- Fits logistic regression models on the Seshat EI collapse panel to explore
  50-year and 150-year horizons:
    * collapse_next_50y
    * collapse_next_150y
- Computes for each row:
    * n
    * θ* (decision threshold where P(collapse)=0.5)
    * Percentile of θ* in η-ratio distribution
    * AUC
- Saves CSV + SI-formatted Markdown.

"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

# Script lives in: <repo_root>/src/tables
REPO_ROOT = Path(__file__).resolve().parents[2]

# Horizon panel for 50y / 150y robustness
PANEL_PATH = REPO_ROOT / "data" / "final" / "seshat_EI_collapse_panel_w100.csv"

# Canonical cross-validated thresholds for 100-year horizon (Table 1 source)
THRESH_SUMMARY_PATH = REPO_ROOT / "results" / "thresholds" / "threshold_cv_summary_w100.csv"

# Put S4 alongside other tables
OUT_DIR = REPO_ROOT / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "table_S4_eta_horizons.csv"
OUT_MD  = OUT_DIR / "table_S4_eta_horizons.md"

PREDICTOR = "eta_ratio"

# For 50 & 150-year robustness, we use the panel labels
ROBUSTNESS_HORIZONS = [
    ("η-ratio (50-year horizon)",  "collapse_next_50y"),
    ("η-ratio (150-year horizon)", "collapse_next_150y"),
]


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------

def fit_model(df: pd.DataFrame, predictor: str, target: str):
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
    pct = 100.0 * np.mean(vals <= theta)

    return len(df), theta, pct, auc


def get_canonical_eta_row() -> dict:
    """
    Pull the canonical 100-year η-ratio threshold and percentile from the
    cross-validated summary used by Table 1.

    This ensures the S4 100-year line matches the manuscript's 95.2nd percentile.
    """
    if not THRESH_SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Missing threshold summary: {THRESH_SUMMARY_PATH}")

    df = pd.read_csv(THRESH_SUMMARY_PATH)

    # Dataset label used in make_table1_core_models.py
    sub = df[df["dataset"] == "Seshat_eta_ratio_w100"]
    if sub.empty:
        raise ValueError("No Seshat_eta_ratio_w100 row found in threshold_cv_summary_w100.csv")

    r = sub.iloc[0]

    return {
        "Horizon (Model)": "η-ratio (100-year horizon)",
        "n": int(r["n"]),
        "theta": float(r["theta_mean"]),
        "percentile": float(r["threshold_percentile"]),
        "auc": float(r["auc_mean"]),
    }


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

def main():
    print("[INFO] Loading panel:", PANEL_PATH)
    panel = pd.read_csv(PANEL_PATH)

    rows = []

    # 1. Canonical 100-year row (cross-validated η threshold, matches Table 1)
    print("[INFO] Using canonical η-ratio threshold for 100-year horizon from",
          THRESH_SUMMARY_PATH)
    rows.append(get_canonical_eta_row())

    # 2. Robustness rows for 50y and 150y horizons, fit directly on the panel
    for label, target in ROBUSTNESS_HORIZONS:
        if target not in panel.columns:
            print(f"[WARN] Missing target column: {target}, skipping {label}.")
            continue

        sub = panel.dropna(subset=[PREDICTOR, target])
        if sub.empty:
            print(f"[WARN] No rows for {label}, skipping.")
            continue

        print(f"[INFO] Processing robustness horizon: {label}")

        n, theta, pct, auc = fit_model(sub, PREDICTOR, target)

        rows.append(
            {
                "Horizon (Model)": label,
                "n": n,
                "theta": theta,
                "percentile": pct,
                "auc": auc,
            }
        )

    out = pd.DataFrame(rows)
    out_rounded = out.copy()
    out_rounded["theta"] = out_rounded["theta"].round(2)
    out_rounded["percentile"] = out_rounded["percentile"].round(1)
    out_rounded["auc"] = out_rounded["auc"].round(2)

    out_rounded.to_csv(OUT_CSV, index=False)
    print("[OK] Wrote CSV →", OUT_CSV)

    # ------------------ Markdown (SI-formatted) ------------------
    with open(OUT_MD, "w", encoding="utf-8") as f:
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
        f.write("- The 100-year η-ratio row uses the cross-validated threshold summary\n")
        f.write("  from threshold_cv_summary_w100.csv (as in Table 1).\n")
        f.write("- θ* is the logistic decision threshold where P(collapse)=0.5.\n")
        f.write("- Percentiles are computed on the 0–100 scale.\n")

    print("[OK] Wrote Markdown →", OUT_MD)


if __name__ == "__main__":
    main()
