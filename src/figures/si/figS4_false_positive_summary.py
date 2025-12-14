"""
Supplementary Figure S4
High-Stress Survivors (False Positives) in the Instability Band

Definition:
    False positives are polity-windows where:
        • eta_ratio >= eta_star (100-year logistic threshold)
        • collapse_next_100y == 0

Purpose:
    Identifies windows operating above the instability threshold that did NOT collapse.
    This supports interpreting the 95th-percentile instability region as a probabilistic
    vulnerability band rather than a deterministic tipping point.

Output:
    figures/Figure_S4_false_positive_summary.pdf
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Path Configuration (Robust)
# ---------------------------------------------------------------------
# Script location: collapse_thresholds/src/figures/
# Project root:    collapse_thresholds/
HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]  # go up two levels to reach the repo root

DATA_DIR = ROOT / "data" / "final"
FIG_DIR = ROOT / "figures"

SESHAT_FILE = DATA_DIR / "seshat_EI_collapse_panel_w100.csv"

TOP_K = 12  # number of highest-percentile false positives to display


# ---------------------------------------------------------------------
# Logistic Threshold Estimation
# ---------------------------------------------------------------------
def fit_eta_threshold(df: pd.DataFrame) -> dict:
    """
    Fit collapse_next_100y ~ eta_ratio and compute the logistic threshold eta*.
    """
    required_cols = ["eta_ratio", "collapse_next_100y"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    sub = df[required_cols].dropna()
    y = sub["collapse_next_100y"].astype(int).to_numpy()
    x = sub["eta_ratio"].to_numpy()

    if len(sub) == 0 or len(np.unique(y)) < 2:
        return {"n": len(sub), "eta_star": np.nan, "percentile": np.nan, "auc": np.nan}

    X = x.reshape(-1, 1)
    clf = LogisticRegression(solver="lbfgs")
    clf.fit(X, y)

    coef = clf.coef_[0, 0]
    intercept = clf.intercept_[0]

    eta_star = np.nan if coef == 0 else float(-intercept / coef)
    percentile = float(np.mean(x <= eta_star)) if not np.isnan(eta_star) else np.nan

    try:
        auc = float(roc_auc_score(y, clf.predict_proba(X)[:, 1]))
    except Exception:
        auc = np.nan

    return {
        "n": len(sub),
        "eta_star": eta_star,
        "percentile": percentile,
        "auc": auc
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    print(f"[LOAD] Reading EI panel: {SESHAT_FILE}")

    df = pd.read_csv(SESHAT_FILE)

    # Threshold estimation
    stats = fit_eta_threshold(df)
    eta_star = stats["eta_star"]
    pct_star = stats["percentile"] * 100 if stats["percentile"] == stats["percentile"] else np.nan

    print(f"[THRESHOLD] n={stats['n']}, eta*={eta_star:.3f}, pct={pct_star:.2f}%, AUC={stats['auc']:.3f}")

    # Percentile computation
    df = df.copy()
    df["eta_pct"] = df["eta_ratio"].rank(pct=True) * 100.0

    # False positives = above threshold AND no collapse
    fp_mask = (df["eta_ratio"] >= eta_star) & (df["collapse_next_100y"] == 0)
    df_fp = df.loc[fp_mask].copy()
    print(f"[INFO] False-positive windows found: {len(df_fp)}")

    if df_fp.empty:
        print("[WARN] No false positives detected — plot not generated.")
        return

    # Label: "slug (year)"
    df_fp["label"] = df_fp["eim_slug"].astype(str) + " (" + df_fp["year"].astype(int).astype(str) + ")"

    # Select top K false positives by percentile
    df_fp = df_fp.sort_values("eta_pct", ascending=False).head(TOP_K)

    labels = df_fp["label"].tolist()
    pct_vals = df_fp["eta_pct"].tolist()

    FIG_DIR.mkdir(exist_ok=True, parents=True)
    outpath = FIG_DIR / "Figure_S4_false_positive_summary.pdf"

    # Plot settings
    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.linewidth": 1.2,
        "axes.labelsize": 11,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 9,
    })

    fig, ax = plt.subplots(figsize=(7, 5))
    y_positions = np.arange(len(labels))

    ax.barh(y_positions, pct_vals, color="gray")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

    ax.set_xlabel("Percentile of η_ratio")
    ax.set_xlim(0, 105)

    ax.set_title("Figure S4. High-Stress Survivors Above the η* Threshold", fontweight="bold")

    if pct_star == pct_star:
        ax.axvline(pct_star, linestyle="--", linewidth=1.0)

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")

    print(f"[SAVE] {outpath}")


if __name__ == "__main__":
    main()
