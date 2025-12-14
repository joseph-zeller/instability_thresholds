"""
figS2_horizon_robustness.py
--------------------------------------------------------------
Supplementary Figure S2:
"Horizon Robustness of Instability Thresholds"

For each horizon H ∈ {50, 100, 150} and dataset:

    - Seshat EI panel: collapse_next_Hy ~ eta_ratio
    - SPC1 panel:      collapse_next_Hy ~ SPC1

we estimate the logistic threshold X* where p(collapse) = 0.5 and
express it as a percentile within the empirical distribution of
the predictor.

Output:
    figures/Figure_S2_horizon_robustness.pdf
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
DATA_DIR = ROOT / "data" / "final"
FIG_DIR = ROOT / "figures"

SESHAT_FILE = DATA_DIR / "seshat_EI_collapse_panel_w100.csv"
SPC1_FILE   = DATA_DIR / "SPC1_collapse_panel_w100_horizons.csv"

HORIZONS = (50, 100, 150)


def compute_threshold_stats(df: pd.DataFrame, label_col: str, predictor_col: str) -> dict:
    """
    Fit 1D logistic model: label_col ~ predictor_col

    Returns:
        n, x_star (threshold), percentile, auc
    """
    sub = df[[predictor_col, label_col]].dropna()

    y = sub[label_col].astype(int).to_numpy()
    x = sub[predictor_col].to_numpy()

    n = len(sub)
    if n == 0 or len(np.unique(y)) < 2:
        return {"n": n, "x_star": np.nan, "percentile": np.nan, "auc": np.nan}

    X = x.reshape(-1, 1)

    clf = LogisticRegression(solver="lbfgs")
    clf.fit(X, y)

    coef = clf.coef_[0, 0]
    intercept = clf.intercept_[0]

    if coef == 0:
        x_star = np.nan
    else:
        x_star = float(-intercept / coef)

    if np.isnan(x_star):
        pct = np.nan
    else:
        pct = float(np.mean(x <= x_star))

    try:
        probs = clf.predict_proba(X)[:, 1]
        auc = float(roc_auc_score(y, probs))
    except Exception:
        auc = np.nan

    return {"n": n, "x_star": x_star, "percentile": pct, "auc": auc}


def main():
    print(f"[ROOT] {ROOT}")
    print(f"[LOAD] Seshat EI panel: {SESHAT_FILE}")
    print(f"[LOAD] SPC1 panel:     {SPC1_FILE}")

    if not SESHAT_FILE.exists():
        raise FileNotFoundError(SESHAT_FILE)
    if not SPC1_FILE.exists():
        raise FileNotFoundError(SPC1_FILE)

    seshat = pd.read_csv(SESHAT_FILE)
    spc1   = pd.read_csv(SPC1_FILE)

    seshat_pcts = []
    spc1_pcts = []

    print("\n[RUN] Seshat (predictor = eta_ratio)")
    for H in HORIZONS:
        label_col = f"collapse_next_{H}y"
        stats = compute_threshold_stats(seshat, label_col=label_col, predictor_col="eta_ratio")
        pct = stats["percentile"] * 100 if stats["percentile"] == stats["percentile"] else np.nan
        seshat_pcts.append(pct)
        print(f"  H={H}y: n={stats['n']}, X*={stats['x_star']:.3f}, pct={pct:.2f}%, AUC={stats['auc']:.3f}")

    print("\n[RUN] SPC1 (predictor = SPC1)")
    for H in HORIZONS:
        label_col = f"collapse_next_{H}y"
        stats = compute_threshold_stats(spc1, label_col=label_col, predictor_col="SPC1")
        pct = stats["percentile"] * 100 if stats["percentile"] == stats["percentile"] else np.nan
        spc1_pcts.append(pct)
        print(f"  H={H}y: n={stats['n']}, X*={stats['x_star']:.3f}, pct={pct:.2f}%, AUC={stats['auc']:.3f}")

    # ----------------- Figure -----------------
    FIG_DIR.mkdir(exist_ok=True, parents=True)
    outpath = FIG_DIR / "Figure_S2_horizon_robustness.pdf"

    horizons = np.array(HORIZONS, dtype=float)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.linewidth": 1.2,
        "axes.labelsize": 11,
        "axes.titlesize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
    })

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(horizons, seshat_pcts, marker="o", linewidth=2.0,
            label="Seshat (η-ratio)")
    ax.plot(horizons, spc1_pcts, marker="s", linewidth=2.0,
            label="SPC1 panel")

    # Optional reference band
    ax.axhline(90, linestyle="--", linewidth=0.8)
    ax.axhline(97, linestyle="--", linewidth=0.8)

    ax.set_xlabel("Collapse horizon (years)")
    ax.set_ylabel("Percentile of η* threshold")
    ax.set_xticks(horizons)
    ax.set_ylim(0, 105)

    ax.legend(frameon=False)
    ax.set_title("Figure S2. Horizon Robustness of Instability Thresholds",
                 fontweight="bold")

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    print(f"\n[SAVE] Figure S2 written to:\n  {outpath}")


if __name__ == "__main__":
    main()
