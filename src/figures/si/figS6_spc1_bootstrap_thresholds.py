#!/usr/bin/env python
"""
Figure S6: Bootstrap distributions of the SPC1 collapse threshold (w100).

Panel A: Histogram of bootstrap SPC1* thresholds, with observed value and
         95% bootstrap confidence interval.
Panel B: Histogram of bootstrap SPC1* percentiles (within the SPC1
         distribution), again with observed value and 95% CI.

Inputs
------
data/final/SPC1_collapse_panel_w100.csv
data/final/bootstrap_spc1_threshold_w100.csv

Outputs
-------
figures/Figure_S6_SPC1_bootstrap_thresholds.png
figures/Figure_S6_SPC1_bootstrap_thresholds.pdf

Run from repo root:
    python src/figures/figS6_spc1_bootstrap_thresholds.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


# ---------------------------------------------------------------------
# Paths and config
# ---------------------------------------------------------------------

DATA_DIR = Path("data/final")
FIG_DIR = Path("figures")

SPC1_PANEL_FILE = DATA_DIR / "SPC1_collapse_panel_w100.csv"
BOOT_FILE = DATA_DIR / "bootstrap_spc1_threshold_w100.csv"

FIG_PNG = FIG_DIR / "Figure_S6_SPC1_bootstrap_thresholds.png"
FIG_PDF = FIG_DIR / "Figure_S6_SPC1_bootstrap_thresholds.pdf"

PREDICTOR_COL = "SPC1"
OUTCOME_COL = "collapse_next_100y"


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def fit_logistic_threshold_x(x: np.ndarray, y: np.ndarray) -> float:
    """
    Fit logistic regression of collapse on predictor x and return the
    threshold x* at which P(collapse) = 0.5.

    Model:
        logit(P) = b0 + b1 * x
    Threshold:
        0 = b0 + b1 * x*  =>  x* = -b0 / b1

    Returns NaN if the slope is zero (degenerate fit).
    """
    X = x.reshape(-1, 1)
    y = y.astype(int)

    clf = LogisticRegression(
        solver="lbfgs",
        penalty="l2",
        C=1.0,
        fit_intercept=True,
        max_iter=10_000,
    )
    clf.fit(X, y)

    b0 = float(clf.intercept_[0])
    b1 = float(clf.coef_[0][0])

    if b1 == 0:
        return np.nan

    return float(-b0 / b1)


def empirical_percentile(x_values: np.ndarray, threshold: float) -> float:
    """Percentile rank of 'threshold' within x_values: 100 * P(X <= threshold)."""
    if np.isnan(threshold):
        return np.nan
    return 100.0 * np.mean(x_values <= threshold)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    # --------------------- Load SPC1 panel ---------------------------
    if not SPC1_PANEL_FILE.exists():
        raise FileNotFoundError(f"SPC1 panel not found: {SPC1_PANEL_FILE}")

    df_panel = pd.read_csv(SPC1_PANEL_FILE)

    if PREDICTOR_COL not in df_panel.columns:
        raise KeyError(f"Column '{PREDICTOR_COL}' not found in {SPC1_PANEL_FILE}")
    if OUTCOME_COL not in df_panel.columns:
        raise KeyError(f"Column '{OUTCOME_COL}' not found in {SPC1_PANEL_FILE}")

    df_panel_clean = df_panel[[PREDICTOR_COL, OUTCOME_COL]].dropna().copy()
    x = df_panel_clean[PREDICTOR_COL].to_numpy(dtype=float)
    y = df_panel_clean[OUTCOME_COL].to_numpy(dtype=int)

    obs_star = fit_logistic_threshold_x(x, y)
    obs_pct = empirical_percentile(x, obs_star)

    print(f"[INFO] Observed SPC1* threshold   : {obs_star:.6f}")
    print(f"[INFO] Observed SPC1* percentile  : {obs_pct:.3f}")

    # --------------------- Load bootstrap draws ----------------------
    if not BOOT_FILE.exists():
        raise FileNotFoundError(f"Bootstrap file not found: {BOOT_FILE}")

    df_boot = pd.read_csv(BOOT_FILE)

    if "spc1_star_boot" not in df_boot.columns or "percentile_boot" not in df_boot.columns:
        raise KeyError("Expected columns 'spc1_star_boot' and 'percentile_boot' in bootstrap file.")

    stars = df_boot["spc1_star_boot"].to_numpy(dtype=float)
    percs = df_boot["percentile_boot"].to_numpy(dtype=float)  # 0–100 scale

    # Remove NaNs if present
    mask_valid = ~np.isnan(stars) & ~np.isnan(percs)
    stars = stars[mask_valid]
    percs = percs[mask_valid]

    print(f"[INFO] Using {len(stars)} valid bootstrap draws.")

    # --------------------- Bootstrap CIs -----------------------------
    ci_star_low, ci_star_high = np.percentile(stars, [2.5, 97.5])
    ci_pct_low, ci_pct_high = np.percentile(percs, [2.5, 97.5])

    print(f"[INFO] 95% CI SPC1*           : [{ci_star_low:.6f}, {ci_star_high:.6f}]")
    print(f"[INFO] 95% CI SPC1* percentile: [{ci_pct_low:.3f}, {ci_pct_high:.3f}]")

    # For plotting, clip distributions at 1st–99th percentiles
    lo_star, hi_star = np.percentile(stars, [1, 99])
    lo_pct, hi_pct = np.percentile(percs, [1, 99])

    stars_plot = np.clip(stars, lo_star, hi_star)
    percs_plot = np.clip(percs, lo_pct, hi_pct)

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 10

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8))

    ax1.hist(stars_plot, bins=30)
    ax1.axvline(obs_star, linestyle="--", linewidth=1)
    ax1.axvline(ci_star_low, linestyle=":", linewidth=1)
    ax1.axvline(ci_star_high, linestyle=":", linewidth=1)
    ax1.set_xlabel("SPC1* threshold")
    ax1.set_ylabel("Bootstrap frequency")
    ax1.set_title("A. SPC1* bootstrap distribution",
                  fontsize=10, fontweight="bold", pad=6)

    ax2.hist(percs_plot, bins=30)
    ax2.axvline(obs_pct, linestyle="--", linewidth=1)
    ax2.axvline(ci_pct_low, linestyle=":", linewidth=1)
    ax2.axvline(ci_pct_high, linestyle=":", linewidth=1)
    ax2.set_xlabel("SPC1* percentile")
    ax2.set_ylabel("Bootstrap frequency")
    ax2.set_title("B. SPC1* percentile bootstrap distribution",
                  fontsize=10, fontweight="bold", pad=6)

    fig.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(FIG_PDF, dpi=300, bbox_inches="tight")
    print(f"[SAVE] Figure S6 written to:\n  {FIG_PNG}\n  {FIG_PDF}")


if __name__ == "__main__":
    main()
