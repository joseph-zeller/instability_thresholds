#!/usr/bin/env python
"""
Bootstrap confidence intervals for SPC1 collapse threshold (w100).

This script:
  1. Loads the SPC1 collapse panel (w100) from data/final/.
  2. Fits a logistic regression of collapse_next_100y on SPC1.
  3. Computes the observed threshold SPC1* where P(collapse) = 0.5.
  4. Computes the percentile of SPC1* within the empirical SPC1 distribution.
  5. Uses bootstrap resampling to estimate uncertainty:
       - Resample rows with replacement B times
       - Refit logistic, recompute SPC1*
       - Record SPC1* and its percentile
  6. Saves all bootstrap draws to CSV.
  7. Computes percentile-based 95% CIs for:
       - SPC1*
       - SPC1* percentile
  8. Writes a plain-text summary with the key results.

Usage (from repo root):
    python src/robustness/bootstrap_spc1_threshold_w100.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DATA_FILE = Path("data/final/SPC1_collapse_panel_w100.csv")

PREDICTOR_COL = "SPC1"
OUTCOME_COL = "collapse_next_100y"

N_BOOTSTRAP = 10_000
RANDOM_SEED = 123  # affects resampling order only

OUT_CSV = Path("data/final/bootstrap_spc1_threshold_w100.csv")
SUMMARY_TXT = Path("data/final/bootstrap_spc1_threshold_w100_summary.txt")


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------


def fit_logistic_threshold_x(x: np.ndarray, y: np.ndarray) -> float:
    """
    Fit logistic regression of collapse on predictor x and return the threshold x*
    at which P(collapse) = 0.5.

    Model:
        logit(P) = b0 + b1 * x

    Threshold:
        0 = b0 + b1 * x*
        => x* = -b0 / b1

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

    x_star = -b0 / b1
    return float(x_star)


def empirical_percentile(x_values: np.ndarray, threshold: float) -> float:
    """
    Compute the percentile rank of 'threshold' within the empirical distribution
    of x_values.

    Percentile is defined as:
        100 * P(X <= threshold)

    Returns a value in [0, 100] or NaN if threshold is NaN.
    """
    if np.isnan(threshold):
        return np.nan
    return 100.0 * np.mean(x_values <= threshold)


# ---------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------


def main():
    # --------------------------------------------------------------
    # 1. Load data
    # --------------------------------------------------------------
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)

    if PREDICTOR_COL not in df.columns:
        raise KeyError(f"Predictor column '{PREDICTOR_COL}' not found in {DATA_FILE}")
    if OUTCOME_COL not in df.columns:
        raise KeyError(f"Outcome column '{OUTCOME_COL}' not found in {DATA_FILE}")

    df_clean = df[[PREDICTOR_COL, OUTCOME_COL]].dropna().copy()

    x = df_clean[PREDICTOR_COL].to_numpy(dtype=float)
    y = df_clean[OUTCOME_COL].to_numpy(dtype=int)

    n_obs = len(df_clean)
    print(f"[INFO] Loaded {n_obs} observations from {DATA_FILE}")

    # --------------------------------------------------------------
    # 2. Observed threshold and percentile
    # --------------------------------------------------------------
    x_star_obs = fit_logistic_threshold_x(x, y)
    perc_obs = empirical_percentile(x, x_star_obs)

    print(f"[INFO] Observed {PREDICTOR_COL}* (threshold) : {x_star_obs:.6f}")
    print(f"[INFO] Observed percentile of {PREDICTOR_COL}*: {perc_obs:.3f}")

    # --------------------------------------------------------------
    # 3. Bootstrap resampling
    # --------------------------------------------------------------
    rng = np.random.default_rng(RANDOM_SEED)

    boot_thresholds = np.empty(N_BOOTSTRAP, dtype=float)
    boot_percentiles = np.empty(N_BOOTSTRAP, dtype=float)

    print(f"[INFO] Running {N_BOOTSTRAP} bootstrap resamples...")

    indices = np.arange(n_obs, dtype=int)

    for b in range(N_BOOTSTRAP):
        # Sample row indices with replacement
        sample_idx = rng.choice(indices, size=n_obs, replace=True)
        x_boot = x[sample_idx]
        y_boot = y[sample_idx]

        x_star_boot = fit_logistic_threshold_x(x_boot, y_boot)
        perc_boot = empirical_percentile(x, x_star_boot)  # percentile in original x distribution

        boot_thresholds[b] = x_star_boot
        boot_percentiles[b] = perc_boot

        if (b + 1) % 1000 == 0:
            print(f"  [boot] Completed {b + 1}/{N_BOOTSTRAP}")

    # Filter out any NaNs from degenerate fits
    mask_valid = ~np.isnan(boot_thresholds) & ~np.isnan(boot_percentiles)
    valid_count = int(mask_valid.sum())

    boot_thresholds_valid = boot_thresholds[mask_valid]
    boot_percentiles_valid = boot_percentiles[mask_valid]

    if valid_count < N_BOOTSTRAP:
        print(
            f"[WARN] {N_BOOTSTRAP - valid_count} bootstrap resamples produced NaN "
            f"thresholds/percentiles and were excluded."
        )

    # --------------------------------------------------------------
    # 4. Summary statistics and CIs
    # --------------------------------------------------------------
    # Point estimates (observed)
    x_star_hat = x_star_obs
    perc_hat = perc_obs

    # Bootstrap means
    mean_x_star = float(np.mean(boot_thresholds_valid))
    mean_perc = float(np.mean(boot_percentiles_valid))

    # 95% percentile-based confidence intervals
    ci_lower_x_star, ci_upper_x_star = np.percentile(boot_thresholds_valid, [2.5, 97.5])
    ci_lower_perc, ci_upper_perc = np.percentile(boot_percentiles_valid, [2.5, 97.5])

    print("\n[RESULTS] Bootstrap SPC1 threshold (w100)")
    print("------------------------------------------------------------")
    print(f"Observed {PREDICTOR_COL}* (threshold)        : {x_star_hat:.6f}")
    print(f"Observed {PREDICTOR_COL}* percentile         : {perc_hat:.3f}")
    print(f"Bootstrap mean {PREDICTOR_COL}*              : {mean_x_star:.6f}")
    print(f"Bootstrap mean percentile                    : {mean_perc:.3f}")
    print(f"95% CI for {PREDICTOR_COL}*                  : [{ci_lower_x_star:.6f}, {ci_upper_x_star:.6f}]")
    print(f"95% CI for {PREDICTOR_COL}* percentile       : [{ci_lower_perc:.3f}, {ci_upper_perc:.3f}]")
    print(f"Valid bootstrap resamples                    : {valid_count}")

    # --------------------------------------------------------------
    # 5. Save bootstrap draws to CSV
    # --------------------------------------------------------------
    out_df = pd.DataFrame(
        {
            "iteration": np.arange(1, valid_count + 1, dtype=int),
            "spc1_star_boot": boot_thresholds_valid,
            "percentile_boot": boot_percentiles_valid,
        }
    )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)

    print(f"\n[INFO] Bootstrap draws saved to: {OUT_CSV}")

    # --------------------------------------------------------------
    # 6. Save text summary
    # --------------------------------------------------------------
    SUMMARY_TXT.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write("Bootstrap analysis for SPC1* threshold (w100)\n")
        f.write("--------------------------------------------------\n")
        f.write(f"Data file                  : {DATA_FILE}\n")
        f.write(f"Predictor column           : {PREDICTOR_COL}\n")
        f.write(f"Outcome column             : {OUTCOME_COL}\n")
        f.write(f"Observations (n)           : {n_obs}\n")
        f.write(f"Bootstrap resamples (valid): {valid_count}\n")
        f.write("\n")
        f.write(f"Observed {PREDICTOR_COL}*          : {x_star_hat:.6f}\n")
        f.write(f"Observed percentile        : {perc_hat:.6f}\n")
        f.write(f"Bootstrap mean {PREDICTOR_COL}*    : {mean_x_star:.6f}\n")
        f.write(f"Bootstrap mean percentile  : {mean_perc:.6f}\n")
        f.write(f"95% CI {PREDICTOR_COL}*           : [{ci_lower_x_star:.6f}, {ci_upper_x_star:.6f}]\n")
        f.write(f"95% CI percentile          : [{ci_lower_perc:.6f}, {ci_upper_perc:.6f}]\n")

    print(f"[INFO] Text summary saved to: {SUMMARY_TXT}")
    print("\n[DONE] Bootstrap analysis complete.")


if __name__ == "__main__":
    main()
