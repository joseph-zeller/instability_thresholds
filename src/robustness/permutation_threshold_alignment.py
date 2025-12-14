#!/usr/bin/env python3
"""
permutation_threshold_alignment.py

Permutation / randomisation test for cross-dataset threshold percentile convergence.

Null:
  - Draw random thresholds uniformly from each predictor's empirical range
    (eta_ratio for Seshat; SPC1 for SPC1 dataset).
  - Convert each random threshold to an empirical percentile (kind="weak").
  - Compute absolute percentile difference.

Observed:
  - Uses fixed observed thresholds (from compute_thresholds CV summary).

Outputs (written under results/robustness/permutation/):
  - permutation_threshold_alignment_w100_null.csv
  - permutation_threshold_alignment_w100_summary.txt
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import percentileofscore


# -----------------------------
# Config
# -----------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]

SESHAT_CSV = REPO_ROOT / "data" / "final" / "seshat_EI_collapse_panel_w100.csv"
SPC1_CSV   = REPO_ROOT / "data" / "final" / "SPC1_collapse_panel_w100.csv"

OUT_DIR = REPO_ROOT / "results" / "robustness" / "permutation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Observed thresholds (CV-reported)
OBS_ETA_THRESH  = 2.116612
OBS_SPC1_THRESH = 5.059305

N_PERMS = 10_000
SEED = 42
PERCENTILE_KIND = "weak"   # must match SI text / prior runs


# -----------------------------
# Helpers
# -----------------------------

def load_values(path: Path, col: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    df = pd.read_csv(path)
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in {path.name}")
    vals = df[col].dropna().to_numpy(dtype=float)
    if vals.size == 0:
        raise ValueError(f"No non-NaN values found for '{col}' in {path.name}")
    return vals


def pct(values: np.ndarray, x: float) -> float:
    # percentileofscore returns percentile on 0–100 scale
    return float(percentileofscore(values, x, kind=PERCENTILE_KIND))


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    eta_values = load_values(SESHAT_CSV, "eta_ratio")
    spc1_values = load_values(SPC1_CSV, "SPC1")

    min_eta, max_eta = float(np.min(eta_values)), float(np.max(eta_values))
    min_spc1, max_spc1 = float(np.min(spc1_values)), float(np.max(spc1_values))

    print(f"ETA:  n = {len(eta_values)} min = {min_eta} max = {max_eta}")
    print(f"SPC1: n = {len(spc1_values)} min = {min_spc1} max = {max_spc1}")

    obs_perc_eta = pct(eta_values, OBS_ETA_THRESH)
    obs_perc_spc1 = pct(spc1_values, OBS_SPC1_THRESH)
    obs_diff = abs(obs_perc_eta - obs_perc_spc1)

    print(f"Observed percentiles: eta = {obs_perc_eta:.4f}, spc1 = {obs_perc_spc1:.4f}")
    print(f"Observed percentile diff = {obs_diff:.4f}")

    rng = np.random.default_rng(SEED)
    diffs = np.empty(N_PERMS, dtype=float)

    for i in range(N_PERMS):
        r_eta = float(rng.uniform(min_eta, max_eta))
        r_spc1 = float(rng.uniform(min_spc1, max_spc1))

        p_eta = pct(eta_values, r_eta)
        p_spc1 = pct(spc1_values, r_spc1)

        diffs[i] = abs(p_eta - p_spc1)

    p_two_tailed = float(np.mean(diffs <= obs_diff))
    p_one_tailed = p_two_tailed / 2.0

    print(f"p_two_tailed = {p_two_tailed:.6f}")
    print(f"p_one_tailed (convergence) = {p_one_tailed:.6f}")

    # -----------------------------
    # Write outputs
    # -----------------------------
    null_csv = OUT_DIR / "permutation_threshold_alignment_w100_null.csv"
    summary_txt = OUT_DIR / "permutation_threshold_alignment_w100_summary.txt"

    out_df = pd.DataFrame({
        "diff_percentile_points": diffs
    })
    out_df.to_csv(null_csv, index=False)

    summary_lines = [
        "Permutation test: cross-dataset threshold percentile convergence",
        "",
        f"Inputs:",
        f"  Seshat file: {SESHAT_CSV.as_posix()}",
        f"  SPC1 file:   {SPC1_CSV.as_posix()}",
        "",
        f"Distributions:",
        f"  ETA:  n={len(eta_values)} min={min_eta} max={max_eta}",
        f"  SPC1: n={len(spc1_values)} min={min_spc1} max={max_spc1}",
        "",
        "Observed thresholds (CV-reported):",
        f"  obs_eta_thresh  = {OBS_ETA_THRESH}",
        f"  obs_spc1_thresh = {OBS_SPC1_THRESH}",
        "",
        "Observed percentiles:",
        f"  eta  = {obs_perc_eta:.6f}",
        f"  spc1 = {obs_perc_spc1:.6f}",
        f"  diff = {obs_diff:.6f}",
        "",
        "Null model:",
        f"  draws = {N_PERMS}",
        f"  seed  = {SEED}",
        f"  random thresholds ~ Uniform([min,max]) for each predictor",
        f"  percentile mapping kind = {PERCENTILE_KIND}",
        "",
        "P-values:",
        f"  p_two_tailed (diff <= observed) = {p_two_tailed:.6f}",
        f"  p_one_tailed (convergence)      = {p_one_tailed:.6f}",
        "",
        "Outputs:",
        f"  null distribution CSV: {null_csv.as_posix()}",
        f"  summary TXT:           {summary_txt.as_posix()}",
        "",
    ]

    summary_txt.write_text("\n".join(summary_lines), encoding="utf-8")

    print("\nWrote:")
    print(f"  {null_csv.relative_to(REPO_ROOT)}")
    print(f"  {summary_txt.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
