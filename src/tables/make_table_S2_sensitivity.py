"""
Table S2 — Sensitivity of η-ratio Instability Band to Excluding Influential Cases

This script tests whether excluding disputed or influential polities
changes the percentile location of the canonical instability threshold
η* = 2.116612. The threshold itself is NOT re-estimated.

Percentiles are computed relative to the FULL η-ratio distribution.
Model robustness is assessed using 5-fold stratified CV AUC.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

ETA_STAR = 2.116612

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = REPO_ROOT / "data" / "final" / "seshat_EI_collapse_panel_w100.csv"
OUT_FILE  = REPO_ROOT / "output" / "Table_S2_sensitivity.csv"


# ---------------------------------------------------------------------
# Cross-validated AUC
# ---------------------------------------------------------------------

def compute_cv_auc(df):
    X = df[["eta_ratio"]].to_numpy()
    y = df["collapse_next_100y"].astype(int).to_numpy()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y))

    for train, test in skf.split(X, y):
        model = LogisticRegression(
            solver="lbfgs",
            C=1e6,
            max_iter=10000
        )
        model.fit(X[train], y[train])
        oof[test] = model.predict_proba(X[test])[:, 1]

    return roc_auc_score(y, oof)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    # Load and clean data
    df = pd.read_csv(DATA_FILE)
    df = df.dropna(subset=["eta_ratio", "collapse_next_100y"]).copy()
    df["collapse_next_100y"] = df["collapse_next_100y"].astype(int)

    # Full reference distribution (for percentile)
    ref_vals = df["eta_ratio"].to_numpy()
    percentile = 100.0 * np.mean(ref_vals <= ETA_STAR)

    # Identify influential cases
    case_A = ["new_egyptian_kingdom"]

    max_eta_slug = df.loc[df["eta_ratio"].idxmax(), "eim_slug"]
    near_eta_slug = df.loc[
        (df["eta_ratio"] - ETA_STAR).abs().idxmin(),
        "eim_slug"
    ]

    exclusion_sets = {
        "Baseline (all cases)": [],
        "Exclude Case A (New Egyptian Kingdom)": case_A,
        "Exclude Case B (Maximum η-ratio polity)": [max_eta_slug],
        "Exclude Case C (Nearest-to-threshold polity)": [near_eta_slug],
        "Exclude Cases A–C": list(set(case_A + [max_eta_slug, near_eta_slug]))
    }

    rows = []

    for label, slugs in exclusion_sets.items():
        sub = df if not slugs else df[~df["eim_slug"].isin(slugs)].copy()

        auc = compute_cv_auc(sub)

        rows.append({
            "Model": label,
            "n": len(sub),
            "eta_star_fixed": ETA_STAR,
            "percentile": percentile,
            "AUC": auc
        })

    out = pd.DataFrame(rows)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_FILE, index=False)

    print("\nTable S2 written to:", OUT_FILE)
    print(out.round(3))


if __name__ == "__main__":
    main()
