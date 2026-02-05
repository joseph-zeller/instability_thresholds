#!/usr/bin/env python3
"""
SI Table S6 — Robustness of η-ratio thresholds across polity population strata

Definitions (locked):
- θ* = decision boundary where P(collapse)=0.5 from full-sample logistic fit in each stratum
      θ* = -β0 / β1
- Percentile = weak percentile within stratum: Pr(η ≤ θ*) * 100
- AUC = stratified cross-validated (OOF) AUC with adaptive folds:
        k = min(5, minority_class_count). If minority_class_count < 2, AUC = NA.

Input:
- data/final/seshat_EI_collapse_panel_w100.csv

Outputs:
- tables/S6_population_strata.csv
- tables/S6_population_strata.md
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


# -----------------------------
# Paths
# -----------------------------
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]

DATA_FILE = REPO_ROOT / "data" / "final" / "seshat_EI_collapse_panel_w100.csv"

OUT_DIR = REPO_ROOT / "output" / "tables"
OUT_CSV = OUT_DIR / "S6_population_strata.csv"
OUT_MD = OUT_DIR / "S6_population_strata.md"


# -----------------------------
# Config
# -----------------------------
PRED = "eta_ratio"
POP = "log_Pop"
YCOL = "collapse_next_100y"

MAX_SPLITS = 5
RANDOM_STATE = 42

LOGIT_KW = dict(
    solver="lbfgs",
    C=1e6,
    max_iter=10000,
)


# -----------------------------
# Helpers
# -----------------------------
def require_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")


def weak_percentile(values: np.ndarray, theta: float) -> float:
    return float(100.0 * np.mean(values <= theta))


def fit_theta_star(df: pd.DataFrame) -> float:
    """Full-sample θ* from logistic decision boundary P=0.5."""
    X = df[[PRED]].to_numpy()
    y = df[YCOL].astype(int).to_numpy()

    # If one class only, θ* is not meaningful; return NaN
    if len(np.unique(y)) < 2:
        return np.nan

    clf = LogisticRegression(**LOGIT_KW)
    clf.fit(X, y)

    b0 = float(clf.intercept_[0])
    b1 = float(clf.coef_[0][0])

    if b1 == 0:
        return np.nan
    return float(-b0 / b1)


def cv_auc_oof_adaptive(df: pd.DataFrame) -> tuple[float, int]:
    """
    Stratified CV AUC with adaptive folds:
      k = min(MAX_SPLITS, minority_class_count)
    If minority_class_count < 2 → AUC undefined (NA).
    Returns (auc, k_used).
    """
    X = df[[PRED]].to_numpy()
    y = df[YCOL].astype(int).to_numpy()

    # Need both classes
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return np.nan, 0

    minority = int(counts.min())
    if minority < 2:
        # Cannot do stratified CV; too few events
        return np.nan, 0

    k = min(MAX_SPLITS, minority)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)

    oof = np.zeros(len(y), dtype=float)

    for tr, te in skf.split(X, y):
        y_tr = y[tr]
        # Guard against one-class training fold (can still happen with tiny samples)
        if len(np.unique(y_tr)) < 2:
            return np.nan, k

        clf = LogisticRegression(**LOGIT_KW)
        clf.fit(X[tr], y_tr)
        oof[te] = clf.predict_proba(X[te])[:, 1]

    return float(roc_auc_score(y, oof)), k


def write_markdown(df: pd.DataFrame) -> None:
    lines = []
    lines.append("Table S6. Threshold robustness across polity population strata")
    lines.append("---------------------------------------------------------------------------")
    lines.append("Population Stratum                 |  n  |   θ*    | Percentile |  AUC | k")
    lines.append("---------------------------------------------------------------------------")

    for _, r in df.iterrows():
        auc = r["AUC"]
        auc_str = f"{auc:.2f}" if pd.notna(auc) else "NA"
        k = int(r["k"])
        theta = r["θ*"]
        theta_str = f"{theta:.2f}" if pd.notna(theta) else "NA"
        lines.append(
            f"{r['Population Stratum']:<32} | {int(r['n']):>3} | {theta_str:>7} | {r['Percentile']:>9} | {auc_str:>5} | {k:>1}"
        )

    lines.append("---------------------------------------------------------------------------")
    lines.append("Notes:")
    lines.append("- Population strata are terciles of log population.")
    lines.append("- θ* is the P(collapse)=0.5 boundary from full-sample logistic fits within each stratum.")
    lines.append("- Percentile is weak percentile within each stratum’s η-ratio distribution.")
    lines.append(f"- AUC is stratified out-of-fold AUC with adaptive folds: k = min(5, minority class count) (random_state={RANDOM_STATE}).")
    lines.append("- If a stratum has <2 events in the minority class, cross-validated AUC is undefined and reported as NA.")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_FILE.exists():
        raise FileNotFoundError(DATA_FILE)

    df = pd.read_csv(DATA_FILE)
    require_cols(df, [PRED, POP, YCOL])

    df = df.dropna(subset=[PRED, POP, YCOL]).copy()
    df[YCOL] = df[YCOL].astype(int)

    # Create terciles of population
    df["pop_stratum"] = pd.qcut(df[POP], 3, labels=["Small", "Medium", "Large"])

    rows = []

    for label in ["Small", "Medium", "Large"]:
        sub = df[df["pop_stratum"] == label].copy()

        eta_vals = sub[PRED].to_numpy()

        theta = fit_theta_star(sub)
        pct = weak_percentile(eta_vals, theta) if np.isfinite(theta) else np.nan

        auc, k_used = cv_auc_oof_adaptive(sub)

        rows.append(
            {
                "Population Stratum": f"{label} polities",
                "n": int(len(sub)),
                "θ*": round(float(theta), 2) if np.isfinite(theta) else np.nan,
                "Percentile": (f"{pct:.1f}%" if np.isfinite(pct) else "NA"),
                "AUC": round(float(auc), 2) if np.isfinite(auc) else np.nan,
                "k": int(k_used),
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    write_markdown(out)

    print("[make_table_S6_population_strata] Wrote:")
    print(f"  {OUT_CSV}")
    print(f"  {OUT_MD}")


if __name__ == "__main__":
    main()
