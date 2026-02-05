#!/usr/bin/env python3
"""
SI Table S5 — Robustness of η-ratio thresholds across temporal horizons (50y, 100y, 150y)

Definitions (locked):
- θ* = predictor value where the fitted logistic assigns P(collapse)=0.5
      computed from full-sample logistic: θ* = -β0/β1
- Percentile = weak percentile of θ* in the full η distribution: Pr(η <= θ*) * 100
- AUC = 5-fold stratified cross-validated AUC using out-of-fold probabilities (OOF)

Inputs:
- data/final/seshat_EI_collapse_panel_w100.csv (preferred)
  or auto-discovered Seshat collapse panel under data/final.

Outputs:
- tables/S5_eta_horizons.csv
- tables/S5_eta_horizons.md
"""

from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


# -----------------------------
# Paths
# -----------------------------
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]  # .../collapse_thresholds/src/tables -> repo root

DATA_DIR = REPO_ROOT / "data" / "final"
PREFERRED = DATA_DIR / "seshat_EI_collapse_panel_w100.csv"

OUT_DIR = REPO_ROOT / "output" / "tables"
OUT_CSV = OUT_DIR / "S5_eta_horizons.csv"
OUT_MD = OUT_DIR / "S5_eta_horizons.md"


# -----------------------------
# Config
# -----------------------------
PRED = "eta_ratio"
HORIZONS = [
    ("η-ratio (50-year horizon)", "collapse_next_50y"),
    ("η-ratio (100-year horizon)", "collapse_next_100y"),
    ("η-ratio (150-year horizon)", "collapse_next_150y"),
]

N_SPLITS = 5
RANDOM_STATE = 42

LOGIT_KW = dict(
    solver="lbfgs",
    C=1e6,          # effectively unregularised
    max_iter=10000,
)


# -----------------------------
# Helpers
# -----------------------------
def find_seshat_panel() -> Path:
    """Locate the Seshat EI collapse panel CSV."""
    if PREFERRED.exists():
        return PREFERRED

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Missing data directory: {DATA_DIR}")

    candidates = []
    for p in DATA_DIR.glob("*.csv"):
        name = p.name.lower()
        if ("seshat" in name) and ("collapse" in name) and ("panel" in name):
            candidates.append(p)

    if not candidates:
        raise FileNotFoundError(
            "Could not find Seshat collapse panel in data/final.\n"
            "Expected something like 'seshat_EI_collapse_panel_w100.csv'."
        )

    # Prefer EI + w100 if present, else newest
    def score(path: Path) -> tuple[int, int, float]:
        n = path.name.lower()
        return (
            int("ei" in n),
            int("w100" in n),
            path.stat().st_mtime,
        )

    return sorted(candidates, key=score, reverse=True)[0]


def require_cols(df: pd.DataFrame, cols: list[str], src: Path) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {src}: {missing}. Available: {list(df.columns)}")


def weak_percentile(values: np.ndarray, theta: float) -> float:
    """Weak percentile = Pr(x <= theta) * 100."""
    return float(100.0 * np.mean(values <= theta))


def fit_theta_star_full(df: pd.DataFrame, ycol: str) -> float:
    """Fit logistic on full sample and return θ* = -β0/β1."""
    X = df[[PRED]].to_numpy()
    y = df[ycol].astype(int).to_numpy()

    clf = LogisticRegression(**LOGIT_KW)
    clf.fit(X, y)

    b0 = float(clf.intercept_[0])
    b1 = float(clf.coef_[0][0])

    if b1 == 0:
        return np.nan
    return float(-b0 / b1)


def cv_auc_oof(df: pd.DataFrame, ycol: str) -> float:
    """5-fold stratified CV AUC using out-of-fold predictions."""
    X = df[[PRED]].to_numpy()
    y = df[ycol].astype(int).to_numpy()

    if len(np.unique(y)) < 2:
        return np.nan

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(y), dtype=float)

    for tr, te in skf.split(X, y):
        clf = LogisticRegression(**LOGIT_KW)
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]

    return float(roc_auc_score(y, oof))


def write_markdown(out: pd.DataFrame) -> None:
    lines = []
    lines.append("Table S5. Threshold robustness across alternative collapse horizons (50y, 100y, 150y) using η-ratio")
    lines.append("-------------------------------------------------------------------------------")
    lines.append("Horizon (Model)                  |  n  |   θ*   | Percentile |  AUC")
    lines.append("-------------------------------------------------------------------------------")
    for _, r in out.iterrows():
        auc = r["AUC"]
        auc_str = f"{auc:.2f}" if pd.notna(auc) else "NA"
        lines.append(
            f"{r['Horizon (Model)']:<30} | {int(r['n']):>3} | {r['θ*']:>6.2f} | {r['Percentile']:>9} | {auc_str:>5}"
        )
    lines.append("-------------------------------------------------------------------------------")
    lines.append("Notes:")
    lines.append("- θ* is the decision boundary where P(collapse)=0.5, computed from a full-sample logistic fit (θ* = -β0/β1).")
    lines.append("- Percentile is computed relative to the full η-ratio distribution (weak percentile: Pr(η ≤ θ*)).")
    lines.append(f"- AUC is mean 5-fold stratified cross-validated (OOF) AUC (random_state={RANDOM_STATE}).")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    seshat_path = find_seshat_panel()
    df = pd.read_csv(seshat_path)

    require_cols(df, [PRED], seshat_path)

    df = df.dropna(subset=[PRED]).copy()
    eta_vals_full = df[PRED].to_numpy()

    rows = []
    for label, ycol in HORIZONS:
        require_cols(df, [ycol], seshat_path)

        sub = df.dropna(subset=[ycol]).copy()
        sub[ycol] = sub[ycol].astype(int)

        theta = fit_theta_star_full(sub, ycol=ycol)
        pct = weak_percentile(eta_vals_full, theta) if np.isfinite(theta) else np.nan
        auc = cv_auc_oof(sub, ycol=ycol)

        rows.append(
            {
                "Horizon (Model)": label,
                "n": int(len(sub)),
                "θ*": round(float(theta), 2) if np.isfinite(theta) else np.nan,
                "Percentile": (f"{pct:.1f}%" if np.isfinite(pct) else "NA"),
                "AUC": round(float(auc), 2) if np.isfinite(auc) else np.nan,
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    write_markdown(out)

    print("[make_table_S5_eta_horizons] Input:")
    print(f"  {seshat_path}")
    print("[make_table_S5_eta_horizons] Wrote:")
    print(f"  {OUT_CSV}")
    print(f"  {OUT_MD}")


if __name__ == "__main__":
    main()
