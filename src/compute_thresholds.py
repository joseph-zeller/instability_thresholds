#!/usr/bin/env python
"""
compute_thresholds.py

Reproduce instability thresholds for:

  1) Seshat EI panel: eta_ratio → collapse_next_100y
  2) SPC1 panel: SPC1 → collapse_next_100y

Outputs (written to results/thresholds/):

  - threshold_cv_folds_w100.csv
  - threshold_cv_summary_w100.csv
  - threshold_simple_fullsample_w100.csv

Notes:
- Percentiles are reported on the 0–100 scale (e.g., 95.238 = 95.238th percentile).
- SPC1 uses stratified 5-fold CV (n sufficient).
- For small-n datasets (e.g., Seshat n=21), the manuscript/SI emphasise the
  cross-validated percentile threshold; the full-sample fit is reported for
  diagnostics only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


# ---------------------
# Config
# ---------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "final"
OUT_DIR = REPO_ROOT / "results" / "thresholds"

SESHAT_FILE = DATA_DIR / "seshat_EI_collapse_panel_w100.csv"
SPC1_FILE = DATA_DIR / "SPC1_collapse_panel_w100.csv"

RANDOM_STATE = 42
N_SPLITS = 5


@dataclass
class DatasetSpec:
    name: str
    path: Path
    predictor: str
    outcome: str


SPECS: List[DatasetSpec] = [
    DatasetSpec(
        name="Seshat_eta_ratio_w100",
        path=SESHAT_FILE,
        predictor="eta_ratio",
        outcome="collapse_next_100y",
    ),
    DatasetSpec(
        name="SPC1_w100",
        path=SPC1_FILE,
        predictor="SPC1",
        outcome="collapse_next_100y",
    ),
]


# ---------------------
# Helpers
# ---------------------

def _ensure_outdir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_xy(spec: DatasetSpec) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load predictor X and outcome y from a panel.
    Returns (x_full, X, y) where:
      - x_full is the predictor vector used for percentile calculations (NaNs dropped)
      - X is the 2D sklearn design matrix
      - y is the binary label array
    """
    if not spec.path.exists():
        raise FileNotFoundError(f"Missing input file: {spec.path}")

    df = pd.read_csv(spec.path)

    if spec.predictor not in df.columns:
        raise KeyError(f"Predictor column '{spec.predictor}' not found in {spec.path.name}")
    if spec.outcome not in df.columns:
        raise KeyError(f"Outcome column '{spec.outcome}' not found in {spec.path.name}")

    d = df[[spec.predictor, spec.outcome]].dropna().copy()
    d[spec.outcome] = d[spec.outcome].astype(int)
    d[spec.predictor] = d[spec.predictor].astype(float)

    x_full = d[spec.predictor].values
    X = d[[spec.predictor]].values
    y = d[spec.outcome].values

    return x_full, X, y


def _threshold_from_model(intercept: float, coef: float) -> float:
    """Return theta = -b0/b1 for P(collapse)=0.5, or NaN if coef ~ 0."""
    if abs(coef) < 1e-8:
        return float("nan")
    return float(-intercept / coef)


def _percentile_0_100(x_full: np.ndarray, threshold: float) -> float:
    """Empirical percentile position (0–100) using weak inequality."""
    if np.isnan(threshold):
        return float("nan")
    return float(np.mean(x_full <= threshold) * 100.0)


def _compute_threshold_stats_cv(
    dataset_name: str,
    predictor_col: str,
    x_full: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
) -> Dict[str, object]:
    """
    Stratified K-fold CV threshold estimation.
    Threshold θ is defined as -b0/b1 where P(collapse)=0.5.
    Percentile computed against the full empirical distribution of the predictor.
    """
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    fold_results: List[Dict[str, object]] = []
    thetas: List[float] = []
    aucs: List[float] = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        model = LogisticRegression(solver="lbfgs", max_iter=1000)
        model.fit(X_train, y_train)

        intercept = float(model.intercept_[0])
        coef = float(model.coef_[0, 0])

        theta = _threshold_from_model(intercept, coef)

        # AUC on held-out fold (undefined if fold has a single class)
        y_proba = model.predict_proba(X_test)[:, 1]
        if len(np.unique(y_test)) == 1:
            auc = float("nan")
        else:
            auc = float(roc_auc_score(y_test, y_proba))

        fold_results.append(
            {
                "dataset": dataset_name,
                "predictor": predictor_col,
                "fold": fold_idx,
                "train_size": int(len(train_idx)),
                "test_size": int(len(test_idx)),
                "intercept": intercept,
                "coef": coef,
                "theta": theta,
                "auc": auc,
            }
        )

        thetas.append(theta)
        aucs.append(auc)

    theta_arr = np.array(thetas, dtype=float)
    auc_arr = np.array(aucs, dtype=float)

    mean_theta = float(np.nanmean(theta_arr))
    std_theta = float(np.nanstd(theta_arr, ddof=1)) if np.sum(~np.isnan(theta_arr)) > 1 else float("nan")

    thr_pct = _percentile_0_100(x_full, mean_theta)

    mean_auc = float(np.nanmean(auc_arr))
    std_auc = float(np.nanstd(auc_arr, ddof=1)) if np.sum(~np.isnan(auc_arr)) > 1 else float("nan")

    summary = {
        "dataset": dataset_name,
        "predictor": predictor_col,
        "n": int(len(y)),
        "pos": int(np.sum(y == 1)),
        "neg": int(np.sum(y == 0)),
        "theta_mean": mean_theta,
        "theta_sd": std_theta,
        "threshold_percentile": thr_pct,
        "auc_mean": mean_auc,
        "auc_sd": std_auc,
    }

    return {"folds": fold_results, "summary": summary}


def _compute_simple_threshold(
    dataset_name: str,
    predictor_col: str,
    x_full: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
) -> Dict[str, object]:
    """
    Fit a single logistic model on the full sample and compute θ and percentile.
    This is included for diagnostics; manuscript/SI emphasise CV percentile thresholds
    for small-n datasets (e.g., Seshat).
    """
    model = LogisticRegression(solver="lbfgs", max_iter=1000)
    model.fit(X, y)

    intercept = float(model.intercept_[0])
    coef = float(model.coef_[0, 0])

    theta = _threshold_from_model(intercept, coef)
    thr_pct = _percentile_0_100(x_full, theta)

    # AUC on full sample (informative only)
    y_proba = model.predict_proba(X)[:, 1]
    auc = float(roc_auc_score(y, y_proba)) if len(np.unique(y)) > 1 else float("nan")

    simple = {
        "dataset": dataset_name,
        "predictor": predictor_col,
        "n": int(len(y)),
        "pos": int(np.sum(y == 1)),
        "neg": int(np.sum(y == 0)),
        "intercept": intercept,
        "coef": coef,
        "theta": theta,
        "threshold_percentile": thr_pct,
        "auc_fullsample": auc,
    }

    print(f"\n=== {dataset_name}: simple (full-sample) model; DIAGNOSTIC ONLY ===")
    print("  Note: For small-n datasets (e.g., Seshat n=21), the manuscript/SI report")
    print("  cross-validated percentile thresholds (see CV summary), not this full-sample fit.")
    print(f"  Intercept: {intercept:.4f}")
    print(f"  Coefficient ({predictor_col}): {coef:.4f}")
    if np.isnan(theta):
        print("  Threshold θ (p=0.5): NaN (coef ~ 0)")
    else:
        print(f"  Threshold θ (p=0.5): {theta:.4f}")
    print(f"  Threshold percentile: {thr_pct:.4f}")

    return simple


def _print_cv_summary(summary: Dict[str, object]) -> None:
    """Print the cross-validated threshold summary in the same units used in the paper."""
    print(f"\n=== {summary['dataset']}: cross-validated (reported) ===")
    print(f"  n = {summary['n']} | pos = {summary['pos']} | neg = {summary['neg']}")
    print(f"  Mean θ (p=0.5): {summary['theta_mean']:.4f} ± {summary['theta_sd']:.4f}")
    print(f"  Threshold percentile: {summary['threshold_percentile']:.4f}")
    print(f"  AUC (mean±sd): {summary['auc_mean']:.4f} ± {summary['auc_sd']:.4f}")


# ---------------------
# Main
# ---------------------

def main() -> None:
    _ensure_outdir()

    all_fold_rows: List[Dict[str, object]] = []
    all_summary_rows: List[Dict[str, object]] = []
    all_simple_rows: List[Dict[str, object]] = []

    for spec in SPECS:
        print("\n" + "=" * 72)
        print(f"Dataset: {spec.name}")
        print(f"File:    {spec.path}")
        print(f"X:       {spec.predictor}")
        print(f"y:       {spec.outcome}")
        print("=" * 72)

        x_full, X, y = _load_xy(spec)

        print(
            f"[LOAD] Rows after dropna: {len(y)} | positives: {int(np.sum(y==1))} | negatives: {int(np.sum(y==0))}"
        )
        print(f"[PRED] min={np.min(x_full):.6f} max={np.max(x_full):.6f} mean={np.mean(x_full):.6f}")

        # CV summary + folds (this is what we treat as 'reported' for small-n)
        cv = _compute_threshold_stats_cv(spec.name, spec.predictor, x_full, X, y)
        all_fold_rows.extend(cv["folds"])
        all_summary_rows.append(cv["summary"])
        _print_cv_summary(cv["summary"])

        # Full-sample model (diagnostic)
        simple = _compute_simple_threshold(spec.name, spec.predictor, x_full, X, y)
        all_simple_rows.append(simple)

    folds_df = pd.DataFrame(all_fold_rows)
    summary_df = pd.DataFrame(all_summary_rows)
    simple_df = pd.DataFrame(all_simple_rows)

    folds_out = OUT_DIR / "threshold_cv_folds_w100.csv"
    summary_out = OUT_DIR / "threshold_cv_summary_w100.csv"
    simple_out = OUT_DIR / "threshold_simple_fullsample_w100.csv"

    folds_df.to_csv(folds_out, index=False)
    summary_df.to_csv(summary_out, index=False)
    simple_df.to_csv(simple_out, index=False)

    print("\n" + "-" * 72)
    print("Wrote:")
    print(f"  {folds_out.relative_to(REPO_ROOT)}")
    print(f"  {summary_out.relative_to(REPO_ROOT)}")
    print(f"  {simple_out.relative_to(REPO_ROOT)}")
    print("-" * 72)


if __name__ == "__main__":
    main()
