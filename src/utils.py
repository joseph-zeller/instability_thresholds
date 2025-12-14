# src/utils.py
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
DATA_FINAL = ROOT / "data" / "final"
CONFIG_DIR = ROOT / "config"
RESULTS_THRESHOLDS = ROOT / "results" / "thresholds"


def load_panel(window: int) -> pd.DataFrame:
    """Load w50/w100/w150 panel."""
    fname = DATA_FINAL / f"seshat_EI_collapse_panel_w{window}.csv"
    if not fname.exists():
        raise FileNotFoundError(f"Panel not found: {fname}")
    return pd.read_csv(fname)


def load_yaml(name: str) -> dict:
    path = CONFIG_DIR / name
    with open(path, "r") as f:
        return yaml.safe_load(f)


def logistic_threshold(
    df: pd.DataFrame,
    predictor: str,
    outcome: str,
) -> dict:
    """
    Fit univariate logistic regression: outcome ~ predictor.
    Returns dict with alpha, beta, eta_star, percentile, auc, n.
    """

    # Drop rows with NA in predictor or outcome
    data = df[[predictor, outcome]].dropna()
    n = len(data)
    if n < 10:
        raise ValueError(f"Too few observations (n={n}) to fit model.")

    X = sm.add_constant(data[predictor].values)
    y = data[outcome].values

    model = sm.Logit(y, X)
    res = model.fit(disp=False)

    alpha, beta = res.params  # const, eta_ratio
    eta_star = -alpha / beta

    # Percentile of eta_star within predictor distribution
    x = data[predictor].values
    percentile = np.mean(x <= eta_star) * 100.0

    # AUC
    y_pred = res.predict(X)
    auc = roc_auc_score(y, y_pred)

    return {
        "alpha": alpha,
        "beta": beta,
        "eta_star": eta_star,
        "percentile": percentile,
        "auc": auc,
        "n": n,
    }


def apply_transform(series: pd.Series, method: str) -> pd.Series:
    """
    Transform eta_ratio for robustness tests.
    method ∈ {"raw", "zscore", "rint", "minmax"}.
    """
    x = series.dropna().astype(float)

    if method == "raw":
        return series

    if method == "zscore":
        mu = x.mean()
        sigma = x.std(ddof=0)
        z = (series - mu) / sigma
        return z

    if method == "rint":
        # rank-based inverse normal transform
        ranks = series.rank(method="average")
        p = (ranks - 0.5) / len(ranks)
        from scipy.stats import norm
        return norm.ppf(p)

    if method == "minmax":
        xmin = x.min()
        xmax = x.max()
        scaled = (series - xmin) / (xmax - xmin)
        return scaled

    raise ValueError(f"Unknown transform method: {method}")
