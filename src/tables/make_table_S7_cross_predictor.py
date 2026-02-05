#!/usr/bin/env python3
"""
SI Table S7 — Cross-predictor convergence of instability thresholds (η-ratio vs SPC1)

What this table is:
- A shared-sample convergence check across two independent internal-load representations.

Outputs (locked):
- tables/S7_cross_predictor_convergence.csv
- tables/S7_cross_predictor_convergence.md

Stats (reviewer-defensible):
- Pearson r between predictors on shared sample
- AUC (5-fold stratified CV, OOF) for each predictor
- High-stress overlap at >=90th percentile within each predictor (shared sample)
- Percentile convergence permutation test:
    * Fit θ* for each predictor on shared sample via full-sample logistic boundary (P=0.5): θ* = -β0/β1
    * Convert each θ* to weak percentile in its predictor distribution (shared sample)
    * Observed separation = |p_eta - p_spc1|
    * Null: draw random thresholds uniformly from each predictor's empirical range, convert to weak percentiles,
            compute separation; p = Pr(separation_null <= separation_obs)

Notes:
- This permutation p-value is a convergence robustness result and belongs in S7 (or as an adjacent robustness item),
  not in S2.
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

DATA_DIR = REPO_ROOT / "data" / "final"

SESHAT_FILE = DATA_DIR / "seshat_EI_collapse_panel_w100.csv"

OUT_DIR = REPO_ROOT / "output" / "tables"
OUT_CSV = OUT_DIR / "S7_cross_predictor_convergence.csv"
OUT_MD = OUT_DIR / "S7_cross_predictor_convergence.md"


# -----------------------------
# Config
# -----------------------------
ETA = "eta_ratio"
SPC = "SPC1"
YCOL = "collapse_next_100y"

# Optional merge keys (only used if no pre-merged shared file exists)
# We'll try inner-merge on the best available common keys.
PREFERRED_KEYS = [
    ["eim_slug", "window_start"],     # if present
    ["eim_slug", "t_start"],          # if present
    ["eim_slug", "year"],             # if present
    ["eim_slug"],                     # last resort
]

# CV settings (match other tables)
N_SPLITS = 5
RANDOM_STATE = 42

LOGIT_KW = dict(
    solver="lbfgs",
    C=1e6,
    max_iter=10000,
)

# Permutation test settings
N_PERM = 100000
PERM_SEED = 42

# High-stress definition
HIGH_PCTL = 90.0


# -----------------------------
# Helpers
# -----------------------------
def require_cols(df: pd.DataFrame, cols: list[str], src: Path | str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {src}: {missing}. Found: {list(df.columns)}")


def weak_percentile(values: np.ndarray, theta: float) -> float:
    """Weak percentile = Pr(x <= theta) * 100."""
    return float(100.0 * np.mean(values <= theta))


def find_premerged_shared() -> Path | None:
    """
    Prefer a pre-merged shared file if present: any CSV in data/final containing both predictors.
    """
    if not DATA_DIR.exists():
        return None

    candidates = []
    for p in DATA_DIR.glob("*.csv"):
        name = p.name.lower()
        if ("shared" in name or "merged" in name or "cross" in name) and ("w100" in name or "100" in name):
            try:
                df = pd.read_csv(p, nrows=5)
            except Exception:
                continue
            if ETA in df.columns and SPC in df.columns and YCOL in df.columns:
                candidates.append(p)

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        # Choose newest
        return sorted(candidates, key=lambda x: x.stat().st_mtime, reverse=True)[0]
    return None


def load_shared_sample() -> tuple[pd.DataFrame, str]:
    """
    Load a shared sample with columns [eta_ratio, SPC1, collapse_next_100y].
    Strategy:
      1) Use pre-merged file if found.
      2) Otherwise attempt to merge SPC1 panel + Seshat EI panel on best available keys.
    """
    pre = find_premerged_shared()
    if pre is not None:
        df = pd.read_csv(pre)
        require_cols(df, [ETA, SPC, YCOL], pre)
        df = df.dropna(subset=[ETA, SPC, YCOL]).copy()
        df[YCOL] = df[YCOL].astype(int)
        return df, f"premerged:{pre.name}"

    # Need to build shared sample
    if not SESHAT_FILE.exists():
        raise FileNotFoundError(f"Missing Seshat EI panel: {SESHAT_FILE}")

    ses = pd.read_csv(SESHAT_FILE)
    require_cols(ses, [ETA, YCOL], SESHAT_FILE)

    # Find an SPC1 panel file
    spc_candidates = []
    for p in DATA_DIR.glob("*.csv"):
        n = p.name.lower()
        if ("spc1" in n) and ("collapse" in n) and ("panel" in n):
            spc_candidates.append(p)
    if not spc_candidates:
        raise FileNotFoundError("Could not find an SPC1 collapse panel in data/final (name containing 'SPC1', 'collapse', 'panel').")

    spc_path = sorted(spc_candidates, key=lambda x: x.stat().st_mtime, reverse=True)[0]
    spc = pd.read_csv(spc_path)
    require_cols(spc, [SPC, YCOL], spc_path)

    # Choose best merge keys
    for keys in PREFERRED_KEYS:
        if all(k in ses.columns for k in keys) and all(k in spc.columns for k in keys):
            merged = ses.merge(spc, on=keys, how="inner", suffixes=("_eta", "_spc"))
            # Prefer using one y; if both exist, ensure consistency
            if f"{YCOL}_eta" in merged.columns and f"{YCOL}_spc" in merged.columns:
                # Keep only consistent labels
                merged = merged[merged[f"{YCOL}_eta"].astype(int) == merged[f"{YCOL}_spc"].astype(int)].copy()
                merged[YCOL] = merged[f"{YCOL}_eta"].astype(int)
            elif YCOL in merged.columns:
                merged[YCOL] = merged[YCOL].astype(int)
            else:
                continue

            # Ensure predictors exist (merge may rename)
            if ETA not in merged.columns:
                # if renamed
                if f"{ETA}_eta" in merged.columns:
                    merged[ETA] = merged[f"{ETA}_eta"]
            if SPC not in merged.columns:
                if f"{SPC}_spc" in merged.columns:
                    merged[SPC] = merged[f"{SPC}_spc"]

            if ETA in merged.columns and SPC in merged.columns and YCOL in merged.columns:
                merged = merged.dropna(subset=[ETA, SPC, YCOL]).copy()
                merged[YCOL] = merged[YCOL].astype(int)
                return merged, f"merged:{SESHAT_FILE.name}+{spc_path.name} on {keys}"

    raise ValueError(
        "Could not construct a shared sample (no compatible merge keys found).\n"
        "Either create a pre-merged shared file in data/final with columns eta_ratio, SPC1, collapse_next_100y,\n"
        "or ensure both panels share at least 'eim_slug' (and ideally a window/time identifier)."
    )


def fit_theta_star_full(df: pd.DataFrame, pred: str) -> float:
    X = df[[pred]].to_numpy()
    y = df[YCOL].astype(int).to_numpy()

    if len(np.unique(y)) < 2:
        return np.nan

    clf = LogisticRegression(**LOGIT_KW)
    clf.fit(X, y)
    b0 = float(clf.intercept_[0])
    b1 = float(clf.coef_[0][0])
    if b1 == 0:
        return np.nan
    return float(-b0 / b1)


def cv_auc_oof(df: pd.DataFrame, pred: str) -> float:
    X = df[[pred]].to_numpy()
    y = df[YCOL].astype(int).to_numpy()

    if len(np.unique(y)) < 2:
        return np.nan

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(y), dtype=float)

    for tr, te in skf.split(X, y):
        y_tr = y[tr]
        if len(np.unique(y_tr)) < 2:
            return np.nan
        clf = LogisticRegression(**LOGIT_KW)
        clf.fit(X[tr], y_tr)
        oof[te] = clf.predict_proba(X[te])[:, 1]

    return float(roc_auc_score(y, oof))


def permutation_pvalue_percentile_convergence(vals_a: np.ndarray, vals_b: np.ndarray, p_obs: float) -> float:
    """
    One-tailed p-value for unusually small percentile separation under random thresholds.
    vals_a, vals_b are predictor arrays (shared sample).
    p_obs is observed abs percentile separation (in percent units, 0..100).
    """
    rng = np.random.default_rng(PERM_SEED)

    a_min, a_max = float(np.min(vals_a)), float(np.max(vals_a))
    b_min, b_max = float(np.min(vals_b)), float(np.max(vals_b))

    # Draw random thresholds uniformly over empirical ranges
    ta = rng.uniform(a_min, a_max, size=N_PERM)
    tb = rng.uniform(b_min, b_max, size=N_PERM)

    # Convert each threshold to weak percentile (0..100)
    pa = 100.0 * (vals_a.reshape(1, -1) <= ta.reshape(-1, 1)).mean(axis=1)
    pb = 100.0 * (vals_b.reshape(1, -1) <= tb.reshape(-1, 1)).mean(axis=1)

    sep = np.abs(pa - pb)

    # One-tailed: probability of being as close or closer than observed
    return float(np.mean(sep <= p_obs))


def write_kv_csv(kv: list[tuple[str, str]]) -> None:
    out = pd.DataFrame(kv, columns=["Metric", "Value"])
    out.to_csv(OUT_CSV, index=False)


def write_md(kv: list[tuple[str, str]]) -> None:
    lines = []
    lines.append("Table S7. Cross-predictor convergence of instability thresholds (η-ratio vs SPC1)")
    lines.append("---------------------------------------------------------------------------")
    lines.append("Metric                                           | Value")
    lines.append("---------------------------------------------------------------------------")
    for k, v in kv:
        lines.append(f"{k:<48} | {v}")
    lines.append("---------------------------------------------------------------------------")
    lines.append("Notes:")
    lines.append(f"- Computed on the shared sample containing both predictors (η-ratio and SPC1) with {YCOL}.")
    lines.append(f"- High stress is defined as ≥{HIGH_PCTL:.0f}th percentile within each predictor (shared sample).")
    lines.append(f"- AUC values are 5-fold stratified cross-validated (OOF) AUC (random_state={RANDOM_STATE}).")
    lines.append("- Permutation p-value tests whether observed percentile convergence is unusually close under random thresholds.")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df, source = load_shared_sample()
    require_cols(df, [ETA, SPC, YCOL], source)

    # Shared sample size
    n = int(len(df))

    # Correlation
    r = float(np.corrcoef(df[ETA].to_numpy(), df[SPC].to_numpy())[0, 1])

    # AUCs (OOF CV)
    auc_eta = cv_auc_oof(df, ETA)
    auc_spc = cv_auc_oof(df, SPC)

    # θ* and percentiles (shared sample distributions)
    theta_eta = fit_theta_star_full(df, ETA)
    theta_spc = fit_theta_star_full(df, SPC)

    p_eta = weak_percentile(df[ETA].to_numpy(), theta_eta) if np.isfinite(theta_eta) else np.nan
    p_spc = weak_percentile(df[SPC].to_numpy(), theta_spc) if np.isfinite(theta_spc) else np.nan
    p_sep = float(abs(p_eta - p_spc)) if np.isfinite(p_eta) and np.isfinite(p_spc) else np.nan

    # High-stress overlap at >=90th pct within each predictor
    eta_cut = np.percentile(df[ETA].to_numpy(), HIGH_PCTL)
    spc_cut = np.percentile(df[SPC].to_numpy(), HIGH_PCTL)

    A = set(df.index[df[ETA] >= eta_cut].tolist())
    B = set(df.index[df[SPC] >= spc_cut].tolist())
    inter = A.intersection(B)

    nA, nB, nI = len(A), len(B), len(inter)
    overlap_pct = (100.0 * nI / min(nA, nB)) if min(nA, nB) > 0 else np.nan

    # Permutation p-value for percentile convergence (optional but useful)
    p_perm = (
        permutation_pvalue_percentile_convergence(df[ETA].to_numpy(), df[SPC].to_numpy(), p_sep)
        if np.isfinite(p_sep)
        else np.nan
    )

    kv = [
        ("Shared-sample n (both predictors available)", str(n)),
        ("Correlation r (η-ratio vs SPC1)", f"{r:.3f}"),
        ("θ* (η-ratio, shared sample)", f"{theta_eta:.3f}" if np.isfinite(theta_eta) else "NA"),
        ("θ* percentile (η-ratio, shared sample)", f"{p_eta:.2f}%" if np.isfinite(p_eta) else "NA"),
        ("θ* (SPC1, shared sample)", f"{theta_spc:.3f}" if np.isfinite(theta_spc) else "NA"),
        ("θ* percentile (SPC1, shared sample)", f"{p_spc:.2f}%" if np.isfinite(p_spc) else "NA"),
        ("Abs percentile separation |pη − pSPC|", f"{p_sep:.2f}" if np.isfinite(p_sep) else "NA"),
        (f"High-stress set size (η ≥ {HIGH_PCTL:.0f}th pct)", str(nA)),
        (f"High-stress set size (SPC1 ≥ {HIGH_PCTL:.0f}th pct)", str(nB)),
        ("Joint high-stress cases |A ∩ B|", str(nI)),
        ("Overlap (% of smaller high-stress set)", f"{overlap_pct:.1f}%" if np.isfinite(overlap_pct) else "NA"),
        ("AUC (η-ratio, shared sample; 5-fold CV)", f"{auc_eta:.3f}" if np.isfinite(auc_eta) else "NA"),
        ("AUC (SPC1, shared sample; 5-fold CV)", f"{auc_spc:.3f}" if np.isfinite(auc_spc) else "NA"),
        (f"Permutation p (percentile convergence; N={N_PERM})", f"{p_perm:.3f}" if np.isfinite(p_perm) else "NA"),
        ("Data source", source),
    ]

    write_kv_csv(kv)
    write_md(kv)

    print("[make_table_S7_cross_predictor] Wrote:")
    print(f"  {OUT_CSV}")
    print(f"  {OUT_MD}")
    print(f"[make_table_S7_cross_predictor] Source: {source}")


if __name__ == "__main__":
    main()
