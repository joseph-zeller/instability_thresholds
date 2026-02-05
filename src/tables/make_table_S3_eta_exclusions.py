#!/usr/bin/env python3
"""
SI Table S3 — Robustness of η-ratio instability threshold to exclusion of influential/disputed cases.

Core rule:
- θ* is NOT re-estimated. It is fixed to the canonical η-ratio θ* reported in Table S2.
- Percentile is computed relative to the FULL η-ratio distribution (weak percentile: Pr(x <= θ*)).

Inputs (expected):
- tables/S2_core_thresholds.csv
- data/final/seshat_EI_collapse_panel_w100.csv

Outputs (locked):
- tables/S3_eta_exclusions.csv
- tables/S3_eta_exclusions.md

This script is intentionally self-contained and reproducible.
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

S2_FILE = REPO_ROOT / "tables" / "S2_core_thresholds.csv"
DATA_FILE = REPO_ROOT / "data" / "final" / "seshat_EI_collapse_panel_w100.csv"

OUT_DIR = REPO_ROOT / "output" / "tables"
OUT_CSV = OUT_DIR / "S3_eta_exclusions.csv"
OUT_MD = OUT_DIR / "S3_eta_exclusions.md"


# -----------------------------
# Configuration
# -----------------------------
PRED = "eta_ratio"
YCOL = "collapse_next_100y"
SLUG = "eim_slug"

# CV config
N_SPLITS = 5
RANDOM_STATE = 42


# -----------------------------
# Helpers
# -----------------------------
def load_eta_star_from_s2() -> float:
    """Load canonical η* from S2_core_thresholds.csv (robust matching)."""
    if not S2_FILE.exists():
        raise FileNotFoundError(f"Missing S2 table: {S2_FILE}")

    s2 = pd.read_csv(S2_FILE)

    if "Dataset" not in s2.columns or "θ*" not in s2.columns:
        raise ValueError(
            f"S2 table must contain columns 'Dataset' and 'θ*'. Found: {list(s2.columns)}"
        )

    ds = s2["Dataset"].astype(str)

    # Robust match: accept Greek η, 'ratio', or 'Seshat'
    mask = (
        ds.str.contains("Seshat", case=False, na=False)
        | ds.str.contains("η", na=False)
        | ds.str.contains("ratio", case=False, na=False)
        | ds.str.contains("eta", case=False, na=False)
    )

    row = s2[mask]
    if row.empty:
        raise ValueError(
            f"Could not identify η-ratio row in {S2_FILE}. "
            f"Available Dataset values: {s2['Dataset'].tolist()}"
        )

    eta_star = float(row.iloc[0]["θ*"])
    return eta_star


def require_cols(df: pd.DataFrame, cols: list[str], src: Path) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {src}: {missing}. Available: {list(df.columns)}")


def weak_percentile(values: np.ndarray, theta: float) -> float:
    """Weak percentile = Pr(x <= theta) * 100."""
    return float(100.0 * np.mean(values <= theta))


def cv_auc_5fold(df: pd.DataFrame) -> float:
    """Out-of-fold AUC using 5-fold stratified CV."""
    X = df[[PRED]].to_numpy()
    y = df[YCOL].astype(int).to_numpy()

    if len(np.unique(y)) < 2:
        return np.nan

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(y), dtype=float)

    for tr_idx, te_idx in skf.split(X, y):
        model = LogisticRegression(
            solver="lbfgs",
            C=1e6,          # effectively unregularised
            max_iter=10000,
        )
        model.fit(X[tr_idx], y[tr_idx])
        oof[te_idx] = model.predict_proba(X[te_idx])[:, 1]

    return float(roc_auc_score(y, oof))


def write_markdown_table(df: pd.DataFrame) -> None:
    lines = []
    lines.append("Table S3. Robustness of the η-ratio instability threshold to exclusion of influential/disputed cases")
    lines.append("-------------------------------------------------------------------------------")
    lines.append("Model Specification                          |  n  |   θ*   | Percentile |   AUC")
    lines.append("-------------------------------------------------------------------------------")
    for _, r in df.iterrows():
        auc = r["AUC"]
        auc_str = f"{auc:.2f}" if pd.notna(auc) else "NA"
        lines.append(
            f"{r['Model Specification']:<42} | {int(r['n']):>3} | {r['θ*']:>6.2f} | {r['Percentile']:>10} | {auc_str:>5}"
        )
    lines.append("-------------------------------------------------------------------------------")
    lines.append("Notes:")
    lines.append("- θ* is fixed to the canonical η-ratio threshold reported in Table S2.")
    lines.append("- Percentile is computed against the full η-ratio distribution (weak percentile: Pr(x ≤ θ*)).")
    lines.append(f"- AUC is mean 5-fold stratified cross-validated (random_state={RANDOM_STATE}).")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    eta_star = load_eta_star_from_s2()
    print(f"[make_table_S3_eta_exclusions] Using canonical η* from S2: {eta_star:.6f}")

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Missing Seshat panel file: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)
    require_cols(df, [PRED, YCOL, SLUG], DATA_FILE)

    df = df.dropna(subset=[PRED, YCOL, SLUG]).copy()
    df[YCOL] = df[YCOL].astype(int)

    # Full-distribution percentile (held fixed across exclusion scenarios)
    full_values = df[PRED].to_numpy()
    pct = weak_percentile(full_values, eta_star)

    # Identify candidate influential cases from the data
    max_eta_slug = str(df.loc[df[PRED].idxmax(), SLUG])
    near_eta_slug = str(df.loc[(df[PRED] - eta_star).abs().idxmin(), SLUG])

    # Case A: you previously referenced "New Egyptian Kingdom"
    # We attempt to match it robustly. If not found, we warn and exclude nothing.
    slugs = df[SLUG].astype(str)
    caseA_candidates = ["new_egyptian_kingdom", "new-egyptian-kingdom", "egyptian_kingdom"]
    caseA_found = []
    for tok in caseA_candidates:
        hits = slugs[slugs.str.contains(tok, case=False, na=False)].unique().tolist()
        caseA_found.extend(hits)
    caseA_found = sorted(set(caseA_found))
    if not caseA_found:
        print(
            "[make_table_S3_eta_exclusions] WARNING: Could not auto-identify 'New Egyptian Kingdom' slug. "
            "Case A exclusion will exclude 0 rows. If you want an exact slug, hardcode it here.",
            file=sys.stderr,
        )

    exclusion_sets = [
        ("Baseline (all cases)", []),
        ("Exclude Case A (New Egyptian Kingdom)", caseA_found),
        ("Exclude Case B (Maximum η-ratio polity)", [max_eta_slug]),
        ("Exclude Case C (Near-threshold polity)", [near_eta_slug]),
        ("Exclude Cases A–C", sorted(set(caseA_found + [max_eta_slug, near_eta_slug]))),
    ]

    rows = []
    for label, exclude_slugs in exclusion_sets:
        sub = df if not exclude_slugs else df[~df[SLUG].isin(exclude_slugs)].copy()

        auc = cv_auc_5fold(sub)

        rows.append(
            {
                "Model Specification": label,
                "n": int(len(sub)),
                "θ*": round(float(eta_star), 2),
                "Percentile": f"{pct:.2f}%",
                "AUC": (round(float(auc), 2) if pd.notna(auc) else np.nan),
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    write_markdown_table(out)

    print("[make_table_S3_eta_exclusions] Wrote:")
    print(f"  {OUT_CSV}")
    print(f"  {OUT_MD}")

    # Helpful debug summary
    print("[make_table_S3_eta_exclusions] Identified slugs:")
    print(f"  Case A (New Egyptian Kingdom) slugs: {caseA_found if caseA_found else 'NONE'}")
    print(f"  Case B (max η) slug: {max_eta_slug}")
    print(f"  Case C (nearest θ*) slug: {near_eta_slug}")


if __name__ == "__main__":
    main()
