#!/usr/bin/env python3
"""
make_table_S1_pca_complexity.py

Table S1: PCA of informational/institutional complexity indicators.

- Loads an input CSV containing five informational/institutional indicators
- Z-standardises the indicators
- Runs PCA
- Outputs:
  * output/tables/table_S1_pca_complexity.csv   (PC1/PC2 loadings)
  * output/tables/table_S1_pca_complexity.md    (fixed-width SI table)
  * output/tables/table_S1_pca_variance.csv     (PC1/PC2 variance explained)
  * output/tables/table_S1_pca_diagnostics.txt  (repro diagnostics)

Usage:
  python src/tables/make_table_S1_pca_complexity.py
  python src/tables/make_table_S1_pca_complexity.py --input results/pca/pca_complexity_inputs.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# -----------------------------
# Repo paths
# -----------------------------

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
OUT_DIR = REPO_ROOT / "output" / "tables"


# -----------------------------
# Column mapping
# -----------------------------

# Each concept maps to a list of acceptable column names (case-insensitive match).
COL_ALIASES: dict[str, list[str]] = {
    "Hierarchy": [
        "Hierarchy", "hierarchy", "Hier", "HIER", "hier", "soc_hierarchy",
    ],
    "Governance": [
        "Governance", "governance", "Gov", "GOV", "gov", "soc_governance",
    ],
    "Infrastructure": [
        "Infrastructure", "infrastructure", "Infra", "INFRA", "infra", "soc_infrastructure",
    ],
    "Information systems": [
        "Information systems", "Information_systems", "information systems",
        "Info", "INFO", "info", "information", "info_systems", "soc_information",
    ],
    "Monetisation": [
        "Monetisation", "Monetization", "monetisation", "monetization",
        "Money", "MONEY", "money", "soc_monetisation", "soc_monetization",
    ],
}

ORDERED_LABELS = ["Hierarchy", "Governance", "Infrastructure", "Information systems", "Monetisation"]


def _normalise_col(s: str) -> str:
    return "".join(ch.lower() for ch in s.strip())


def resolve_columns(df: pd.DataFrame) -> dict[str, str]:
    """
    Resolve the required PCA columns in df.

    Returns mapping: pretty_label -> actual_column_name
    Raises KeyError if any concept cannot be resolved.
    """
    norm_to_actual = {_normalise_col(c): c for c in df.columns}

    resolved: dict[str, str] = {}
    for pretty in ORDERED_LABELS:
        found = None
        for alias in COL_ALIASES[pretty]:
            key = _normalise_col(alias)
            if key in norm_to_actual:
                found = norm_to_actual[key]
                break
        if found is None:
            raise KeyError(
                f"Could not resolve required PCA column for '{pretty}'.\n"
                f"Tried aliases: {COL_ALIASES[pretty]}\n"
                f"Available columns: {list(df.columns)}"
            )
        resolved[pretty] = found

    return resolved


def find_input_file(explicit: str | None = None) -> Path:
    """
    Locate the PCA input CSV.

    Preference order:
      1) --input (if provided)
      2) results/pca/pca_complexity_inputs.csv
      3) results/pca/*complexity*.csv
      4) data/final/*equinox*.csv
      5) data/final/*seshat*.csv
    """
    if explicit:
        p = (REPO_ROOT / explicit).resolve() if not Path(explicit).is_absolute() else Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"--input provided but file not found: {p}")
        return p

    tried: list[Path] = []

    candidates = [
        REPO_ROOT / "results" / "pca" / "pca_complexity_inputs.csv",
    ]
    for c in candidates:
        tried.append(c)
        if c.exists():
            return c

    # broader fallbacks
    patterns = [
        REPO_ROOT / "results" / "pca",
        REPO_ROOT / "data" / "final",
    ]

    for base in patterns:
        if not base.exists():
            continue
        for pat in ["*complexity*.csv", "*equinox*.csv", "*seshat*.csv"]:
            for c in sorted(base.glob(pat)):
                tried.append(c)
                if c.exists():
                    return c

    msg = "Could not find a suitable PCA input CSV. Tried (first 20 shown):\n"
    msg += "\n".join(str(p) for p in tried[:20])
    msg += "\n\nFix: run with --input <path-to-your-pca-input.csv>"
    raise FileNotFoundError(msg)


def write_si_fixedwidth_table(table: pd.DataFrame, var_pc1: float, var_pc2: float, out_md: Path) -> None:
    """
    Writes a fixed-width (Consolas-friendly) table block to .md.
    """
    lines = []
    lines.append("Table S1. Principal component analysis of informational complexity indicators")
    lines.append("-------------------------------------------------------------------------------")
    lines.append("Indicator               |   PC1 loading   |   PC2 loading")
    lines.append("-------------------------------------------------------------------------------")

    for idx, row in table.iterrows():
        lines.append(f"{idx:<23} | {row['PC1']:>13.3f}   | {row['PC2']:>10.3f}")

    lines.append("-------------------------------------------------------------------------------")
    lines.append(f"Variance explained (%)  | {var_pc1:>13.2f}   | {var_pc2:>10.2f}")
    lines.append("-------------------------------------------------------------------------------")
    lines.append(f"Cumulative (%)          | {var_pc1 + var_pc2:>15.2f}")
    lines.append("-------------------------------------------------------------------------------")
    lines.append("Notes:")
    lines.append("– All indicators were z-standardised prior to PCA.")
    lines.append("– PCA was computed on pooled polity-window observations.")
    lines.append("– PC1 represents accumulated institutional and coordination complexity.")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None, help="Path to PCA input CSV (relative to repo root or absolute)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    in_path = find_input_file(args.input)
    df = pd.read_csv(in_path)

    colmap = resolve_columns(df)
    X = df[list(colmap.values())].dropna()
    n_rows = int(X.shape[0])

    if n_rows < 5:
        raise ValueError(
            f"Not enough complete rows for PCA after dropna(): n={n_rows}. "
            f"Check missingness in {in_path}."
        )

    # Standardise
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X.to_numpy(dtype=float))

    # PCA fit (full, but we only report PC1/PC2)
    pca = PCA(n_components=len(colmap))
    pca.fit(Xz)

    # Loadings: features x components
    loadings = pd.DataFrame(
        pca.components_.T,
        index=list(colmap.keys()),  # pretty labels
        columns=[f"PC{i+1}" for i in range(len(colmap))],
    )

    # Orient PC1 to be positive on average (sign is arbitrary)
    if float(loadings["PC1"].mean()) < 0:
        loadings *= -1

    table_s1 = loadings[["PC1", "PC2"]].copy().round(3)
    table_s1.index.name = "Indicator"

    # Variance (PC1/PC2)
    var_pct = 100.0 * pca.explained_variance_ratio_
    var_pc1 = float(var_pct[0])
    var_pc2 = float(var_pct[1])

    var_out = pd.DataFrame(
        {
            "PC": ["PC1", "PC2"],
            "Variance_explained": [float(pca.explained_variance_ratio_[0]), float(pca.explained_variance_ratio_[1])],
            "Variance_percent": [var_pc1, var_pc2],
            "Cumulative_percent": [var_pc1, var_pc1 + var_pc2],
        }
    )

    # Outputs
    out_csv = OUT_DIR / "table_S1_pca_complexity.csv"
    out_md = OUT_DIR / "table_S1_pca_complexity.md"
    out_var = OUT_DIR / "table_S1_pca_variance.csv"
    out_diag = OUT_DIR / "table_S1_pca_diagnostics.txt"

    table_s1.to_csv(out_csv, index=True)
    var_out.to_csv(out_var, index=False)

    write_si_fixedwidth_table(table_s1, var_pc1, var_pc2, out_md)

    diag_lines = []
    diag_lines.append("Table S1 PCA diagnostics")
    diag_lines.append("========================")
    diag_lines.append(f"Input file: {in_path.as_posix()}")
    diag_lines.append(f"Rows used (complete cases): {n_rows}")
    diag_lines.append("Resolved columns (pretty -> actual):")
    for k, v in colmap.items():
        diag_lines.append(f"  {k}: {v}")
    diag_lines.append("")
    diag_lines.append("Variance explained (percent):")
    diag_lines.append(var_out.to_string(index=False))
    diag_lines.append("")
    out_diag.write_text("\n".join(diag_lines) + "\n", encoding="utf-8")

    print("[make_table_S1_pca_complexity] Input:")
    print(f"  {in_path}")
    print("[make_table_S1_pca_complexity] Wrote:")
    print(f"  {out_csv}")
    print(f"  {out_md}")
    print(f"  {out_var}")
    print(f"  {out_diag}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n❌ Script failed:\n")
        print(e)
        sys.exit(1)
