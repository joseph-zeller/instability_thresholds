#!/usr/bin/env python3
"""
SI Table S2 — Core thresholds (η-ratio and SPC1)

Reads the canonical threshold summary produced by the modelling pipeline:
  results/thresholds/threshold_cv_summary_w100.csv

Writes locked SI outputs (never hand-edit):
  tables/S2_core_thresholds.csv
  tables/S2_core_thresholds.md

Design principles:
- Script defines truth.
- S2 is *only* core thresholds (no permutation tests, no robustness variants).
- Output schema is locked to SI: Dataset | n | θ* | Percentile | AUC
"""

from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd


# -----------------------------
# Repo paths (robust on Windows)
# -----------------------------

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]  # .../collapse_thresholds/src/tables -> repo root (.../collapse_thresholds)

IN_FILE = REPO_ROOT / "results" / "thresholds" / "threshold_cv_summary_w100.csv"

OUT_DIR = REPO_ROOT / "output" / "tables"
OUT_CSV = OUT_DIR / "S2_core_thresholds.csv"
OUT_MD = OUT_DIR / "S2_core_thresholds.md"


# -----------------------------
# Expected input schema
# -----------------------------
# We accept a few common column-name variants to reduce brittleness,
# but we *write* a single locked output schema.

REQUIRED_ANY_OF = {
    "dataset": ["dataset", "Dataset"],
    "n": ["n", "N"],
    "theta": ["theta_mean", "theta_star", "theta", "theta*"],
    "percentile": ["threshold_percentile", "percentile", "theta_percentile"],
    "auc": ["auc_mean", "auc", "AUC"],
}

# The two rows we must emit in S2 (locked)
# We’ll match these by looking for tokens in the dataset/predictor fields.
ROW_SPECS = [
    {
        "label": "Seshat (η-ratio)",
        "match_any": ["eta", "η", "eta_ratio", "eta-ratio"],
    },
    {
        "label": "SPC1",
        "match_any": ["spc1", "SPC1"],
    },
]


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower()
        if key in cols_lower:
            return cols_lower[key]
    return None


def _require_cols(df: pd.DataFrame) -> dict[str, str]:
    mapping: dict[str, str] = {}
    missing = []
    for logical_name, candidates in REQUIRED_ANY_OF.items():
        col = _find_col(df, candidates)
        if col is None:
            missing.append((logical_name, candidates))
        else:
            mapping[logical_name] = col
    if missing:
        msg = ["Missing required columns in threshold summary CSV:"]
        for logical, cands in missing:
            msg.append(f"  - need one of {cands} (for '{logical}')")
        raise ValueError("\n".join(msg))
    return mapping


def _select_row(df: pd.DataFrame, label: str, match_any: list[str]) -> pd.DataFrame:
    # We try to match against 'dataset' and also 'predictor' if present.
    haystacks = []
    if "dataset" in df.columns:
        haystacks.append(df["dataset"].astype(str))
    if "predictor" in df.columns:
        haystacks.append(df["predictor"].astype(str))

    if not haystacks:
        # Fall back to searching all columns as string
        haystacks = [df.astype(str).agg(" ".join, axis=1)]

    mask = False
    for h in haystacks:
        h_low = h.str.lower()
        m = False
        for token in match_any:
            m = m | h_low.str.contains(str(token).lower(), na=False)
        mask = mask | m

    out = df[mask].copy()
    if out.empty:
        raise ValueError(
            f"Could not find a row for '{label}'. "
            f"Tried matching tokens {match_any} against dataset/predictor."
        )

    # If multiple matches, prefer rows that look like the core w100 summary
    # by preferring dataset strings containing 'w100' if available.
    if len(out) > 1 and "dataset" in out.columns:
        w100 = out["dataset"].astype(str).str.lower().str.contains("w100", na=False)
        if w100.any():
            out = out[w100].copy()

    # If still multiple, keep first but warn loudly.
    if len(out) > 1:
        print(
            f"[make_table_S2_core_thresholds] WARNING: multiple rows matched '{label}'. "
            f"Keeping the first {out.index[0]}. Consider making dataset keys unique.",
            file=sys.stderr,
        )
        out = out.iloc[[0]].copy()

    return out


def _format_percent(p: float) -> str:
    # Stored as numeric (e.g., 95.238...) → "95.2%"
    return f"{p:.1f}%"


def main() -> None:
    if not IN_FILE.exists():
        raise FileNotFoundError(
            f"Input not found:\n  {IN_FILE}\n\n"
            "This script expects the thresholds pipeline to have produced:\n"
            "  results/thresholds/threshold_cv_summary_w100.csv"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(IN_FILE)
    colmap = _require_cols(df_raw)

    # Standardise logical columns
    df = df_raw.rename(
        columns={
            colmap["dataset"]: "dataset",
            colmap["n"]: "n",
            colmap["theta"]: "theta",
            colmap["percentile"]: "percentile",
            colmap["auc"]: "auc",
        }
    )

    # Build the two locked rows
    rows = []
    for spec in ROW_SPECS:
        r = _select_row(df, spec["label"], spec["match_any"]).iloc[0]

        # Defensive parsing
        n = int(r["n"])
        theta = float(r["theta"])
        perc = float(r["percentile"])
        auc = float(r["auc"])

        rows.append(
            {
                "Dataset": spec["label"],
                "n": n,
                "θ*": round(theta, 2),
                "Percentile": _format_percent(perc),
                "AUC": round(auc, 2),
            }
        )

    out = pd.DataFrame(rows)

    # Hard lock: exactly two rows, in expected order
    expected = [s["label"] for s in ROW_SPECS]
    if out["Dataset"].tolist() != expected:
        raise ValueError(
            f"S2 ordering drift. Expected {expected}, got {out['Dataset'].tolist()}."
        )

    # Write CSV
    out.to_csv(OUT_CSV, index=False)

    # Write Markdown (no external deps)
    md_lines = []
    md_lines.append("| Dataset | n | θ* | Percentile | AUC |")
    md_lines.append("|---|---:|---:|---:|---:|")
    for _, r in out.iterrows():
        md_lines.append(
            f"| {r['Dataset']} | {r['n']} | {r['θ*']:.2f} | {r['Percentile']} | {r['AUC']:.2f} |"
        )
    OUT_MD.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print("[make_table_S2_core_thresholds] Input:")
    print(f"  {IN_FILE}")
    print("[make_table_S2_core_thresholds] Wrote:")
    print(f"  {OUT_CSV}")
    print(f"  {OUT_MD}")


if __name__ == "__main__":
    main()
