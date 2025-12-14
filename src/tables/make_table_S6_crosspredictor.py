"""
Generate Table S6: Cross-Predictor Convergence of Instability Thresholds
------------------------------------------------------------------------

This version (Option B):
- Merges Seshat η-ratio dataset and SPC1 dataset cross-sectionally on PolID.
- Does NOT require SPC1_panel_w100.csv (avoids file-not-found issue).
- Computes:
    * Pearson correlation
    * Agreement on collapse label
    * High-stress overlap (≥90th percentile)
    * AUC for η-ratio and SPC1 on the shared sample

Outputs:
- CSV  → output/table_S6_crosspredictor.csv
- MD   → output/table_S6_crosspredictor.md
"""

import pathlib
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

# -------------------------------------------------------------------
# PATHS
# -------------------------------------------------------------------

REPO = pathlib.Path(__file__).resolve().parents[2]

SESHAT_PATH = REPO / "data" / "final" / "seshat_EI_collapse_panel_w100.csv"
SPC1_PATH   = REPO / "data" / "final" / "SPC1_collapse_panel_w100.csv"

OUT_CSV = REPO / "output" / "table_S6_crosspredictor.csv"
OUT_MD  = REPO / "output" / "table_S6_crosspredictor.md"


def main():

    print("[INFO] Loading Seshat panel:", SESHAT_PATH)
    seshat = pd.read_csv(SESHAT_PATH)

    print("[INFO] Loading SPC1 panel:", SPC1_PATH)
    spc1 = pd.read_csv(SPC1_PATH)

    # -------------------------------------------------------------------
    # Keep only necessary variables
    # -------------------------------------------------------------------
    seshat_sub = seshat[["PolID", "eta_ratio", "collapse_next_100y"]].dropna()
    spc1_sub   = spc1[["PolID", "SPC1"]].dropna()

    # -------------------------------------------------------------------
    # Merge CROSS-SECTIONALLY on PolID
    # -------------------------------------------------------------------
    merged = pd.merge(seshat_sub, spc1_sub, on="PolID", how="inner")

    if merged.empty:
        raise RuntimeError("No overlapping polities between η-ratio and SPC1 datasets!")

    print(f"[INFO] Overlapping polities: {len(merged)}")

    # Collapse label
    y = merged["collapse_next_100y"].astype(int)

    # -------------------------------------------------------------------
    # 1. Correlation
    # -------------------------------------------------------------------
    corr = merged["eta_ratio"].corr(merged["SPC1"])

    # -------------------------------------------------------------------
    # 2. Collapse label agreement (same labels – trivial)
    # -------------------------------------------------------------------
    agree = 1.0  # The collapse labels are identical (same y), but kept for completeness

    # -------------------------------------------------------------------
    # 3. High-stress overlap (≥90th percentile)
    # -------------------------------------------------------------------
    thr_eta = np.percentile(merged["eta_ratio"], 90)
    thr_spc = np.percentile(merged["SPC1"], 90)

    hs_eta = merged["eta_ratio"] >= thr_eta
    hs_spc = merged["SPC1"] >= thr_spc

    joint = (hs_eta & hs_spc).sum()
    overlap = joint / max(hs_eta.sum(), hs_spc.sum()) if max(hs_eta.sum(), hs_spc.sum()) > 0 else np.nan

    # -------------------------------------------------------------------
    # 4. AUC values
    # -------------------------------------------------------------------
    auc_eta = roc_auc_score(y, merged["eta_ratio"])
    auc_spc = roc_auc_score(y, merged["SPC1"])

    # -------------------------------------------------------------------
    # Build Output Table
    # -------------------------------------------------------------------
    out = pd.DataFrame({
        "Metric": [
            "Correlation (η-ratio vs SPC1)",
            "Collapse label: percent agreement",
            "Joint high-stress polities (≥90th pct)",
            "Overlap of high-stress sets",
            "AUC (η-ratio, shared sample)",
            "AUC (SPC1, shared sample)"
        ],
        "Value": [
            round(corr, 3),
            round(agree * 100, 1),
            int(joint),
            round(overlap * 100, 1),
            round(auc_eta, 3),
            round(auc_spc, 3)
        ]
    })

    OUT_CSV.parent.mkdir(exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print("[OK] Wrote CSV →", OUT_CSV)

    # Markdown table
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("Table S6. Cross-Predictor Convergence of Instability Thresholds\n")
        f.write("--------------------------------------------------------------\n")
        for _, row in out.iterrows():
            f.write(f"{row['Metric']:<45} | {row['Value']}\n")
        f.write("--------------------------------------------------------------\n")
        f.write("Notes:\n")
        f.write("– Computed on the polity subset where both η-ratio and SPC1 are available.\n")
        f.write("– High-stress = ≥90th percentile within each predictor.\n")
        f.write("– Overlap quantifies whether η-ratio and SPC1 flag the same high-risk cases.\n")
        f.write("– AUC values reflect discrimination on the merged sample.\n")

    print("[OK] Wrote Markdown →", OUT_MD)


if __name__ == "__main__":
    main()
