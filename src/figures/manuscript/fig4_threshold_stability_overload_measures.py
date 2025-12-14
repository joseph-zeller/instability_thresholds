#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def find_repo_root(start: Path) -> Path:
    for p in [start] + list(start.parents):
        if (p / "README.md").exists() and (p / "src").exists():
            return p
    return start.parents[2]


REPO = find_repo_root(Path(__file__).resolve())
DATA_SESHAT = REPO / "data" / "final" / "seshat_EI_collapse_panel_w100.csv"
DATA_SPC1   = REPO / "data" / "final" / "SPC1_collapse_panel_w100.csv"

OUTDIR = REPO / "figures" / "manuscript"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUTDIR / "fig4_threshold_stability_overload_measures.png"
OUT_PDF = OUTDIR / "fig4_threshold_stability_overload_measures.pdf"

ETA_XSTAR = 2.116612
SPC1_XSTAR = 5.03

BAND_LO, BAND_HI = 90, 97


def main() -> None:
    df_s = pd.read_csv(DATA_SESHAT)
    df_p = pd.read_csv(DATA_SPC1)

    xs = df_s["eta_ratio"].astype(float)
    xs = xs[xs.notna()].to_numpy()

    xp = df_p["SPC1"].astype(float)
    xp = xp[xp.notna()].to_numpy()

    eta_pct = 100.0 * float(np.mean(xs <= ETA_XSTAR))
    spc1_pct = 100.0 * float(np.mean(xp <= SPC1_XSTAR))

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=(8.2, 2.4), dpi=300)

    # Band
    ax.axvspan(BAND_LO, BAND_HI, color="grey", alpha=0.18, label="Universal instability band (90–97th percentile)")

    # Points (y positions just to separate labels)
    ax.scatter([eta_pct], [0.35], s=60)
    ax.scatter([spc1_pct], [0.65], s=60, marker="D")

    ax.text(eta_pct + 0.2, 0.35, f"{eta_pct:.1f}th pct.", va="center", fontsize=9)
    ax.text(spc1_pct + 0.2, 0.65, f"{spc1_pct:.1f}th pct.", va="center", fontsize=9)

    ax.set_xlim(88, 100)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.35, 0.65])
    ax.set_yticklabels(["Seshat ηᵢ/ηₑ", "SPC1 overload index"])

    ax.set_xlabel("Percentile of overload metric at collapse threshold")
    ax.set_title("Stability of collapse thresholds across overload measures")
    ax.legend(loc="lower center", frameon=False, fontsize=9)

    plt.tight_layout()
    fig.savefig(OUT_PNG, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)

    print("[OK] Saved:", OUT_PNG)
    print("[OK] Saved:", OUT_PDF)


if __name__ == "__main__":
    main()
