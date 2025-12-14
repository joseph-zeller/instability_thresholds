#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.stats import gaussian_kde
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


# ----------------------------
# Repo / paths
# ----------------------------

def find_repo_root(start: Path) -> Path:
    for p in [start] + list(start.parents):
        if (p / "README.md").exists() and (p / "src").exists():
            return p
    return start.parents[2]  # fallback (works if file is src/figures/*)


REPO = find_repo_root(Path(__file__).resolve())
DATA = REPO / "data" / "final" / "SPC1_collapse_panel_w100.csv"
OUTDIR = REPO / "figures" / "manuscript"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUTDIR / "fig1a_spc1_distribution.png"
OUT_PDF = OUTDIR / "fig1a_spc1_distribution.pdf"

# Canonical reference (used only for annotation)
SPC1_XSTAR = 5.03  # manuscript canonical
BAND_LO, BAND_HI = 0.95, 0.96


def main() -> None:
    df = pd.read_csv(DATA)

    # column checks
    if "SPC1" not in df.columns:
        raise ValueError("SPC1 column not found in SPC1_collapse_panel_w100.csv")
    ycol = "collapse_next_100y"
    if ycol not in df.columns:
        raise ValueError(f"{ycol} not found in SPC1 panel.")

    x = df["SPC1"].astype(float)
    y = df[ycol].astype(int)

    # drop NA
    m = x.notna() & y.notna()
    x = x[m].to_numpy()
    y = y[m].to_numpy()

    # percentiles (for shading band)
    q_lo = float(np.quantile(x, BAND_LO))
    q_hi = float(np.quantile(x, BAND_HI))

    # percentile position of canonical x*
    pct_xstar = 100.0 * float(np.mean(x <= SPC1_XSTAR))

    # ----------------------------
    # Plot
    # ----------------------------
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=300)

    # histogram (density)
    ax.hist(x, bins=12, density=True, alpha=0.25, label="Histogram")

    # KDE (optional)
    if _HAVE_SCIPY and len(np.unique(x)) > 3:
        kde = gaussian_kde(x)
        xs = np.linspace(x.min(), x.max(), 400)
        ax.plot(xs, kde(xs), linewidth=2.0, label="KDE")

    # 95–96 band
    ax.axvspan(q_lo, q_hi, alpha=0.18, label="95–96th percentile band")

    # x* line
    ax.axvline(SPC1_XSTAR, linestyle="--", linewidth=2.0,
               label=f"x* ≈ {SPC1_XSTAR:.2f} ({pct_xstar:.1f}th pct.)")

    # Rug ticks: black = no collapse, red = collapse
    y_min, y_max = ax.get_ylim()
    base_nc = y_min + 0.02 * (y_max - y_min)
    base_c  = y_min + 0.05 * (y_max - y_min)

    ax.plot(x[y == 0], np.full(np.sum(y == 0), base_nc), "|", markersize=10,
            label=f"No collapse within 100y (n={int(np.sum(y==0))})")
    ax.plot(x[y == 1], np.full(np.sum(y == 1), base_c), "|", markersize=10,
            label=f"Collapse within 100y (n={int(np.sum(y==1))})")

    ax.set_title("Distribution of SPC1 overload index prior to collapse\n(100-year fixed-window panel)")
    ax.set_xlabel("Informational overload index (SPC1)")
    ax.set_ylabel("Density")
    ax.legend(loc="upper left", frameon=False, fontsize=9)

    plt.tight_layout()
    fig.savefig(OUT_PNG, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)

    print("[OK] Saved:", OUT_PNG)
    print("[OK] Saved:", OUT_PDF)


if __name__ == "__main__":
    main()
