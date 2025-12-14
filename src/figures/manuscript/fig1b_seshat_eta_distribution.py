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


def find_repo_root(start: Path) -> Path:
    for p in [start] + list(start.parents):
        if (p / "README.md").exists() and (p / "src").exists():
            return p
    return start.parents[2]


REPO = find_repo_root(Path(__file__).resolve())
DATA = REPO / "data" / "final" / "seshat_EI_collapse_panel_w100.csv"
OUTDIR = REPO / "figures" / "manuscript"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUTDIR / "fig1b_seshat_eta_distribution.png"
OUT_PDF = OUTDIR / "fig1b_seshat_eta_distribution.pdf"

# Canonical reference
ETA_XSTAR = 2.116612  # manuscript canonical x*
BAND_LO, BAND_HI = 0.95, 0.96


def main() -> None:
    df = pd.read_csv(DATA)

    # columns
    if "eta_ratio" not in df.columns:
        raise ValueError("eta_ratio column not found in seshat_EI_collapse_panel_w100.csv")
    ycol = "collapse_next_100y"
    if ycol not in df.columns:
        raise ValueError(f"{ycol} not found in Seshat panel.")

    x = df["eta_ratio"].astype(float)
    y = df[ycol].astype(int)

    m = x.notna() & y.notna()
    x = x[m].to_numpy()
    y = y[m].to_numpy()

    q_lo = float(np.quantile(x, BAND_LO))
    q_hi = float(np.quantile(x, BAND_HI))

    pct_xstar = 100.0 * float(np.mean(x <= ETA_XSTAR))

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=300)

    ax.hist(x, bins=10, density=True, alpha=0.25, label="Histogram")

    if _HAVE_SCIPY and len(np.unique(x)) > 3:
        kde = gaussian_kde(x)
        xs = np.linspace(x.min(), x.max(), 400)
        ax.plot(xs, kde(xs), linewidth=2.0, label="KDE")

    ax.axvspan(q_lo, q_hi, alpha=0.18, label="95–96th percentile band")
    ax.axvline(ETA_XSTAR, linestyle="--", linewidth=2.0,
               label=f"x* ≈ {ETA_XSTAR:.2f} ({pct_xstar:.1f}th pct.)")

    y_min, y_max = ax.get_ylim()
    base_nc = y_min + 0.02 * (y_max - y_min)
    base_c  = y_min + 0.05 * (y_max - y_min)

    ax.plot(x[y == 0], np.full(np.sum(y == 0), base_nc), "|", markersize=10,
            label=f"No collapse within 100y (n={int(np.sum(y==0))})")
    ax.plot(x[y == 1], np.full(np.sum(y == 1), base_c), "|", markersize=10,
            label=f"Collapse within 100y (n={int(np.sum(y==1))})")

    ax.set_title("Distribution of ηᵢ/ηₑ prior to collapse (Seshat, 100-year window)")
    ax.set_xlabel("ηᵢ/ηₑ overload ratio")
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
