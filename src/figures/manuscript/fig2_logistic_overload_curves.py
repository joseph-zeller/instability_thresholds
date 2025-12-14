#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


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

OUT_PNG = OUTDIR / "fig2_logistic_overload_curves.png"
OUT_PDF = OUTDIR / "fig2_logistic_overload_curves.pdf"

ETA_XSTAR = 2.116612
SPC1_XSTAR = 5.03


def fit_curve(x: np.ndarray, y: np.ndarray):
    model = LogisticRegression(solver="lbfgs")
    model.fit(x.reshape(-1, 1), y)
    return model


def main() -> None:
    df_s = pd.read_csv(DATA_SESHAT)
    df_p = pd.read_csv(DATA_SPC1)

    # Seshat
    xs = df_s["eta_ratio"].astype(float)
    ys = df_s["collapse_next_100y"].astype(int)
    ms = xs.notna() & ys.notna()
    xs = xs[ms].to_numpy()
    ys = ys[ms].to_numpy()

    # SPC1
    xp = df_p["SPC1"].astype(float)
    yp = df_p["collapse_next_100y"].astype(int)
    mp = xp.notna() & yp.notna()
    xp = xp[mp].to_numpy()
    yp = yp[mp].to_numpy()

    m_s = fit_curve(xs, ys)
    m_p = fit_curve(xp, yp)

    # evaluation grid
    grid_s = np.linspace(xs.min(), xs.max(), 400)
    grid_p = np.linspace(xp.min(), xp.max(), 400)

    ps = m_s.predict_proba(grid_s.reshape(-1, 1))[:, 1]
    pp = m_p.predict_proba(grid_p.reshape(-1, 1))[:, 1]

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6), dpi=300, sharey=True)

    # Panel A: Seshat
    ax = axes[0]
    ax.scatter(xs, ys, s=12, alpha=0.35)
    ax.plot(grid_s, ps, linewidth=2.2)
    ax.axvline(ETA_XSTAR, linestyle="--", linewidth=2.0)
    ax.set_title("A  Seshat overload vs collapse")
    ax.set_xlabel("ηᵢ/ηₑ overload ratio")
    ax.set_ylabel("Probability of collapse")
    ax.set_ylim(-0.05, 1.05)
    ax.text(ETA_XSTAR, 0.92, f"x = {ETA_XSTAR:.2f}", ha="center", va="top", fontsize=9)

    # Panel B: SPC1
    ax = axes[1]
    ax.scatter(xp, yp, s=12, alpha=0.35)
    ax.plot(grid_p, pp, linewidth=2.2)
    ax.axvline(SPC1_XSTAR, linestyle="--", linewidth=2.0)
    ax.set_title("B  SPC1 overload vs collapse")
    ax.set_xlabel("SPC1 overload index")
    ax.set_ylim(-0.05, 1.05)
    ax.text(SPC1_XSTAR, 0.92, f"x = {SPC1_XSTAR:.2f}", ha="center", va="top", fontsize=9)

    plt.tight_layout()
    fig.savefig(OUT_PNG, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)

    print("[OK] Saved:", OUT_PNG)
    print("[OK] Saved:", OUT_PDF)


if __name__ == "__main__":
    main()
