#!/usr/bin/env python3
"""
Figure 5 (Manuscript). Conceptual model of energetic–informational overload.

Schematic illustration only (no empirical data).
Designed to be compact and page-layout friendly.
Outputs:
  figures/manuscript/fig5_conceptual_overload_mechanism.(png|pdf)
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# Repo / paths
# ----------------------------

def find_repo_root(start: Path) -> Path:
    """
    Find repository root by walking upward until we see a 'src' dir
    (and ideally a README.md). Falls back gracefully.
    """
    for p in [start] + list(start.parents):
        if (p / "src").exists():
            return p
    return start.parents[2]


REPO = find_repo_root(Path(__file__).resolve())
OUTDIR = REPO / "figures" / "manuscript"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUTDIR / "fig5_conceptual_overload_mechanism.png"
OUT_PDF = OUTDIR / "fig5_conceptual_overload_mechanism.pdf"


# ----------------------------
# Conceptual trajectories
# ----------------------------

def conceptual_series() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Returns x, eta_E, eta_I, and the start of the 'instability regime' in x.
    All values are normalised and illustrative.
    """
    x = np.linspace(0, 10, 400)

    # Energetic renewal capacity (slow, near-linear growth)
    eta_E = 1.00 + 0.06 * x

    # Informational/coordination load (accelerating)
    eta_I = 0.55 + 0.05 * x + 0.028 * x**2

    # Conceptual onset of upper-tail instability
    instability_x = 6.2

    return x, eta_E, eta_I, instability_x


# ----------------------------
# Plot
# ----------------------------

def main() -> None:
    x, eta_E, eta_I, instability_x = conceptual_series()

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # Compact size to avoid Word pagination issues
    fig, ax = plt.subplots(figsize=(5.4, 2.7), dpi=300)

    ax.plot(x, eta_E, lw=2.0, label="Energetic renewal capacity (ηₑ)")
    ax.plot(x, eta_I, lw=2.0, label="Informational / coordination load (ηᵢ)")

    # Shade upper-tail regime (conceptual)
    ax.axvspan(instability_x, x.max(), alpha=0.18, label="Upper-tail instability regime")

    # Vertical marker for regime boundary
    ax.axvline(instability_x, linestyle=":", lw=1.2)

    # Annotation (kept small & inside plot area)
    y_at = float(np.interp(instability_x, x, eta_I))
    ax.annotate(
        "Heightened\nvulnerability",
        xy=(instability_x + 0.15, y_at),
        xytext=(7.35, y_at - 0.35),
        arrowprops=dict(arrowstyle="->", lw=0.8),
        ha="left",
        va="center",
    )

    ax.set_xlabel("System development / time")
    ax.set_ylabel("Relative scale (normalised)")

    # Light grid for readability (subtle)
    ax.yaxis.grid(True, linestyle="--", alpha=0.25)

    ax.legend(frameon=False, loc="upper left")

    plt.tight_layout()
    fig.savefig(OUT_PNG, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)

    print("[OK] Saved:")
    print(" ", OUT_PNG)
    print(" ", OUT_PDF)


if __name__ == "__main__":
    main()
