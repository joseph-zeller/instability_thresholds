#!/usr/bin/env python3
"""
Figure S5 – PCA Biplot of Informational Complexity Variables

Replicates the original PCA biplot style:
  - Orange "x" polity points
  - Light grey grid
  - Black loading arrows with orange heads
  - Title: "Figure S5. PCA Biplot of Informational Complexity Variables"

Inputs
------
data/final/seshat_EI_collapse_panel_w100.csv
    Columns used: Gov, Hier, Infra, Info, Money

Output
------
figures/Figure_S5_PCA_biplot.pdf
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)
DATA_FILE = os.path.join(BASE_DIR, "data", "final",
                         "seshat_EI_collapse_panel_w100.csv")
OUTFILE = os.path.join(BASE_DIR, "figures",
                       "Figure_S5_PCA_biplot.pdf")

INFO_VARS = ["Gov", "Hier", "Infra", "Info", "Money"]


# ---------------------------------------------------------------------
# Data + PCA
# ---------------------------------------------------------------------

def load_data():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Data file not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)

    missing = [v for v in INFO_VARS if v not in df.columns]
    if missing:
        raise ValueError(f"Missing required variables in dataset: {missing}")

    return df.dropna(subset=INFO_VARS)


def run_pca(df):
    """Standardised PCA (correlation-matrix style)."""
    X = df[INFO_VARS].values
    X_std = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2)
    scores = pca.fit_transform(X_std)
    loadings = pca.components_.T       # (variables × PCs)
    explained = pca.explained_variance_ratio_ * 100.0
    return scores, loadings, explained


# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------

def plot_pca(scores, loadings, explained):
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 10

    fig, ax = plt.subplots(figsize=(8, 5))

    # Orange "x" points for polity-windows
    ax.scatter(scores[:, 0],
               scores[:, 1],
               marker="x",
               s=35,
               color="#f4a300",
               alpha=0.8)

    # Loading arrows (black lines, orange heads)
    arrow_scale = 2.5
    for i, var in enumerate(INFO_VARS):
        x_end = loadings[i, 0] * arrow_scale
        y_end = loadings[i, 1] * arrow_scale

        ax.arrow(0, 0,
                 x_end, y_end,
                 linewidth=1.3,
                 color="black",
                 head_width=0.06,
                 head_length=0.10,
                 length_includes_head=True)

        ax.text(x_end * 1.05,
                y_end * 1.05,
                var,
                fontsize=10,
                va="center")

    # Labels + title
    ax.set_xlabel(f"PC1 ({explained[0]:.1f}% var. explained)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1f}% var. explained)")
    ax.set_title(
        "Figure S5. PCA Biplot of Informational Complexity Variables",
        fontweight="bold",
        pad=10,
    )

    # Light grey grid to match the original look
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.4)

    fig.tight_layout()
    plt.savefig(OUTFILE, dpi=300, bbox_inches="tight")
    print(f"[SAVE] Figure S5 written to: {OUTFILE}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    print("[LOAD] Reading:", DATA_FILE)
    df = load_data()

    print("[PCA] Running PCA...")
    scores, loadings, explained = run_pca(df)

    print("[PLOT] Generating PCA biplot...")
    plot_pca(scores, loadings, explained)


if __name__ == "__main__":
    main()
