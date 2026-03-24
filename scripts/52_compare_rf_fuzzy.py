"""
Model Intercomparison & Divergence Analysis

GOAL: Quantifies the statistical and spatial discrepancy between the
Random Forest (Machine Learning) and the Fuzzy Inference System (Expert Knowledge).

RESPONSIBILITIES:
- Merges the two susceptibility surfaces on geographic coordinates to
  perform a pixel-by-pixel comparative audit.
- Calculates the Pearson Correlation Coefficient to measure the degree of
  alignment between data-driven and rule-based modeling paradigms.
- Visualizes probability distributions via scatter and histogram plots to
  detect systematic bias or sensitivity differences.
- Generates a 'Difference Map' (RF - Fuzzy) using a divergent colormap
  (coolwarm) to isolate geographic regions of high model uncertainty.
- Saves high-resolution (300 DPI) diagnostic figures for the final
  project synthesis and scientific validation.

INPUTS:
    - outputs/maps/nepal_landslide_probabilities_reduced.csv
    - outputs/maps/nepal_fuzzy_probabilities.csv
OUTPUTS:
    - outputs/figures/rf_vs_fuzzy_scatter.png
    - outputs/figures/rf_vs_fuzzy_histogram.png
    - outputs/figures/rf_minus_fuzzy_map.png
"""

from __future__ import annotations
from pathlib import Path

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RF_DEFAULT = r"outputs\maps\nepal_landslide_probabilities_reduced.csv"
FUZZY_DEFAULT = r"outputs\maps\nepal_fuzzy_probabilities.csv"
OUTDIR_DEFAULT = r"outputs\figures"


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--rf", default=RF_DEFAULT)
    parser.add_argument("--fuzzy", default=FUZZY_DEFAULT)
    parser.add_argument("--outdir", default=OUTDIR_DEFAULT)

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rf = pd.read_csv(args.rf)
    fuzzy = pd.read_csv(args.fuzzy)

    df = pd.merge(
        rf,
        fuzzy,
        on=["longitude", "latitude"],
        how="inner"
    )

    df = df.rename(
        columns={
            "landslide_probability": "rf_prob",
            "fuzzy_prob": "fuzzy_prob"
        }
    )

    print("Merged rows:", len(df))

    rf_scores = df["rf_prob"]
    fuzzy_scores = df["fuzzy_prob"]

    corr = np.corrcoef(rf_scores, fuzzy_scores)[0,1]

    print("Correlation RF vs Fuzzy:", round(corr,3))

    # Scatter comparison
    plt.figure(figsize=(6,6))

    plt.scatter(
        rf_scores,
        fuzzy_scores,
        alpha=0.7,
        s=25
    )

    plt.plot([0,1],[0,1],"k--")

    plt.xlabel("RF Probability")
    plt.ylabel("Fuzzy Probability")

    plt.title(f"RF vs Fuzzy Susceptibility\nCorrelation = {corr:.2f}")

    plt.xlim(0,1)
    plt.ylim(0,1)

    plt.tight_layout()

    scatter_path = outdir / "rf_vs_fuzzy_scatter.png"
    plt.savefig(scatter_path, dpi=300)

    plt.close()

    # Histogram comparison
    plt.figure(figsize=(7,5))

    plt.hist(
        rf_scores,
        bins=25,
        alpha=0.6,
        label="RF"
    )

    plt.hist(
        fuzzy_scores,
        bins=25,
        alpha=0.6,
        label="Fuzzy"
    )

    plt.xlabel("Susceptibility Score")
    plt.ylabel("Count")

    plt.title("Distribution of RF and Fuzzy Scores")

    plt.legend()

    plt.tight_layout()

    hist_path = outdir / "rf_vs_fuzzy_histogram.png"
    plt.savefig(hist_path, dpi=300)

    plt.close()

    # Difference map
    df["difference"] = df["rf_prob"] - df["fuzzy_prob"]

    plt.figure(figsize=(7,7))

    sc = plt.scatter(
        df["longitude"],
        df["latitude"],
        c=df["difference"],
        cmap="coolwarm",
        s=25,
        vmin=-1,
        vmax=1
    )

    plt.colorbar(sc,label="RF − Fuzzy")

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    plt.title("RF vs Fuzzy Difference Map")

    plt.tight_layout()

    diff_path = outdir / "rf_minus_fuzzy_map.png"
    plt.savefig(diff_path, dpi=300)

    plt.close()

    print("Saved:")
    print(scatter_path)
    print(hist_path)
    print(diff_path)


if __name__ == "__main__":
    main()