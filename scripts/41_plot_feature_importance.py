"""
Feature Importance Visualization: Scientific Evidence Ranking

GOAL: Produces publication-quality horizontal bar charts to rank the
environmental drivers of landslides based on their impact on model performance.

RESPONSIBILITIES:
- Loads the spatial permutation importance summary generated in Script 36.
- Maps technical database column names to human-readable 'Pretty Names'
  for professional reporting and clear communication.
- Visualizes the Mean Performance Drop (ROC-AUC or PR-AUC) with associated
  Standard Deviation error bars to indicate spatial stability.
- Provides a clear hierarchy of landslide triggers, allowing researchers
  to distinguish between primary drivers (e.g., Slope) and secondary
  modulators (e.g., Aspect).
- Saves a high-resolution (300 DPI) visualization optimized for inclusion
  in hazard assessment reports and scientific publications.

INPUT:  outputs/experiments/spatial_perm_importance_summary.csv
OUTPUT: outputs/figures/feature_importance_[metric].png
"""

from __future__ import annotations
from pathlib import Path

import argparse
import matplotlib.pyplot as plt
import pandas as pd


def pretty_name(name: str) -> str:
    mapping = {
        "elev_m": "Elevation",
        "log_precip_bio12": "Log Precipitation",
        "aspect_sin": "Aspect (sin)",
        "aspect_cos": "Aspect (cos)",
        "log_spi": "Log SPI",
        "twi": "TWI",
        "slope_deg": "Slope",
        "log_dist_river": "Log Dist. to River",
        "log_sca": "Log SCA",
    }
    return mapping.get(name, name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="outputs/experiments/spatial_perm_importance_summary.csv",
        help="Summary CSV from script 36",
    )
    parser.add_argument(
        "--outdir",
        default="outputs/figures",
    )
    parser.add_argument(
        "--metric",
        choices=["pr", "roc"],
        default="pr",
        help="Which importance drop to plot",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    if args.metric == "pr":
        mean_col = "pr_drop_mean"
        std_col = "pr_drop_std"
        xlab = "Mean PR-AUC Drop After Permutation"
        out_name = "feature_importance_pr_auc.png"
        title = "Spatial Permutation Importance (PR-AUC)"
    else:
        mean_col = "roc_drop_mean"
        std_col = "roc_drop_std"
        xlab = "Mean ROC-AUC Drop After Permutation"
        out_name = "feature_importance_roc_auc.png"
        title = "Spatial Permutation Importance (ROC-AUC)"

    df = df.sort_values(mean_col, ascending=True).copy()
    df["feature_pretty"] = df["feature"].apply(pretty_name)

    plt.figure(figsize=(8, 5))
    plt.barh(
        df["feature_pretty"],
        df[mean_col],
        xerr=df[std_col],
    )
    plt.xlabel(xlab)
    plt.ylabel("Feature")
    plt.title(title)
    plt.tight_layout()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    outpath = outdir / out_name
    plt.savefig(outpath, dpi=args.dpi)
    plt.close()

    print("Saved figure:", outpath)


if __name__ == "__main__":
    main()