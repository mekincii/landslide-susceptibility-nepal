"""
Model Calibration Audit: Probabilistic Reliability Visualization

GOAL: Generates a Reliability Diagram to assess the correspondence between
predicted landslide probabilities and actual observed frequencies.

RESPONSIBILITIES:
- Visualizes the 'Calibration Curve' against the ideal 45-degree diagonal
  to identify systematic overconfidence or underconfidence in risk scores.
- Aggregates probability bins across all spatial folds to provide a robust,
  national-scale assessment of model trustworthiness.
- Integrates the quantitative Brier Score and Expected Calibration Error (ECE)
  from experimental logs into the final report.
- Provides critical evidence for risk communicators, ensuring that
  susceptibility percentages (0-100%) are statistically grounded in reality.
- Saves a publication-quality diagnostic plot (300 DPI) for the final
  technical validation of the Nepal Hazard Model.

INPUTS:
    - outputs/experiments/spatial_reliability.csv
    - outputs/experiments/spatial_reliability_summary.txt
OUTPUT:
    - outputs/figures/reliability_diagram.png
"""

from __future__ import annotations
from pathlib import Path

import argparse
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="outputs/experiments/spatial_reliability.csv",
        help="CSV from script 37",
    )
    parser.add_argument(
        "--summary",
        default="outputs/experiments/spatial_reliability_summary.txt",
        help="Summary txt from script 37",
    )
    parser.add_argument(
        "--outdir",
        default="outputs/figures",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    curve = (
        df.groupby("bin", as_index=False)
        .agg(
            mean_pred=("mean_pred", "mean"),
            frac_pos=("frac_pos", "mean"),
        )
        .sort_values("mean_pred")
    )

    summary_text = ""
    summary_path = Path(args.summary)
    if summary_path.exists():
        summary_text = summary_path.read_text(encoding="utf-8")

    plt.figure(figsize=(6, 6))

    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect Calibration")

    plt.plot(
        curve["mean_pred"],
        curve["frac_pos"],
        marker="o",
        label="Model"
    )

    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Landslide Frequency")
    plt.title("Reliability Diagram (Spatial CV)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    outpath = outdir / "reliability_diagram.png"
    plt.savefig(outpath, dpi=args.dpi)
    plt.close()

    print("Saved figure:", outpath)

    if summary_text:
        print("\nSummary:")
        print(summary_text)


if __name__ == "__main__":
    main()