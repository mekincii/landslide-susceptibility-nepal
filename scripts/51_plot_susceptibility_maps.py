"""
Comparative Susceptibility Synthesis: ML vs. Expert Knowledge

GOAL: Generates high-fidelity comparative visualizations to benchmark the
Random Forest (Data-Driven) against the Fuzzy Inference System (Knowledge-Driven).

RESPONSIBILITIES:
- Normalizes disparate prediction outputs into a unified 'Susceptibility Score'
  framework for objective comparison.
- Produces individual high-resolution maps of the Nepal study area to
  audit the spatial distribution of risk for both modeling paradigms.
- Generates a side-by-side 'Divergence Map' to highlight geographic areas
  where machine logic and expert geomorphological rules agree or conflict.
- Utilizes perceptually uniform color scaling (Inferno) and absolute
  normalization (0.0-1.0) to ensure cartographic integrity.
- Facilitates the final qualitative validation step before selecting the
  optimal hazard map for national policy implementation.

INPUTS:
    - outputs/maps/nepal_landslide_probabilities_reduced.csv (RF)
    - outputs/maps/nepal_fuzzy_probabilities.csv (FIS)
OUTPUTS:
    - outputs/figures/rf_vs_fuzzy_susceptibility_points.png
    - Individual model maps (PNG)
"""

from __future__ import annotations
from pathlib import Path

import argparse
import matplotlib.pyplot as plt
import pandas as pd


RF_DEFAULT = r"outputs\maps\nepal_landslide_probabilities_reduced.csv"
FUZZY_DEFAULT = r"outputs\maps\nepal_fuzzy_probabilities.csv"
OUTDIR_DEFAULT = r"outputs\figures"


def find_coord_cols(df: pd.DataFrame):
    lon_candidates = ["longitude", "lon", "x"]
    lat_candidates = ["latitude", "lat", "y"]

    lon_col = next((c for c in lon_candidates if c in df.columns), None)
    lat_col = next((c for c in lat_candidates if c in df.columns), None)

    if lon_col is None or lat_col is None:
        raise ValueError("Could not find coordinate columns.")

    return lon_col, lat_col


def prepare_rf(df: pd.DataFrame) -> pd.DataFrame:
    lon_col, lat_col = find_coord_cols(df)

    if "landslide_probability" not in df.columns:
        raise ValueError("RF CSV must contain 'landslide_probability'.")

    df = df.dropna(subset=[lon_col, lat_col, "landslide_probability"]).copy()

    return df[[lon_col, lat_col, "landslide_probability"]].rename(
        columns={
            lon_col: "longitude",
            lat_col: "latitude",
            "landslide_probability": "score",
        }
    )


def prepare_fuzzy(df: pd.DataFrame) -> pd.DataFrame:
    lon_col, lat_col = find_coord_cols(df)

    if "fuzzy_prob" not in df.columns:
        raise ValueError("Fuzzy CSV must contain 'fuzzy_prob'.")

    df = df.dropna(subset=[lon_col, lat_col, "fuzzy_prob"]).copy()

    return df[[lon_col, lat_col, "fuzzy_prob"]].rename(
        columns={
            lon_col: "longitude",
            lat_col: "latitude",
            "fuzzy_prob": "score",
        }
    )


def plot_single(df: pd.DataFrame, title: str, outpath: Path, dpi: int = 300):
    plt.figure(figsize=(7, 7))

    sc = plt.scatter(
        df["longitude"],
        df["latitude"],
        c=df["score"],
        s=10,
        cmap="inferno",
        vmin=0.0,
        vmax=1.0,
    )

    plt.colorbar(sc, label="Susceptibility Score")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=dpi)
    plt.close()


def plot_side_by_side(df_rf: pd.DataFrame, df_fuzzy: pd.DataFrame, outpath: Path, dpi: int = 300):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

    sc1 = axes[0].scatter(
        df_rf["longitude"],
        df_rf["latitude"],
        c=df_rf["score"],
        s=10,
        cmap="inferno",
        vmin=0.0,
        vmax=1.0,
    )
    axes[0].set_title("RF Reduced Susceptibility")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")

    sc2 = axes[1].scatter(
        df_fuzzy["longitude"],
        df_fuzzy["latitude"],
        c=df_fuzzy["score"],
        s=10,
        cmap="inferno",
        vmin=0.0,
        vmax=1.0,
    )
    axes[1].set_title("Fuzzy Rule Susceptibility")
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")

    cbar = fig.colorbar(sc2, ax=axes.ravel().tolist(), shrink=0.9)
    cbar.set_label("Susceptibility Score")

    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--rf", default=RF_DEFAULT)
    parser.add_argument("--fuzzy", default=FUZZY_DEFAULT)
    parser.add_argument("--outdir", default=OUTDIR_DEFAULT)
    parser.add_argument("--dpi", type=int, default=300)
 
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rf_df_raw = pd.read_csv(args.rf)
    fuzzy_df_raw = pd.read_csv(args.fuzzy)

    rf_df = prepare_rf(rf_df_raw)
    fuzzy_df = prepare_fuzzy(fuzzy_df_raw)

    print(f"RF mapped rows    : {len(rf_df)}")
    print(f"Fuzzy mapped rows : {len(fuzzy_df)}")

    rf_out = outdir / "rf_reduced_susceptibility_points.png"
    fuzzy_out = outdir / "fuzzy_rule_susceptibility_points.png"
    compare_out = outdir / "rf_vs_fuzzy_susceptibility_points.png"

    plot_single(rf_df, "RF Reduced Susceptibility at Mapped Points", rf_out, dpi=args.dpi)
    plot_single(fuzzy_df, "Fuzzy Rule Susceptibility at Mapped Points", fuzzy_out, dpi=args.dpi)
    plot_side_by_side(rf_df, fuzzy_df, compare_out, dpi=args.dpi)

    print("Saved:", rf_out)
    print("Saved:", fuzzy_out)
    print("Saved:", compare_out)


if __name__ == "__main__":
    main()