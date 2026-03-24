"""
Susceptibility Visualization: National Probability Distribution Map

GOAL: Generates a high-resolution spatial visualization of landslide
probabilities across Nepal to identify regional hazard clusters and
validate model predictions against known topographic features.

RESPONSIBILITIES:
- Automatically detects and extracts geographic coordinates (Lon/Lat or X/Y)
  from the inference CSV produced in the previous stage.
- Renders a continuous 'Pseudo-Raster' map using perceptually uniform
  color scaling (Inferno) to highlight high-risk landslide zones.
- Enforces an absolute probability scale (0.0 - 1.0) to ensure consistent
  visual interpretation across different model iterations.
- Generates publication-quality PNG output (300 DPI) for cartographic
  review and project documentation.
- Serves as the primary 'Sanity Gate' for verifying the spatial logic of
  the finalized machine learning model.

INPUT:  outputs/maps/nepal_landslide_probabilities_full.csv
OUTPUT: outputs/figures/nepal_landslide_probabilities_full_map.png
"""

from __future__ import annotations
from pathlib import Path

import argparse
import matplotlib.pyplot as plt
import pandas as pd


def find_coordinate_columns(df: pd.DataFrame):
    possible_lon = ["lon", "longitude", "x"]
    possible_lat = ["lat", "latitude", "y"]

    lon_col = None
    lat_col = None

    for c in possible_lon:
        if c in df.columns:
            lon_col = c
            break

    for c in possible_lat:
        if c in df.columns:
            lat_col = c
            break

    if lon_col is None or lat_col is None:
        raise ValueError(
            "Could not find coordinate columns. Expected lon/lat or longitude/latitude."
        )

    return lon_col, lat_col


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Probability CSV produced by script 39",
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

    if "landslide_probability" not in df.columns:
        raise ValueError("Column 'landslide_probability' not found.")

    lon_col, lat_col = find_coordinate_columns(df)

    print("Using coordinates:", lon_col, lat_col)

    x = df[lon_col]
    y = df[lat_col]
    p = df["landslide_probability"]

    plt.figure(figsize=(8, 8))

    sc = plt.scatter(
        x,
        y,
        c=p,
        s=4,
        cmap="inferno",
        vmin=0,
        vmax=1,
    )

    plt.colorbar(sc, label="Landslide Probability")

    plt.title("Nepal Landslide Susceptibility Map")

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    plt.tight_layout()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    name = Path(args.input).stem

    outpath = outdir / f"{name}_map.png"

    plt.savefig(outpath, dpi=args.dpi)

    print("Saved figure:", outpath)

    plt.close()


if __name__ == "__main__":
    main()