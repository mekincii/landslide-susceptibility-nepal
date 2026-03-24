"""
GLC Raw Data Processor: CSV to GeoJSON Converter

GOAL: Converts raw Global Land Cover (GLC) tabular data into a spatially-enabled
GeoJSON format to facilitate geographic analysis and mapping.

RESPONSIBILITIES:
- Locates raw CSV data relative to the project root directory.
- Validates data integrity (checks for missing files and required coordinate columns).
- Filters out records with null coordinates to ensure spatial consistency.
- Projects the data into the WGS 84 (EPSG:4326) coordinate reference system.
- Persists the processed GeoDataFrame to the 'data/processed' directory.

INPUT:  data/raw/glc/glc.csv (Must contain 'longitude' and 'latitude' columns)
OUTPUT: data/processed/glc_all_points.geojson
"""

from __future__ import annotations
from pathlib import Path
from shapely.geometry import Point

import pandas as pd
import geopandas as gpd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_CSV = PROJECT_ROOT / "data" / "raw" / "glc" / "glc.csv"
OUT_GEOJSON = PROJECT_ROOT / "data" / "processed" / "glc_all_points.geojson"


def main() -> None:
    if not RAW_CSV.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {RAW_CSV}\n"
            f"Expected path: data/raw/glc/glc.csv"
        )

    OUT_GEOJSON.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_CSV)
    required_cols = {"longitude", "latitude"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in CSV: {sorted(missing)}\n"
            f"Available columns: {list(df.columns)}"
        )

    df = df.dropna(subset=["longitude", "latitude"]).copy()

    geometry = [Point(xy) for xy in zip(df["longitude"], df["latitude"])]

    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    gdf.to_file(OUT_GEOJSON, driver="GeoJSON")



if __name__ == "__main__":
    main()
