"""
River Proximity Integration & Distance Transformation

GOAL: Samples the 'Distance-to-River' raster at training point locations
and applies logarithmic scaling to model the decaying influence of fluvial
erosion on landslide susceptibility.

RESPONSIBILITIES:
- Loads the hydro-topographic feature set, including robust handling of
  geometries stored in Well-Known Binary (WKB) format.
- Implements a window-based raster sampling routine to extract metric
  distances from the distance-to-river surface (UTM 45N).
- Applies a log transformation to the distance values to better
  represent the non-linear relationship between river proximity and
  slope instability (toe-undercutting proxy).
- Conducts a statistical quality check, reporting missing value ratios and
  percentile distributions for both raw and log-transformed distance features.
- Exports the expanded feature matrix for 'River-inclusive' model experiments.

INPUTS:
    - data/processed/features_nepal_rxx_realxxx_model_hydro.parquet
    - data/processed/dist_to_river_m_nepal.tif
OUTPUT:
    - data/processed/features_nepal_rxx_realxxx_model_hydro_river.parquet
"""

from pathlib import Path
from shapely import wkb

import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd

ROOT = Path(__file__).resolve().parents[1]

IN_DATA = ROOT / "data" / "processed" / "features_nepal_r05_real001_model_hydro.parquet"
DIST_RIVER = ROOT / "data" / "processed" / "dist_to_river_m_nepal.tif"
OUT_DATA = ROOT / "data" / "processed" / "features_nepal_r05_real001_model_hydro_river.parquet"


def sample_raster_at_points(raster_path: Path, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    with rasterio.open(raster_path) as src:
        nodata = src.nodata
        vals = []
        for x, y in zip(xs, ys):
            try:
                row, col = src.index(x, y)
                if row < 0 or col < 0 or row >= src.height or col >= src.width:
                    vals.append(np.nan)
                    continue
                v = src.read(1, window=((row, row + 1), (col, col + 1)))[0, 0]
                if nodata is not None and np.isclose(v, nodata):
                    vals.append(np.nan)
                else:
                    vals.append(float(v))
            except Exception:
                vals.append(np.nan)
        return np.array(vals, dtype=float)


def main():
    df = pd.read_parquet(IN_DATA)

    if "geometry" not in df.columns:
        raise RuntimeError("Expected a geometry column in the parquet. It should come from GeoPandas export.")

    df["geometry"] = df["geometry"].apply(
        lambda g: wkb.loads(g) if isinstance(g, (bytes, bytearray, memoryview)) else g)

    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:32645")
    xs = gdf.geometry.x.values
    ys = gdf.geometry.y.values

    dist_m = sample_raster_at_points(DIST_RIVER, xs, ys)

    gdf["dist_river_m"] = dist_m
    gdf["log_dist_river"] = np.log1p(dist_m)

    miss = pd.isna(gdf["dist_river_m"]).mean() * 100.0
    print(f"Missing dist_river_m (%): {miss:.2f}")

    print("\n=== dist_river_m stats (m) ===")
    print(gdf["dist_river_m"].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]))

    print("\n=== log_dist_river stats ===")
    print(gdf["log_dist_river"].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]))

    print("\nWriting:", OUT_DATA)
    gdf.to_parquet(OUT_DATA, index=False)


if __name__ == "__main__":
    main()