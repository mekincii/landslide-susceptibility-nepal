"""
Morphometric Feature Sampling & Dataset Augmentation

GOAL: Integrates Plan and Profile curvature features into the existing
hydro-topographic dataset to capture landform-driven landslide triggers.

RESPONSIBILITIES:
- Loads the hydro-augmented landslide feature set.
- Validates and synchronizes the Coordinate Reference System (CRS) between
  the vector samples and the curvature rasters.
- Performs spatial point-sampling for:
    - Plan Curvature (Flow convergence/divergence proxy).
    - Profile Curvature (Flow acceleration/deceleration proxy).
- Standardizes NoData values into manageable NumPy NaNs for downstream cleaning.
- Audits the success of the sampling operation by reporting the percentage
  of missing values for the new morphometric columns.
- Exports the expanded feature set to a new Parquet file for updated
  machine learning experimentation.

INPUTS:
    - data/processed/features_nepal_rxx_realxxx_model_hydro.parquet
    - data/processed/curv_plan_nepal.tif
    - data/processed/curv_profile_nepal.tif
OUTPUT:
    - data/processed/features_nepal_rxx_realxxx_model_hydro_curv.parquet
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

IN_PATH = PROCESSED / "features_nepal_r05_real001_model_hydro.parquet"

PLAN = PROCESSED / "curv_plan_nepal.tif"
PROF = PROCESSED / "curv_profile_nepal.tif"

OUT_PATH = PROCESSED / "features_nepal_r05_real001_model_hydro_curv.parquet"


def sample_raster(path: Path, gdf: gpd.GeoDataFrame) -> np.ndarray:
    with rasterio.open(path) as src:
        coords = [(geom.x, geom.y) for geom in gdf.geometry]
        vals = np.array([v[0] for v in src.sample(coords)], dtype=np.float32)

        if src.nodata is not None:
            vals = np.where(np.isclose(vals, src.nodata), np.nan, vals)

        return vals


def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Missing input: {IN_PATH}")
    for p in [PLAN, PROF]:
        if not p.exists():
            raise FileNotFoundError(f"Missing raster: {p}")

    gdf = gpd.read_parquet(IN_PATH)

    with rasterio.open(PLAN) as src:
        raster_crs = src.crs

    if gdf.crs != raster_crs:
        print(f"Reprojecting samples -> {raster_crs}")
        gdf = gdf.to_crs(raster_crs)

    gdf["curv_plan"] = sample_raster(PLAN, gdf)
    gdf["curv_profile"] = sample_raster(PROF, gdf)

    miss = gdf[["curv_plan", "curv_profile"]].isna().mean() * 100
    print("\n=== Missing (%) ===")
    print(miss)

    gdf.to_parquet(OUT_PATH, index=False)


if __name__ == "__main__":
    main()