"""
Hydrological Feature Sampling & Dataset Augmentation

GOAL: Appends advanced hydrological indices (TWI, SPI, SCA) to the existing
topographic feature set for landslide susceptibility modeling.

RESPONSIBILITIES:
- Loads the cleaned baseline feature set (Elevation, Slope, Aspect).
- Synchronizes Coordinate Reference Systems (CRS) between the point features
  and the hydrology rasters to ensure spatial alignment.
- Performs point-sampling across three key hydrological layers:
    - TWI: Topographic Wetness Index (soil saturation proxy).
    - SPI: Stream Power Index (erosive potential proxy).
    - SCA: Specific Contributing Area (upslope drainage magnitude).
- Conducts a data integrity audit by reporting missing value percentages
  and basic statistical distributions for the new features.
- Exports the expanded feature matrix to a new Parquet file for updated
  model training.

INPUTS:
    - data/processed/features_nepal_rxx_realxxx_clean.parquet
    - data/processed/hydro_nepal/*.tif (TWI, SPI, SCA)
OUTPUT:
    - data/processed/features_nepal_rxx_realxxx_hydro.parquet
"""

from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio


BASE = Path(__file__).resolve().parents[1]

features_path = BASE / "data/processed/features_nepal_r05_real001_clean.parquet"

hydro_dir = BASE / "data/processed/hydro_nepal"
twi_path = hydro_dir / "twi.tif"
spi_path = hydro_dir / "spi.tif"
sca_path = hydro_dir / "sca_m.tif"

out_path = BASE / "data/processed/features_nepal_r05_real001_hydro.parquet"


def sample_raster(raster_path, gdf):
    with rasterio.open(raster_path) as src:
        coords = [(geom.x, geom.y) for geom in gdf.geometry]
        values = np.array([v[0] for v in src.sample(coords)])
    return values

def main():

    gdf = gpd.read_parquet(features_path)

    with rasterio.open(twi_path) as src:
        raster_crs = src.crs

    if gdf.crs != raster_crs:
        print(f"Reprojecting {gdf.crs} → {raster_crs}")
        gdf = gdf.to_crs(raster_crs)

    gdf["twi"] = sample_raster(twi_path, gdf)
    gdf["spi"] = sample_raster(spi_path, gdf)
    gdf["sca_m"] = sample_raster(sca_path, gdf)

    print("\n=== Missing (%) ===")
    print(gdf[["twi", "spi", "sca_m"]].isna().mean() * 100)

    print("\n=== Basic Stats ===")
    print(gdf[["twi", "spi", "sca_m"]].describe())

    gdf.to_parquet(out_path, index=False)

    print("Saved:", out_path)


if __name__ == "__main__":
    main()