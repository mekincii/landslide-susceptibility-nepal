"""
Hydrological Feature Transformation & Normalization

GOAL: Refines hydrological variables through logarithmic transformations to
reduce skewness and improve the statistical distribution for machine learning.

RESPONSIBILITIES:
- Loads the hydro-augmented feature set (Elevation, Slope, TWI, SPI, SCA).
- Applies a log transformation (ln(1+x)) to the Stream Power Index (SPI)
  and Specific Contributing Area (SCA) to handle heavy-tailed distributions.
- Normalizes high-magnitude hydrological outliers, ensuring that extreme
  drainage values do not disproportionately bias model decision splits.
- Performs a statistical audit (describe) to verify the range and distribution
  of the newly transformed 'log_spi' and 'log_sca' features.
- Exports the final 'ML-ready' dataset for the secondary baseline model run.

INPUT:  data/processed/features_nepal_rxx_realxxx_hydro.parquet
OUTPUT: data/processed/features_nepal_rxx_realxxx_model_hydro.parquet
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import geopandas as gpd

ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "data" / "processed" / "features_nepal_r05_real001_hydro.parquet"
OUT_PATH = ROOT / "data" / "processed" / "features_nepal_r05_real001_model_hydro.parquet"

def main():
    gdf = gpd.read_parquet(IN_PATH)
    print("Rows:", len(gdf))
    print("CRS :", gdf.crs)

    # Log transforms (standard for heavy-tailed hydro metrics)
    gdf["log_spi"] = np.log1p(gdf["spi"].astype(float))
    gdf["log_sca"] = np.log1p(gdf["sca_m"].astype(float))

    print(gdf[["log_spi", "log_sca", "twi"]].describe())

    gdf.to_parquet(OUT_PATH, index=False)

if __name__ == "__main__":
    main()