"""
Dataset Sanitization: Hydro-Morphometric Feature Cleaning

GOAL: Finalizes the training dataset by removing observations with incomplete
morphometric features (Plan/Profile Curvature) and auditing class balance.

RESPONSIBILITIES:
- Loads the integrated feature set containing elevation, slope, aspect,
  hydrological indices, and curvature values.
- Performs a strict 'Complete Case Analysis' by dropping any sample points
  that return NaN for curvature (typically due to raster edge effects).
- Quantifies the data loss percentage to ensure the filtering process hasn't
  drastically reduced the sample size.
- Audits the post-cleaning class balance (Neg:Pos ratio) to detect any
  unintentional bias introduced by spatial filtering.
- Persists the high-quality, 'ML-ready' dataset to an optimized Parquet file.

INPUT:  data/processed/features_nepal_rxx_realxxx_model_hydro_curv.parquet
OUTPUT: data/processed/features_nepal_rxx_realxxx_model_hydro_curv_clean.parquet
"""

from __future__ import annotations
from pathlib import Path

import geopandas as gpd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

IN_PATH = PROCESSED / "features_nepal_r05_real001_model_hydro_curv.parquet"
OUT_PATH = PROCESSED / "features_nepal_r05_real001_model_hydro_curv_clean.parquet"

REQUIRED = [
    "elev_m",
    "slope_deg",
    "aspect_deg",
    "twi",
    "log_spi",
    "log_sca",
    "curv_plan",
    "curv_profile",
    "label",
]

def main():
    gdf = gpd.read_parquet(IN_PATH)

    before = len(gdf)
    gdf2 = gdf.dropna(subset=["curv_plan", "curv_profile"])
    dropped = before - len(gdf2)

    print(f"\nDropped rows (NaN curvature): {dropped} / {before} ({dropped/before*100:.2f}%)")
    print("Remaining rows:", len(gdf2))

    # quick class balance
    pos = int((gdf2["label"] == 1).sum())
    neg = int((gdf2["label"] == 0).sum())
    print("\n=== Class balance (after) ===")
    print("Positives:", pos)
    print("Negatives:", neg)
    if pos > 0:
        print("Neg:Pos ratio:", round(neg / pos, 2), ":1")

    gdf2.to_parquet(OUT_PATH, index=False)

if __name__ == "__main__":
    main()