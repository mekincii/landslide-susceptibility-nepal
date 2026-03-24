"""
Spatial Data Subset: Clipping GLC to Nepal Study Area

GOAL: Extracts a localized subset of the Global Landslide Catalog (GLC)
specifically for the Nepal study area to reduce data volume and focus analysis.

RESPONSIBILITIES:
- Loads the processed global landslide point data and the Nepal study area mask.
- Synchronizes Coordinate Reference Systems (CRS) to EPSG:4326.
- Performs a Point-in-Polygon (within) spatial query to filter landslide
  events falling inside the Nepal boundary.
- Persists the localized subset to a new GeoJSON for downstream modeling steps.

INPUTS:
    - data/processed/glc_all_points.geojson
    - data/processed/study_area_nepal.geojson
OUTPUT:
    - data/processed/glc_nepal_points.geojson
"""

from __future__ import annotations
from pathlib import Path

import geopandas as gpd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GLC_PATH = PROJECT_ROOT / "data" / "processed" / "glc_all_points.geojson"
STUDY_PATH = PROJECT_ROOT / "data" / "processed" / "study_area_nepal.geojson"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "glc_nepal_points.geojson"


def main() -> None:
    if not GLC_PATH.exists():
        raise FileNotFoundError(f"Missing: {GLC_PATH}")
    if not STUDY_PATH.exists():
        raise FileNotFoundError(f"Missing: {STUDY_PATH}")

    glc = gpd.read_file(GLC_PATH).to_crs("EPSG:4326")
    study = gpd.read_file(STUDY_PATH).to_crs("EPSG:4326")

    poly = study.geometry.iloc[0]
    inside = glc[glc.within(poly)].copy()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    inside.to_file(OUT_PATH, driver="GeoJSON")


if __name__ == "__main__":
    main()
