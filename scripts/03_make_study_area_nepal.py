"""
Study Area Definition: Nepal Boundary Normalization

GOAL: Establishes a unified spatial "mask" for the project by processing
Nepal's administrative boundary into a single, dissolved polygon.

RESPONSIBILITIES:
- Loads the raw Nepal ADM0 (national level) boundary file.
- Performs a 'dissolve' operation to merge any fragmented geometries into one
  continuous study area.
- Standardizes the Coordinate Reference System (CRS) to WGS 84 (EPSG:4326)
  to match the rest of the project pipeline.
- Outputs a simplified GeoJSON to be used for clipping and sampling masks.

INPUT:  data/raw/boundaries/nepal_adm0.geojson
OUTPUT: data/processed/study_area_nepal.geojson
"""

from __future__ import annotations
from pathlib import Path

import geopandas as gpd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

IN_PATH = PROJECT_ROOT / "data" / "raw" / "boundaries" / "nepal_adm0.geojson"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "study_area_nepal.geojson"


def main() -> None:
    if not IN_PATH.exists():
        raise FileNotFoundError(
            f"Nepal ADM0 boundary not found:\n{IN_PATH}\n\n"
            f"Expected file: data/raw/boundaries/nepal_adm0.geojson"
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    gdf = gpd.read_file(IN_PATH)

    study = gdf.dissolve()

    study = study.to_crs("EPSG:4326")

    study.to_file(OUT_PATH, driver="GeoJSON")


if __name__ == "__main__":
    main()
