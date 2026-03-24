"""
Spatial Visualization & Data Validation

GOAL: Produces diagnostic maps to visually verify the spatial alignment
between the Global Landslide Catalog (GLC) and the Nepal study area.

RESPONSIBILITIES:
- Loads global landslide point data and the dissolved Nepal boundary.
- Generates a global-scale overview map (01_glc_global_nepal.png) to confirm
  geographic placement.
- Performs a spatial 'within' query to identify landslide events specifically
  located inside the Nepal boundary.
- Generates a zoomed-in study area map (02_nepal_glc_points.png) with
  dynamic padding for localized inspection.
- Validates that the CRS (EPSG:4326) is consistent across both layers.

INPUTS:
    - data/processed/glc_all_points.geojson
    - data/processed/study_area_nepal.geojson
OUTPUTS:
    - outputs/figures/01_glc_global_nepal.png
    - outputs/figures/02_nepal_glc_points.png
"""

from __future__ import annotations
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GLC_PATH = PROJECT_ROOT / "data" / "processed" / "glc_all_points.geojson"
STUDY_PATH = PROJECT_ROOT / "data" / "processed" / "study_area_nepal.geojson"
OUT_DIR = PROJECT_ROOT / "outputs" / "figures"


def main() -> None:
    if not GLC_PATH.exists():
        raise FileNotFoundError(f"Missing: {GLC_PATH}")
    if not STUDY_PATH.exists():
        raise FileNotFoundError(f"Missing: {STUDY_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    glc = gpd.read_file(GLC_PATH).to_crs("EPSG:4326")
    study = gpd.read_file(STUDY_PATH).to_crs("EPSG:4326")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title("Global Landslide Catalog (GLC) and Nepal study area")

    glc.plot(ax=ax, markersize=1, alpha=0.15)
    study.boundary.plot(ax=ax, linewidth=2)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 80)

    out1 = OUT_DIR / "01_glc_global_nepal.png"
    fig.tight_layout()
    fig.savefig(out1, dpi=200)
    plt.close(fig)

    poly = study.geometry.iloc[0]
    inside = glc[glc.within(poly)].copy()

    bounds = study.total_bounds
    pad_x = (bounds[2] - bounds[0]) * 0.15
    pad_y = (bounds[3] - bounds[1]) * 0.15

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(f"Nepal – GLC landslide events (n={len(inside)})")

    study.plot(ax=ax, alpha=0.15)
    study.boundary.plot(ax=ax, linewidth=2)
    inside.plot(ax=ax, markersize=8, alpha=0.6)

    ax.set_xlim(bounds[0] - pad_x, bounds[2] + pad_x)
    ax.set_ylim(bounds[1] - pad_y, bounds[3] + pad_y)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    out2 = OUT_DIR / "02_nepal_glc_points.png"
    fig.tight_layout()
    fig.savefig(out2, dpi=250)
    plt.close(fig)


if __name__ == "__main__":
    main()
