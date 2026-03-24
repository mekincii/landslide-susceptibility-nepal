"""
Balanced Spatial Sampling & Uncertainty Realization

GOAL: Generates a labeled dataset (1=landslide, 0=non-landslide) by sampling
positive events under spatial uncertainty and generating pseudo-absence points.

RESPONSIBILITIES:
- Implements Monte Carlo sampling for positive events: shifts event locations
  randomly within a buffer defined by their 'location_accuracy' metadata.
- Sets up a 'negative' sampling framework to generate non-landslide points
  within the Nepal study area.
- Enforces a spatial exclusion zone (e.g., 2km) around positive events to
  ensure negative samples represent distinct, non-affected terrain.
- Controls dataset balance via a configurable negative-to-positive ratio.
- Maintains reproducibility using realization IDs and random seeds for
  stochastic uncertainty modeling.

INPUTS:
    - data/processed/glc_nepal_points.geojson
    - data/processed/study_area_nepal.geojson
OUTPUT:
    - data/processed/samples_nepal_r{ratio}_real{id}.geojson
"""

from __future__ import annotations
from pathlib import Path
from shapely.geometry import Point

import random
import geopandas as gpd
import pandas as pd
import numpy as np


NEG_RATIO = 5           # negatives per positive (1, 3, 5, 10)
REALIZATION_ID = 1      # uncertainty realization index
RANDOM_SEED = 42

MIN_BUFFER_KM = 0.25    # minimum uncertainty buffer
UNKNOWN_ACC_KM = 10.0   # mapping for unknown accuracy
NEG_MIN_DIST_KM = 2.0   # negatives must be this far from sampled positives

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GLC_PATH = PROJECT_ROOT / "data" / "processed" / "glc_nepal_points.geojson"
STUDY_PATH = PROJECT_ROOT / "data" / "processed" / "study_area_nepal.geojson"

OUT_PATH = PROJECT_ROOT / "data" / "processed" / f"samples_nepal_r{NEG_RATIO:02d}_real{REALIZATION_ID:03d}.geojson"


def acc_to_km(acc: str) -> float:
    if acc is None:
        return UNKNOWN_ACC_KM

    acc = str(acc).strip().lower()

    mapping = {
        "exact": MIN_BUFFER_KM,
        "1km": 1.0,
        "5km": 5.0,
        "10km": 10.0,
        "25km": 25.0,
        "50km": 50.0,
    }

    return max(mapping.get(acc, UNKNOWN_ACC_KM), MIN_BUFFER_KM)


def sample_point_in_polygon(poly) -> Point:
    minx, miny, maxx, maxy = poly.bounds
    while True:
        p = Point(
            random.uniform(minx, maxx),
            random.uniform(miny, maxy),
        )
        if poly.contains(p):
            return p


def main() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    if not GLC_PATH.exists():
        raise FileNotFoundError(f"Missing: {GLC_PATH}")
    if not STUDY_PATH.exists():
        raise FileNotFoundError(f"Missing: {STUDY_PATH}")

    glc = gpd.read_file(GLC_PATH).to_crs("EPSG:4326")
    study = gpd.read_file(STUDY_PATH).to_crs("EPSG:4326")

    pos_points = []
    pos_rows = []

    for _, row in glc.iterrows():
        r_km = acc_to_km(row.get("location_accuracy"))
        r_deg = r_km / 111.0  # km → degrees (approx)

        buf = row.geometry.buffer(r_deg)
        p = sample_point_in_polygon(buf)

        rec = row.copy()
        rec["label"] = 1
        rec["realization"] = REALIZATION_ID
        rec["buffer_km"] = r_km

        pos_points.append(p)
        pos_rows.append(rec)

    gdf_pos = gpd.GeoDataFrame(pos_rows, geometry=pos_points, crs="EPSG:4326")

    gdf_pos_m = gdf_pos.to_crs("EPSG:3857")
    study_m = study.to_crs("EPSG:3857")
    study_poly_m = study_m.geometry.iloc[0]

    exclusion_union = gdf_pos_m.buffer(NEG_MIN_DIST_KM * 1000.0).union_all()

    n_pos = len(gdf_pos)
    n_neg = NEG_RATIO * n_pos

    neg_points_m = []
    attempts = 0
    max_attempts = n_neg * 200

    while len(neg_points_m) < n_neg and attempts < max_attempts:
        attempts += 1
        p = sample_point_in_polygon(study_poly_m)

        if exclusion_union.contains(p):
            continue

        neg_points_m.append(p)

    if len(neg_points_m) < n_neg:
        raise RuntimeError(
            f"Only sampled {len(neg_points_m)} / {n_neg} negatives. "
            f"Reduce NEG_RATIO or NEG_MIN_DIST_KM."
        )

    gdf_neg = gpd.GeoDataFrame(
        {
            "label": [0] * n_neg,
            "realization": [REALIZATION_ID] * n_neg,
            "buffer_km": [None] * n_neg,
        },
        geometry=neg_points_m,
        crs="EPSG:3857",
    ).to_crs("EPSG:4326")

    gdf_all = gpd.GeoDataFrame(
        pd.concat([gdf_pos, gdf_neg], ignore_index=True),
        crs="EPSG:4326",
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    gdf_all.to_file(OUT_PATH, driver="GeoJSON")

    print(f"Output      : {OUT_PATH}")
    print(f"Positives   : {n_pos}")
    print(f"Negatives   : {n_neg} (ratio {NEG_RATIO}:1)")
    print(f"Realization : {REALIZATION_ID}")
    print(f"Seed        : {RANDOM_SEED}")


if __name__ == "__main__":
    main()
