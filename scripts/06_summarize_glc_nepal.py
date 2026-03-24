"""
Exploratory Data Analysis: Nepal Landslide Catalog Quality Check

GOAL: Performs a comprehensive audit of the clipped Nepal landslide dataset
to assess data completeness, attribute distribution, and temporal coverage.

RESPONSIBILITIES:
- Analyzes temporal bounds (min/max dates) and yearly event distribution.
- Quantifies data gaps by calculating the percentage of missing values per column.
- Profiles key categorical attributes including landslide trigger, size,
  category, setting, and location accuracy.
- Verifies the integrity of the 'country_name' field to ensure the previous
  clipping operation was successful.
- Audits geometric validity (checking for empty or null spatial features).

INPUT:  data/processed/glc_nepal_points.geojson
OUTPUT: Console summary of data quality and attribute statistics.
"""

from __future__ import annotations
from pathlib import Path

import pandas as pd
import geopandas as gpd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IN_PATH = PROJECT_ROOT / "data" / "processed" / "glc_nepal_points.geojson"


def vc(series: pd.Series, top: int = 15) -> pd.DataFrame:
    out = (
        series.astype("object")
        .where(series.notna(), other="(missing)")
        .value_counts()
        .head(top)
        .reset_index()
    )
    out.columns = ["value", "count"]
    out["pct"] = (out["count"] / out["count"].sum() * 100).round(2)
    return out


def main() -> None:
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Missing: {IN_PATH}")

    gdf = gpd.read_file(IN_PATH)

    if "event_date" in gdf.columns:
        dates = pd.to_datetime(gdf["event_date"], errors="coerce")
        n_bad = dates.isna().sum()
        if (len(dates) - n_bad) > 0:
            print("Min date:", dates.min().date())
            print("Max date:", dates.max().date())
            print("\nEvents per year (top 15 years):")
            print(dates.dt.year.value_counts().head(15).to_string())
        print()
    else:
        print("⚠️ No 'event_date' column found.\n")

    miss = (gdf.isna().mean() * 100).sort_values(ascending=False)
    miss_df = miss.head(15).reset_index()
    miss_df.columns = ["column", "missing_%"]
    miss_df["missing_%"] = miss_df["missing_%"].round(2)
    print(miss_df.to_string(index=False))
    print()

    fields = [
        "location_accuracy",
        "landslide_trigger",
        "landslide_size",
        "landslide_category",
        "landslide_setting",
    ]

    for f in fields:
        if f in gdf.columns:
            print(f"=== {f.upper()} (top 15) ===")
            print(vc(gdf[f], top=15).to_string(index=False))
            print()
        else:
            print(f"⚠️ Column not found: {f}\n")

    if "country_name" in gdf.columns:
        print("=== COUNTRY CHECK ===")
        print(vc(gdf["country_name"], top=10).to_string(index=False))
        print()

    print("Geometry type counts:")
    print(gdf.geometry.geom_type.value_counts().to_string())
    print("Any empty geometries:", bool(gdf.geometry.is_empty.any()))
    print("Any null geometries :", bool(gdf.geometry.isna().any()))
    print()


if __name__ == "__main__":
    main()
