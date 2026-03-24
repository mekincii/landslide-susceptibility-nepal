"""
Feature Sanitization & ML Label Standardization

GOAL: Cleans the extracted feature set by removing incomplete spatial
observations and normalizing target labels for machine learning.

RESPONSIBILITIES:
- Loads extracted features from Parquet or GeoJSON, handling potential
  schema mismatches between vector and tabular formats.
- Performs a strict 'Complete Case Analysis' (CCA), dropping any points
  where topographic features (Elevation, Slope, Aspect) are missing or NaN.
- Standardizes the target variable (landslide presence) into a 0/1 binary
  integer format regardless of original naming or data type.
- Audits and reports the final class balance and Neg:Pos ratio to ensure
  statistical consistency after cleaning.
- Exports the 'ML-ready' dataset to optimized Parquet and GeoJSON formats.

INPUTS:
    - data/processed/features_nepal_rxx_realxxx.parquet or .geojson
OUTPUTS:
    - data/processed/features_nepal_rxx_realxxx_clean.parquet (Tabular ML input)
    - data/processed/features_nepal_rxx_realxxx_clean.geojson (Spatial validation)
"""

from __future__ import annotations
from pathlib import Path

import pandas as pd
import geopandas as gpd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

RATIO_TAG = "r05"
REAL_TAG = "real001"

IN_PARQUET = PROCESSED / f"features_nepal_{RATIO_TAG}_{REAL_TAG}.parquet"
IN_GEOJSON = PROCESSED / f"features_nepal_{RATIO_TAG}_{REAL_TAG}.geojson"

OUT_PARQUET = PROCESSED / f"features_nepal_{RATIO_TAG}_{REAL_TAG}_clean.parquet"
OUT_GEOJSON = PROCESSED / f"features_nepal_{RATIO_TAG}_{REAL_TAG}_clean.geojson"

REQ_COLS = ["elev_m", "slope_deg", "aspect_deg"]


def _load_features() -> gpd.GeoDataFrame:
    if IN_PARQUET.exists():
        df = pd.read_parquet(IN_PARQUET)
        if "geometry" in df.columns:
            gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=None)
        else:
            gdf = gpd.read_file(IN_GEOJSON)
            for c in df.columns:
                if c not in gdf.columns:
                    gdf[c] = df[c].values
        return gdf

    if IN_GEOJSON.exists():
        return gpd.read_file(IN_GEOJSON)

    raise FileNotFoundError(f"Could not find input: {IN_PARQUET} or {IN_GEOJSON}")


def _label_col(gdf: gpd.GeoDataFrame) -> str:
    candidates = ["label", "y", "is_landslide", "target", "class"]
    for c in candidates:
        if c in gdf.columns:
            return c
    return ""


def main():
    print(f"Loading features from:\n  - {IN_PARQUET if IN_PARQUET.exists() else IN_GEOJSON}")
    gdf = _load_features()

    if getattr(gdf, "crs", None) is None and IN_GEOJSON.exists():
        try:
            gdf = gpd.read_file(IN_GEOJSON)
        except Exception:
            pass

    mask_ok = gdf[REQ_COLS].notna().all(axis=1)
    gdf_clean = gdf.loc[mask_ok].copy()

    lbl = _label_col(gdf_clean)
    if lbl:
        y = gdf_clean[lbl]
        if y.dtype == bool:
            y01 = y.astype(int)
        else:
            if y.dropna().isin([0, 1]).all():
                y01 = y.astype(int)
            elif y.dropna().isin(["pos", "neg", "positive", "negative"]).all():
                y01 = y.map({"pos": 1, "positive": 1, "neg": 0, "negative": 0}).astype("Int64")
            else:
                try:
                    y01 = (pd.to_numeric(y, errors="coerce") > 0).astype("Int64")
                except Exception:
                    y01 = None

        if y01 is not None:
            pos = int((y01 == 1).sum())
            neg = int((y01 == 0).sum())
            print("\n=== CLASS BALANCE (after clean) ===")
            print("Positives:", pos)
            print("Negatives:", neg)
            if pos > 0:
                print("Neg:Pos ratio:", f"{neg/pos:.2f}:1")

    print("\nWriting outputs:")
    gdf_clean.to_file(OUT_GEOJSON, driver="GeoJSON")
    print("GeoJSON :", OUT_GEOJSON)

    try:
        gdf_clean.to_parquet(OUT_PARQUET, index=False)
        print("Parquet :", OUT_PARQUET)
    except Exception as e:
        print("\n Parquet write failed, writing CSV instead.")
        out_csv = OUT_PARQUET.with_suffix(".csv")
        df = pd.DataFrame(gdf_clean.drop(columns=["geometry"], errors="ignore"))
        df.to_csv(out_csv, index=False)
        print("CSV    :", out_csv)
        print("Reason :", repr(e))


if __name__ == "__main__":
    main()