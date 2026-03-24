"""
Climatological Feature Sampling: Annual Precipitation Integration

GOAL: Attaches long-term annual precipitation totals (CHELSA Bio12) to
the landslide feature set to model climatic controls on slope stability.

RESPONSIBILITIES:
- Extracts spatial coordinates from the integrated feature matrix, with
  support for both raw X/Y columns and encoded WKB geometries.
- Performs high-speed point-sampling of the aligned precipitation raster
  to retrieve localized rainfall values for all training observations.
- Applies a log transformation to handle the high variance of Himalayan
  monsoon precipitation and stabilize model training.
- Conducts a statistical audit of the rainfall distribution (percentiles)
  to ensure the full climatic gradient of Nepal is represented.
- Persists the augmented dataset, now containing Topographic, Hydrologic,
  Fluvial, and Climatological features.

INPUTS:
    - data/processed/features_nepal_rxx_realxxx_model_hydro_river.parquet
    - data/processed/precip_bio12_nepal_utm45n_aligned.tif
OUTPUT:
    - data/processed/features_nepal_rxx_realxxx_model_hydro_river_precip.parquet
"""

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

IN_PARQUET = Path("data/processed/features_nepal_r05_real001_model_hydro_river.parquet")
PRECIP_RASTER = Path("data/processed/precip_bio12_nepal_utm45n_aligned.tif")
OUT_PARQUET = Path("data/processed/features_nepal_r05_real001_model_hydro_river_precip.parquet")

# Column names
COL_PRECIP = "precip_bio12"
COL_LOGP = "log_precip_bio12"


def _parse_geometry(df: pd.DataFrame):
    if "x" in df.columns and "y" in df.columns:
        return df["x"].to_numpy(), df["y"].to_numpy()

    if "geometry" not in df.columns:
        raise ValueError("No geometry column found (expected 'geometry' or 'x'/'y').")

    try:
        from shapely import wkb
        xs, ys = [], []
        for g in df["geometry"].values:
            p = wkb.loads(g) if isinstance(g, (bytes, bytearray, memoryview)) else g
            xs.append(p.x)
            ys.append(p.y)
        return np.array(xs, dtype="float64"), np.array(ys, dtype="float64")
    except Exception as e:
        raise TypeError(
            "Geometry column is not parseable. "
            "If you have 'x'/'y' columns, prefer those. "
            f"Original error: {e}"
        )


def sample_raster_at_points(raster_path: Path, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    with rasterio.open(raster_path) as src:
        coords = list(zip(xs, ys))
        vals = list(src.sample(coords))
        vals = np.array(vals, dtype="float64").reshape(-1)  # single-band
        nodata = src.nodata
        if nodata is not None:
            vals = np.where(vals == nodata, np.nan, vals)
        return vals


def main():
    df = pd.read_parquet(IN_PARQUET)

    print(f"Sampling precipitation raster: {PRECIP_RASTER.resolve()}")
    xs, ys = _parse_geometry(df)
    precip = sample_raster_at_points(PRECIP_RASTER, xs, ys)

    df[COL_PRECIP] = precip
    df[COL_LOGP] = np.log1p(df[COL_PRECIP])

    missing_pct = float(df[COL_PRECIP].isna().mean() * 100.0)
    print(f"Missing {COL_PRECIP} (%): {missing_pct:.2f}")

    print("\nprecip stats")
    print(df[COL_PRECIP].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]))

    print("\nlog_precip stats")
    print(df[COL_LOGP].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]))

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting: {OUT_PARQUET.resolve()}")
    df.to_parquet(OUT_PARQUET, index=False)


if __name__ == "__main__":
    main()