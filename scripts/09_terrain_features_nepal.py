"""
Topographic Feature Extraction & Spatial Sampling

GOAL: Calculates secondary topographic variables (Slope and Aspect) from a
Digital Elevation Model (DEM) and extracts these values for all training samples.

RESPONSIBILITIES:
- Derives Slope (gradient magnitude in degrees) and Aspect (gradient
  direction in degrees) using NumPy-based finite difference methods.
- Generates and persists high-resolution GeoTIFFs for the new terrain layers.
- Performs spatial point-sampling to append Elevation, Slope, and Aspect
  values to the landslide/non-landslide training points.
- Synchronizes Coordinate Reference Systems (CRS) between vector samples
  and raster sources to ensure spatial accuracy.
- Exports the final feature set in multiple formats (GeoJSON, Parquet, CSV)
  ready for statistical analysis or machine learning.

INPUTS:
    - data/processed/dem_nepal_clipped_utm45n.tif (Projected DEM)
    - data/processed/samples_nepal_rxx_realxxx.geojson (Labeled points)
OUTPUTS:
    - data/processed/slope_deg_...tif & aspect_deg_...tif
    - data/processed/features_...parquet (Feature matrix for modeling)
"""

from __future__ import annotations
from rasterio.transform import Affine
from pathlib import Path

import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio


def compute_slope_aspect(
    dem: np.ndarray,
    transform: Affine,
    nodata: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    dem = dem.astype(np.float32, copy=False)

    if nodata is not None:
        dem_mask = np.isclose(dem, nodata) | np.isnan(dem)
    else:
        dem_mask = np.isnan(dem)

    dem_nan = dem.copy()
    dem_nan[dem_mask] = np.nan

    xres = float(transform.a)
    yres = float(abs(transform.e))

    dz_dy, dz_dx = np.gradient(dem_nan, yres, xres)

    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    slope_deg = np.degrees(slope_rad).astype(np.float32)

    aspect_rad = np.arctan2(dz_dy, -dz_dx)
    aspect_deg = (np.degrees(aspect_rad) + 360.0) % 360.0
    aspect_deg = aspect_deg.astype(np.float32)

    invalid = dem_mask | np.isnan(dz_dx) | np.isnan(dz_dy)
    slope_deg[invalid] = np.nan
    aspect_deg[invalid] = np.nan

    return slope_deg, aspect_deg


def write_single_band_geotiff(
    out_path: Path,
    arr: np.ndarray,
    ref_profile: dict,
    nodata_val: float = -9999.0,
) -> None:
    profile = ref_profile.copy()
    profile.update(
        dtype="float32",
        count=1,
        nodata=nodata_val,
        compress="deflate",
        predictor=2,
        tiled=True,
        blockxsize=512,
        blockysize=512,
    )

    data = arr.astype(np.float32, copy=False)
    data_out = np.where(np.isnan(data), nodata_val, data).astype(np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data_out, 1)


def sample_rasters_at_points(
    gdf: gpd.GeoDataFrame,
    raster_paths: dict[str, Path],
) -> gpd.GeoDataFrame:
    gdf_out = gdf.copy()

    for key, rpath in raster_paths.items():
        with rasterio.open(rpath) as src:
            coords = [(geom.x, geom.y) for geom in gdf_out.geometry]
            vals = np.array([v[0] for v in src.sample(coords)], dtype=np.float32)

            if src.nodata is not None:
                vals = np.where(np.isclose(vals, src.nodata), np.nan, vals)

            gdf_out[key] = vals

    return gdf_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dem",
        type=str,
        default=r"data/processed/dem_nepal_clipped_utm45n.tif",
        help="UTM DEM path (meters).",
    )
    ap.add_argument(
        "--samples",
        type=str,
        default=r"data/processed/samples_nepal_r05_real001.geojson",
        help="Samples GeoJSON (pos+neg points).",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=r"data/processed",
        help="Output directory for rasters and tables.",
    )
    ap.add_argument(
        "--prefix",
        type=str,
        default="nepal_r05_real001",
        help="Prefix for output filenames.",
    )
    ap.add_argument(
        "--also_csv",
        action="store_true",
        help="Also write CSV in addition to Parquet.",
    )
    args = ap.parse_args()

    dem_path = Path(args.dem)
    samples_path = Path(args.samples)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    slope_path = out_dir / f"slope_deg_{args.prefix}.tif"
    aspect_path = out_dir / f"aspect_deg_{args.prefix}.tif"

    print(f"DEM: {dem_path}")

    with rasterio.open(dem_path) as src:
        dem = src.read(1)
        profile = src.profile
        transform = src.transform
        crs = src.crs
        nodata = src.nodata

        slope_deg, aspect_deg = compute_slope_aspect(dem, transform, nodata)

        write_single_band_geotiff(slope_path, slope_deg, profile)

        write_single_band_geotiff(aspect_path, aspect_deg, profile)

    gdf = gpd.read_file(samples_path)
    if gdf.crs is None:
        raise RuntimeError("Samples GeoJSON has no CRS. It must have a CRS to reproject.")

    if str(gdf.crs) != str(crs):
        gdf = gdf.to_crs(crs)

    raster_map = {
        "elev_m": dem_path,
        "slope_deg": slope_path,
        "aspect_deg": aspect_path,
    }
    gdf_feat = sample_rasters_at_points(gdf, raster_map)

    out_geojson = out_dir / f"features_{args.prefix}.geojson"
    out_parquet = out_dir / f"features_{args.prefix}.parquet"
    out_csv = out_dir / f"features_{args.prefix}.csv"

    gdf_feat.to_file(out_geojson, driver="GeoJSON")

    df = pd.DataFrame(gdf_feat.drop(columns="geometry"))
    df.to_parquet(out_parquet, index=False)

    if args.also_csv:
        print(f"CSV:      {out_csv}")
        df.to_csv(out_csv, index=False)


if __name__ == "__main__":
    main()
