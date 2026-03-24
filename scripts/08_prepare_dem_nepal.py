"""
Digital Elevation Model (DEM) Pre-processing Pipeline

GOAL: Aggregates raw SRTM tiles into a seamless, clipped, and projected elevation
surface specifically formatted for topographic feature extraction (Slope, Aspect).

RESPONSIBILITIES:
- Mosaics multiple raw SRTM .tif tiles into a single continuous raster layer.
- Implements LZW compression and tiling (512x512 blocks) to optimize storage
  and subsequent read performance.
- Clips the global/regional mosaic to the specific geometry of the Nepal study area.
- Performs a critical CRS transformation from WGS84 (degrees) to UTM Zone 45N (meters).
- Uses Bilinear Resampling during reprojection to maintain elevation surface
  continuity for derivative topographic calculations.

INPUTS:
    - data/raw/dem_srtm_nepal/*.tif (Raw SRTM tiles)
    - data/processed/study_area_nepal.geojson
OUTPUTS:
    - data/processed/dem_nepal_mosaic_epsg4326.tif (Full mosaic)
    - data/processed/dem_nepal_clipped_epsg4326.tif (Clipped WGS84)
    - data/processed/dem_nepal_clipped_utm45n.tif (Projected for modeling)
"""

from __future__ import annotations
from pathlib import Path
from rasterio.merge import merge
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling

import sys
import geopandas as gpd
import numpy as np
import rasterio


ROOT = Path(__file__).resolve().parents[1]

RAW_TILES_DIR = ROOT / "data" / "raw" / "dem_srtm_nepal"
STUDY_AREA_GEOJSON = ROOT / "data" / "processed" / "study_area_nepal.geojson"

OUT_DIR = ROOT / "data" / "processed"
OUT_MOSAIC = OUT_DIR / "dem_nepal_mosaic_epsg4326.tif"
OUT_CLIP = OUT_DIR / "dem_nepal_clipped_epsg4326.tif"
OUT_UTM = OUT_DIR / "dem_nepal_clipped_utm45n.tif"  # projected DEM for slope/aspect


def die(msg: str, code: int = 1) -> None:
    print(f"{msg}")
    sys.exit(code)


def mosaic_tiles(tile_paths: list[Path]) -> tuple[np.ndarray, dict]:
    srcs = [rasterio.open(p) for p in tile_paths]
    try:
        mosaic, out_transform = merge(srcs)
        out_meta = srcs[0].meta.copy()
        out_meta.update(
            {
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_transform,
                "compress": "lzw",
                "tiled": True,
                "blockxsize": 512,
                "blockysize": 512,
            }
        )
        return mosaic, out_meta
    finally:
        for s in srcs:
            s.close()


def clip_raster_to_polygon(raster_path: Path, polygon_gdf: gpd.GeoDataFrame, out_path: Path) -> None:
    with rasterio.open(raster_path) as src:
        poly = polygon_gdf.to_crs(src.crs)
        geoms = [geom for geom in poly.geometry if geom is not None]
        if not geoms:
            die("Study area geometry is empty after reprojection.")

        out_img, out_transform = mask(src, geoms, crop=True)
        out_meta = src.meta.copy()
        out_meta.update(
            {
                "height": out_img.shape[1],
                "width": out_img.shape[2],
                "transform": out_transform,
                "compress": "lzw",
                "tiled": True,
                "blockxsize": 512,
                "blockysize": 512,
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **out_meta) as dst:
        dst.write(out_img)


def reproject_raster(in_path: Path, out_path: Path, dst_crs: str) -> None:
    with rasterio.open(in_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        meta = src.meta.copy()
        meta.update(
            {
                "crs": dst_crs,
                "transform": transform,
                "width": width,
                "height": height,
                "compress": "lzw",
                "tiled": True,
                "blockxsize": 512,
                "blockysize": 512,
            }
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **meta) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear,
                )


def main() -> None:
    if not RAW_TILES_DIR.exists():
        die(f"Tiles folder not found: {RAW_TILES_DIR}")

    tile_paths = sorted(RAW_TILES_DIR.glob("*.tif"))
    if len(tile_paths) == 0:
        die(f"No .tif found in: {RAW_TILES_DIR}")

    print(f"Found {len(tile_paths)} DEM tiles")

    crs_set = set()
    for p in tile_paths[:5]:
        with rasterio.open(p) as src:
            crs_set.add(str(src.crs))
    print(f"Sample CRS (first 5): {crs_set}")

    if not STUDY_AREA_GEOJSON.exists():
        die(f"Study area not found: {STUDY_AREA_GEOJSON}")
    study = gpd.read_file(STUDY_AREA_GEOJSON)
    if study.empty:
        die("Study area GeoJSON is empty.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Mosaicking tiles...")
    mosaic, meta = mosaic_tiles(tile_paths)
    with rasterio.open(OUT_MOSAIC, "w", **meta) as dst:
        dst.write(mosaic)
    print(f"Mosaic saved: {OUT_MOSAIC}")


    clip_raster_to_polygon(OUT_MOSAIC, study, OUT_CLIP)
    print(f"Clipped DEM saved: {OUT_CLIP}")

    reproject_raster(OUT_CLIP, OUT_UTM, "EPSG:32645")
    print(f"UTM DEM saved: {OUT_UTM}")


if __name__ == "__main__":
    main()
