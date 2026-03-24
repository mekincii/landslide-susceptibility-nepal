"""
Precipitation Data Normalization & Grid Alignment

GOAL: Synchronizes coarse global precipitation data (CHELSA Bio12) with the
high-resolution UTM DEM grid used for the Nepal study area.

RESPONSIBILITIES:
- Loads the master DEM to extract the target spatial metadata (CRS,
  Resolution, and Bounding Box).
- Clips the raw global precipitation raster to the localized study area
  extent to optimize processing memory.
- Reprojects the precipitation data from WGS84 (degrees) to UTM Zone 45N
  (meters) to match the project's coordinate system.
- Executes a 'Template-Based' resampling to force the precipitation pixels
  into the exact grid alignment and shape of the DEM.
- Utilizes Bilinear Interpolation to maintain a smooth, continuous surface
  during the upsampling process.

INPUTS:
    - data/raw/precip/precip_bio12_chelsa.tif
    - data/processed/dem_nepal_clipped_utm45n.tif
OUTPUTS:
    - data/processed/precip_bio12_nepal_utm45n_aligned.tif (Primary)
    - Intermediate clipped and projected GeoTIFFs.
"""

from pathlib import Path
from shapely.geometry import box
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask

import rasterio
import geopandas as gpd


ROOT = Path(__file__).resolve().parents[1]

PRECIP_PATH = ROOT / "data" / "raw" / "precip" / "precip_bio12_chelsa.tif"
DEM_PATH = ROOT / "data" / "processed" / "dem_nepal_clipped_utm45n.tif"

OUT_CLIPPED = ROOT / "data" / "processed" / "precip_bio12_nepal_clipped.tif"
OUT_UTM = ROOT / "data" / "processed" / "precip_bio12_nepal_utm45n.tif"
OUT_ALIGNED = ROOT / "data" / "processed" / "precip_bio12_nepal_utm45n_aligned.tif"


def main():
    print("Loading DEM")
    with rasterio.open(DEM_PATH) as dem:
        dem_bounds = dem.bounds
        dem_crs = dem.crs
        dem_transform = dem.transform
        dem_shape = (dem.height, dem.width)

    print("Loading CHELSA precipitation")
    with rasterio.open(PRECIP_PATH) as src:
        print("CRS:", src.crs)
        print("Res:", src.res)

        # Step 1: Clip to DEM bounding box (convert DEM bounds to precip CRS)
        dem_bounds_geom = gpd.GeoDataFrame(
            geometry=[box(*dem_bounds)],
            crs=dem_crs
        ).to_crs(src.crs)

        clipped_img, clipped_transform = mask(
            src,
            dem_bounds_geom.geometry,
            crop=True
        )

        meta = src.meta.copy()
        meta.update({
            "height": clipped_img.shape[1],
            "width": clipped_img.shape[2],
            "transform": clipped_transform
        })

        with rasterio.open(OUT_CLIPPED, "w", **meta) as dst:
            dst.write(clipped_img)

    # Step 2: Reproject to UTM 45N
    print("Reprojecting to UTM 45N...")
    with rasterio.open(OUT_CLIPPED) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dem_crs, src.width, src.height, *src.bounds
        )

        kwargs = src.meta.copy()
        kwargs.update({
            "crs": dem_crs,
            "transform": transform,
            "width": width,
            "height": height
        })

        with rasterio.open(OUT_UTM, "w", **kwargs) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dem_crs,
                resampling=Resampling.bilinear
            )


    # Step 3: Align exactly to DEM grid
    with rasterio.open(OUT_UTM) as src:
        aligned = src.read(
            out_shape=(1, dem_shape[0], dem_shape[1]),
            resampling=Resampling.bilinear
        )

        meta = src.meta.copy()
        meta.update({
            "height": dem_shape[0],
            "width": dem_shape[1],
            "transform": dem_transform
        })

        with rasterio.open(OUT_ALIGNED, "w", **meta) as dst:
            dst.write(aligned)

    print("Precipitation raster aligned to DEM grid.")
    print("Saved:", OUT_ALIGNED)


if __name__ == "__main__":
    main()