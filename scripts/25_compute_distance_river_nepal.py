"""
Spatial Proximity Analysis: Euclidean Distance to River Network

GOAL: Computes the continuous metric distance (in meters) from every landscape
pixel to the nearest river channel to model fluvial undercut risk.

RESPONSIBILITIES:
- Loads the binary river mask generated via flow accumulation thresholding.
- Utilizes the Euclidean Distance Transform (EDT) to calculate the
  shortest-path distance from non-river pixels to the drainage network.
- Scales the raw pixel-distance results by the raster resolution (meters)
  to ensure the output is physically grounded in UTM units.
- Generates a continuous 'Distance-to-River' surface as a float32 GeoTIFF,
  aligned with the project's master DEM and hydrology layers.
- Provides a quantitative proxy for slope-base erosion and river-induced
  destabilization in the Nepal study area.

INPUT:  data/processed/river_mask_nepal.tif
OUTPUT: data/processed/dist_to_river_m_nepal.tif
"""

from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]

RIVER_MASK_PATH = ROOT / "data" / "processed" / "river_mask_nepal.tif"
OUT_DIST_PATH = ROOT / "data" / "processed" / "dist_to_river_m_nepal.tif"


def main():
    with rasterio.open(RIVER_MASK_PATH) as src:
        river = src.read(1)
        meta = src.meta.copy()
        res_x, res_y = src.res  # meters (UTM)
        nodata = src.nodata

    print("Shape:", river.shape, "| Res (m):", (res_x, res_y), "| nodata:", nodata)

    # river cells are 1, non-river 0, nodata 0 (we wrote nodata=0)
    river_bool = (river == 1)
    inv = (~river_bool).astype(np.uint8)

    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError as e:
        raise ImportError("scipy is required. Run: pip install scipy") from e

    dist_pix = distance_transform_edt(inv)  # in pixels
    dist_m = dist_pix * float(res_x)

    # Save as float32 GeoTIFF
    meta.update(dtype=rasterio.float32, count=1, nodata=None)

    print("Writing distance raster:", OUT_DIST_PATH)
    with rasterio.open(OUT_DIST_PATH, "w", **meta) as dst:
        dst.write(dist_m.astype(np.float32), 1)

    print("Distance stats (m):",
          "min=", float(np.nanmin(dist_m)),
          "max=", float(np.nanmax(dist_m)),
          "mean=", float(np.nanmean(dist_m)))


if __name__ == "__main__":
    main()