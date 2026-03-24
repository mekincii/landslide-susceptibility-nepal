"""
River Network Extraction: Flow Accumulation Thresholding

GOAL: Derives a discrete binary river network from the continuous flow
accumulation surface to facilitate 'Distance to River' spatial analysis.

RESPONSIBILITIES:
- Loads the D8 Flow Accumulation raster generated in the hydrology pipeline.
- Applies a static cell-count threshold (2000 cells) to identify pixels that
  constitute a definitive drainage channel or river.
- Generates a binary 'River Mask' (1 = River, 0 = Non-River).
- Optimizes output metadata for 8-bit integer storage to minimize disk
  footprint while maintaining spatial alignment with the DEM.
- Provides the essential input for modeling slope-base erosion and
  toe-undercutting landslide triggers.

INPUT:  data/processed/hydro_nepal/flow_acc_cells.tif
OUTPUT: data/processed/river_mask_nepal.tif
"""

from pathlib import Path

import rasterio
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

FLOW_ACC_PATH = ROOT / "data" / "processed" / "hydro_nepal" / "flow_acc_cells.tif"
OUT_RIVER_PATH = ROOT / "data" / "processed" / "river_mask_nepal.tif"

THRESHOLD = 2000  # flow accumulation threshold (cells)


def main():
    print("Loading flow accumulation raster:", FLOW_ACC_PATH)
    with rasterio.open(FLOW_ACC_PATH) as src:
        acc = src.read(1)
        meta = src.meta.copy()

    print("Shape:", acc.shape)
    print("Applying threshold:", THRESHOLD)

    # river=1, non-river=0
    river_mask = (acc >= THRESHOLD).astype(np.uint8)

    meta.update(
        dtype=rasterio.uint8,
        count=1,
        nodata=0
    )

    print("Writing river mask:", OUT_RIVER_PATH)
    with rasterio.open(OUT_RIVER_PATH, "w", **meta) as dst:
        dst.write(river_mask, 1)

    print("River pixels:", int(river_mask.sum()))


if __name__ == "__main__":
    main()