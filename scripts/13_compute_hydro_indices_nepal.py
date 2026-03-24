"""
Hydrological Feature Engineering: TWI and SPI Derivation

GOAL: Generates advanced hydrological indices (TWI, SPI) to model water
accumulation and erosive potential across the Nepal study area.

RESPONSIBILITIES:
- Conditions the DEM by filling depressions to ensure continuous surface
  drainage flow.
- Computes the D8 Flow Pointer and Flow Accumulation (upslope contributing area).
- Derives the Specific Contributing Area (SCA) in metric units (meters) to
  standardize flow across different raster resolutions.
- Calculates the Topographic Wetness Index (TWI) as a proxy for soil moisture
  and saturation zones.
- Calculates the Stream Power Index (SPI) to quantify the erosive power of
  surface runoff at any given point.
- Ensures strict spatial alignment between input DEM and derivative slope
  rasters to maintain mathematical integrity during cell-wise operations.

INPUTS:
    - data/processed/dem_nepal_clipped_utm45n.tif
    - data/processed/slope_deg_nepal_r05_real001.tif
OUTPUTS:
    - data/processed/hydro_nepal/twi.tif (Topographic Wetness Index)
    - data/processed/hydro_nepal/spi.tif (Stream Power Index)
    - data/processed/hydro_nepal/flow_acc_cells.tif (Raw flow accumulation)
"""

from __future__ import annotations
from pathlib import Path
from whitebox.whitebox_tools import WhiteboxTools

import numpy as np
import rasterio


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

DEM_UTM = PROCESSED / "dem_nepal_clipped_utm45n.tif"
SLOPE_DEG = PROCESSED / "slope_deg_nepal_r05_real001.tif"

OUT_DIR = PROCESSED / "hydro_nepal"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEM_FILLED = OUT_DIR / "dem_filled.tif"
D8_PTR = OUT_DIR / "d8_pointer.tif"
FLOW_ACC = OUT_DIR / "flow_acc_cells.tif"      # upstream contributing cells
SCA = OUT_DIR / "sca_m.tif"                    # approx specific contributing area (meters)
TWI = OUT_DIR / "twi.tif"
SPI = OUT_DIR / "spi.tif"


def _print_raster_info(path: Path, name: str):
    with rasterio.open(path) as src:
        arr = src.read(1, masked=True)
        print(f"\n=== {name} ===")
        print("Path :", path)
        print("CRS  :", src.crs)
        print("Shape:", arr.shape)
        print("Res  :", (src.transform.a, abs(src.transform.e)))
        missing = np.mean(arr.mask) * 100.0 if hasattr(arr, "mask") else float(np.mean(np.isnan(arr))) * 100.0
        print("Missing%:", round(missing, 2))


def main():
    if not DEM_UTM.exists():
        raise FileNotFoundError(f"Missing DEM: {DEM_UTM}")
    if not SLOPE_DEG.exists():
        raise FileNotFoundError(f"Missing slope raster: {SLOPE_DEG}")

    wbt = WhiteboxTools()
    wbt.set_working_dir(str(OUT_DIR))

    print("DEM:", DEM_UTM)
    print("Whitebox working dir:", OUT_DIR)

    wbt.fill_depressions(str(DEM_UTM), str(DEM_FILLED))
    wbt.d8_pointer(str(DEM_FILLED), str(D8_PTR))
    wbt.d8_flow_accumulation(str(DEM_FILLED), str(FLOW_ACC), "cells")

    with rasterio.open(FLOW_ACC) as acc_src, rasterio.open(SLOPE_DEG) as slope_src:
        acc = acc_src.read(1).astype(np.float32)
        acc_nodata = acc_src.nodata

        slope = slope_src.read(1).astype(np.float32)
        slope_nodata = slope_src.nodata

        if (acc_src.crs != slope_src.crs) or (acc_src.transform != slope_src.transform) or (acc_src.shape != slope_src.shape):
            raise RuntimeError("Flow accumulation raster and slope raster are not aligned. They must match CRS/transform/shape.")

        # cellsize in meters (UTM)
        cellsize = float(acc_src.transform.a)

        acc_mask = np.isclose(acc, acc_nodata) if acc_nodata is not None else np.isnan(acc)
        slope_mask = np.isclose(slope, slope_nodata) if slope_nodata is not None else np.isnan(slope)
        mask = acc_mask | slope_mask

        beta = np.deg2rad(slope)
        tan_beta = np.tan(beta)

        # a ≈ (flow_acc_cells * cell_area) / cell_width  = flow_acc_cells * cellsize
        # This yields units of meters (m^2 / m).
        sca = acc * cellsize

        eps = 1e-6
        sca_safe = np.where(mask, np.nan, sca)
        tan_safe = np.where(mask, np.nan, tan_beta)

        # TWI = ln( a / tan(beta) )
        twi = np.log((sca_safe + eps) / (tan_safe + eps)).astype(np.float32)

        # SPI = a * tan(beta)
        spi = (sca_safe * (tan_safe + eps)).astype(np.float32)

        # Write outputs with nodata=-9999
        profile = acc_src.profile.copy()
        profile.update(dtype="float32", count=1, nodata=-9999.0, compress="deflate")

        def _write(path: Path, data: np.ndarray):
            if path.exists():
                path.unlink()

            out = np.where(np.isnan(data), profile["nodata"], data).astype(np.float32)

            local_profile = profile.copy()
            local_profile.update(driver="GTiff", tiled=False)

            for k in ["blockxsize", "blockysize"]:
                local_profile.pop(k, None)

            with rasterio.open(str(path), "w", **local_profile) as dst:
                dst.write(out, 1)

        _write(SCA, sca_safe.astype(np.float32))
        _write(TWI, twi)
        _write(SPI, spi)

    print("Hydrology rasters written:")
    print("-", DEM_FILLED)
    print("-", D8_PTR)
    print("-", FLOW_ACC)
    print("-", SCA)
    print("-", TWI)
    print("-", SPI)

    _print_raster_info(FLOW_ACC, "FLOW_ACC (cells)")
    _print_raster_info(SCA, "SCA (m)")
    _print_raster_info(TWI, "TWI")
    _print_raster_info(SPI, "SPI")


if __name__ == "__main__":
    main()