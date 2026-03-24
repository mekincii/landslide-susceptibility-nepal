"""
Morphometric Feature Extraction: Plan and Profile Curvature

GOAL: Derives second-order topographic features—Plan and Profile Curvature—to
characterize landform shape and its influence on surface flow dynamics.

RESPONSIBILITIES:
- Computes first and second-order partial derivatives of the DEM surface
  using NumPy's gradient method.
- Calculates 'Plan Curvature' to identify areas of flow convergence (valleys)
  and divergence (ridges).
- Calculates 'Profile Curvature' to identify areas of flow acceleration
  (convex slopes) and deceleration (concave slopes).
- Implements numerical safeguards (epsilon) to handle zero-gradient cells
  on flat terrain.
- Performs a statistical audit of the resulting morphometric layers to
  detect extreme outliers and ensure data quality.
- Exports high-resolution GeoTIFFs aligned with the original UTM DEM.

INPUT:  data/processed/dem_nepal_clipped_utm45n.tif
OUTPUTS:
    - data/processed/curv_plan_nepal.tif
    - data/processed/curv_profile_nepal.tif
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import rasterio


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

DEM_UTM = PROCESSED / "dem_nepal_clipped_utm45n.tif"

OUT_PLAN = PROCESSED / "curv_plan_nepal.tif"
OUT_PROFILE = PROCESSED / "curv_profile_nepal.tif"


def write_geotiff_like(src, out_path: Path, arr: np.ndarray, nodata_out: float = -9999.0):
    profile = src.profile.copy()
    profile.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        nodata=nodata_out,
        compress="deflate",
        tiled=False,
    )

    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)

    if out_path.exists():
        out_path.unlink()

    out = np.where(np.isnan(arr), nodata_out, arr).astype(np.float32)
    with rasterio.open(str(out_path), "w", **profile) as dst:
        dst.write(out, 1)


def main():
    if not DEM_UTM.exists():
        raise FileNotFoundError(f"Missing DEM: {DEM_UTM}")

    print("DEM:", DEM_UTM)

    with rasterio.open(DEM_UTM) as src:
        z = src.read(1).astype(np.float32)
        nodata = src.nodata
        dx = float(src.transform.a)
        dy = float(abs(src.transform.e))

        print("CRS   :", src.crs)
        print("Shape :", z.shape)
        print("Res   :", (dx, dy))
        print("Nodata:", nodata)

        # Mask nodata
        if nodata is not None:
            mask = np.isclose(z, nodata)
            z = np.where(mask, np.nan, z)

        # First derivatives p=dz/dx, q=dz/dy
        # np.gradient returns derivatives with respect to rows (y) and cols (x)
        dz_dy, dz_dx = np.gradient(z, dy, dx)

        # Second derivatives r=d2z/dx2, t=d2z/dy2 and mixed s=d2z/dxdy
        d2z_dy2, d2z_dydx = np.gradient(dz_dy, dy, dx)
        d2z_dxdy, d2z_dx2 = np.gradient(dz_dx, dy, dx)

        p = dz_dx
        q = dz_dy
        r = d2z_dx2
        s = 0.5 * (d2z_dydx + d2z_dxdy)  # symmetrized mixed derivative
        t = d2z_dy2

        eps = 1e-12
        pq2 = p * p + q * q

        # Plan curvature (Zevenbergen & Thorne style; sign conventions vary across GIS)
        # Measures convergence/divergence of flow lines
        plan = (q * q * r - 2 * p * q * s + p * p * t) / np.power(pq2 + eps, 1.5)

        # Profile curvature (along-slope curvature)
        # Measures acceleration/deceleration along flow direction
        profile = (p * p * r + 2 * p * q * s + q * q * t) / (np.power(pq2 + eps, 1.5))

        # Keep NaN where DEM was NaN
        plan = np.where(np.isnan(z), np.nan, plan)
        profile = np.where(np.isnan(z), np.nan, profile)

        print("Writing:", OUT_PLAN)
        write_geotiff_like(src, OUT_PLAN, plan)

        print("Writing:", OUT_PROFILE)
        write_geotiff_like(src, OUT_PROFILE, profile)

        def stats(a):
            a2 = a[np.isfinite(a)]
            return float(a2.min()), float(a2.max()), float(np.percentile(a2, 1)), float(np.percentile(a2, 99))

        plan_min, plan_max, plan_p1, plan_p99 = stats(plan)
        prof_min, prof_max, prof_p1, prof_p99 = stats(profile)

        print("\nCurvature Stats")
        print(f"Plan    min={plan_min:.6g}  max={plan_max:.6g}  p1={plan_p1:.6g}  p99={plan_p99:.6g}")
        print(f"Profile min={prof_min:.6g}  max={prof_max:.6g}  p1={prof_p1:.6g}  p99={prof_p99:.6g}")

        missing_pct = float(np.mean(~np.isfinite(plan)) * 100.0)
        print("Missing% (plan) :", round(missing_pct, 2))


if __name__ == "__main__":
    main()