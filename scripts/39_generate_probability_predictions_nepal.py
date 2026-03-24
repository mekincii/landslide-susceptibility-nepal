"""
National Susceptibility Inference: Point-Scale Probability Mapping

GOAL: Applies the finalized production model to the integrated Nepal dataset
to generate high-resolution landslide susceptibility probabilities.

RESPONSIBILITIES:
- Deserializes the trained Random Forest model and its associated metadata
  to ensure 100% feature alignment during inference.
- Dynamically reconstructs circular aspect features (Sin/Cos) if only raw
  degrees are present in the inference dataset.
- Executes high-performance vectorized prediction across the entire
  multidimensional feature matrix.
- Appends a 'landslide_probability' score (0.0 - 1.0) to the point-based
  geographic dataset.
- Exports a master CSV for spatial visualization and hazard zonation,
  facilitating the transition from a machine learning model to a
  policy-relevant hazard map.

INPUTS:
    - outputs/models/rf_balanced_final_nepal_xxx.joblib
    - outputs/models/rf_balanced_final_nepal_xxx_meta.json
    - data/processed/features_nepal_rxx_realxxx_clean.parquet
OUTPUT:
    - outputs/maps/nepal_landslide_probabilities_full.csv
"""

from __future__ import annotations
from pathlib import Path

import argparse
import json
import joblib
import numpy as np
import pandas as pd


def ensure_aspect_sin_cos(df: pd.DataFrame) -> pd.DataFrame:
    if "aspect_sin" in df.columns and "aspect_cos" in df.columns:
        return df

    if "aspect_deg" not in df.columns:
        raise KeyError("Need either aspect_sin/aspect_cos or aspect_deg")

    rad = np.deg2rad(df["aspect_deg"].astype(float).to_numpy())
    df = df.copy()
    df["aspect_sin"] = np.sin(rad)
    df["aspect_cos"] = np.cos(rad)

    return df


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        help="Path to trained model (.joblib)",
    )

    parser.add_argument(
        "--meta",
        required=True,
        help="Path to model metadata json",
    )

    parser.add_argument(
        "--data",
        default="data/processed/features_nepal_r05_real001_model_hydro_river_precip_clean.parquet",
    )

    parser.add_argument(
        "--outdir",
        default="outputs/maps",
    )

    args = parser.parse_args()

    model = joblib.load(args.model)

    meta = json.loads(Path(args.meta).read_text())

    features = meta["features"]

    print("Model features:", features)

    df = pd.read_parquet(args.data)

    df = ensure_aspect_sin_cos(df)

    missing = [f for f in features if f not in df.columns]

    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    X = df[features].to_numpy(dtype=float)


    probs = model.predict_proba(X)[:, 1]

    df_out = df.copy()

    df_out["landslide_probability"] = probs

    outdir = Path(args.outdir)

    outdir.mkdir(parents=True, exist_ok=True)

    out_path = outdir / "nepal_landslide_probabilities_full.csv"

    df_out.to_csv(out_path, index=False)

    print("Saved predictions to:", out_path)

    print("Probability summary:")

    print("min:", probs.min())
    print("max:", probs.max())
    print("mean:", probs.mean())


if __name__ == "__main__":
    main()