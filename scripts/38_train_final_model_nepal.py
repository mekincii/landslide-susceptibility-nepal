"""
Final Production Model Fitting & Serialization

GOAL: Trains the definitive landslide susceptibility model for Nepal using
the optimized 'Balanced' Random Forest configuration and persists the
results for large-scale spatial inference.

RESPONSIBILITIES:
- Fits a high-capacity Random Forest ensemble (800 trees) on the complete
  Nepal training set, incorporating lessons from previous spatial CV runs.
- Supports dual feature-set configurations: 'Full' (all metrics) or
  'Reduced' (optimized via permutation importance to maximize generalization).
- Implements regularization (min_samples_leaf=2) to ensure a smooth
  probability surface and prevent over-fitting to local topographic noise.
- Serializes the trained model using Joblib for high-performance deployment.
- Generates a JSON metadata manifest containing feature names, data sources,
  and model parameters to ensure full lineage and reproducibility.

INPUT:  data/processed/features_nepal_rxx_realxxx_model_hydro_river_precip_clean.parquet
OUTPUTS: outputs/models/rf_balanced_final_nepal_xxx.joblib (Model Weights)
         outputs/models/rf_balanced_final_nepal_xxx_meta.json (Metadata)
"""

from __future__ import annotations
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

import argparse
import json
import joblib
import numpy as np
import pandas as pd


DATA_DEFAULT = r"data\processed\features_nepal_r05_real001_model_hydro_river_precip_clean.parquet"
LABEL_COL = "label"


def ensure_aspect_sin_cos(df: pd.DataFrame) -> pd.DataFrame:
    if "aspect_sin" in df.columns and "aspect_cos" in df.columns:
        return df

    if "aspect_deg" not in df.columns:
        raise KeyError("Need either aspect_sin/aspect_cos or aspect_deg in the dataset.")

    rad = np.deg2rad(df["aspect_deg"].astype(float).to_numpy())
    df = df.copy()
    df["aspect_sin"] = np.sin(rad)
    df["aspect_cos"] = np.cos(rad)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA_DEFAULT)
    ap.add_argument("--outdir", default=r"outputs\models")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--feature-set",
        choices=["reduced", "full"],
        default="reduced",
        help="Use reduced final features from permutation importance, or the full best dataset features.",
    )
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    df = ensure_aspect_sin_cos(df)

    # Final feature choices
    if args.feature_set == "reduced":
        features = [
            "elev_m",
            "log_precip_bio12",
            "aspect_sin",
            "aspect_cos",
            "log_spi",
            "twi",
        ]
        model_name = "rf_balanced_final_nepal_reduced"
    else:
        features = [
            "elev_m",
            "slope_deg",
            "aspect_sin",
            "aspect_cos",
            "twi",
            "log_spi",
            "log_sca",
            "log_dist_river",
            "log_precip_bio12",
        ]
        model_name = "rf_balanced_final_nepal_full"

    required = features + [LABEL_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df = df.dropna(subset=required).copy()

    X = df[features].to_numpy(dtype=float)
    y = df[LABEL_COL].to_numpy(dtype=int)

    print(f"Rows used: {len(df)}")
    print(f"Positive ratio: {y.mean():.4f}")
    print(f"Feature set ({args.feature_set}): {features}")

    model = RandomForestClassifier(
        n_estimators=800,
        random_state=args.seed,
        n_jobs=-1,
        class_weight="balanced",
        min_samples_leaf=2,
    )

    model.fit(X, y)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    model_path = outdir / f"{model_name}.joblib"
    meta_path = outdir / f"{model_name}_meta.json"

    joblib.dump(model, model_path)

    meta = {
        "model_name": model_name,
        "feature_set": args.feature_set,
        "features": features,
        "label_col": LABEL_COL,
        "rows_used": int(len(df)),
        "positive_ratio": float(y.mean()),
        "model_class": "RandomForestClassifier",
        "model_params": model.get_params(),
        "source_data": str(Path(args.data)),
        "notes": (
            "Reduced feature set selected using spatial permutation importance."
            if args.feature_set == "reduced"
            else "Full feature set from best spatial-CV-performing configuration."
        ),
    }

    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Saved model:", model_path)
    print("Saved meta :", meta_path)


if __name__ == "__main__":
    main()