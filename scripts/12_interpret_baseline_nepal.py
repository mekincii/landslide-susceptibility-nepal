"""
Model Training, Inference & Feature Importance Analysis

GOAL: Trains the final baseline Random Forest model on the complete Nepal
dataset to generate landslide susceptibility probabilities and evaluate
topographic variable influence.

RESPONSIBILITIES:
- Fits a high-capacity Random Forest classifier (300 trees) using the full
  set of cleaned topographic features.
- Generates continuous probability scores (0.0 to 1.0) for the training points
  to identify high-confidence predictions and difficult-to-model outliers.
- Quantifies 'Feature Importance' to determine the relative impact of Elevation,
  Slope, and Aspect (Sine/Cosine) on landslide occurrence.
- Performs diagnostic printouts of True Positives and False Negatives to
  assist in identifying model bias or gaps in the topographic feature set.
- Exports the enriched feature matrix containing both original data and
  predicted probabilities for final spatial mapping.

INPUT:  data/processed/features_nepal_rxx_realxxx_clean.parquet
OUTPUT: data/processed/features_nepal_rxx_realxxx_with_probs.parquet
"""

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "features_nepal_r05_real001_clean.parquet"
OUT_PATH = ROOT / "data" / "processed" / "features_nepal_r05_real001_with_probs.parquet"


def prepare_features(df):
    y = df["label"].astype(int).values

    elev = df["elev_m"].values
    slope = df["slope_deg"].values
    aspect = df["aspect_deg"].values

    aspect_rad = np.deg2rad(aspect)
    aspect_sin = np.sin(aspect_rad)
    aspect_cos = np.cos(aspect_rad)

    X = np.column_stack([elev, slope, aspect_sin, aspect_cos])

    return X, y


def main():
    df = pd.read_parquet(DATA_PATH)
    print("Rows:", len(df))

    X, y = prepare_features(df)

    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y)

    probs = model.predict_proba(X)[:, 1]
    df["prob_landslide"] = probs

    df.to_parquet(OUT_PATH, index=False)

    print("\n=== Example True Positives (high confidence) ===")
    tp = df[(df["label"] == 1)].sort_values("prob_landslide", ascending=False).head(5)
    print(tp[["elev_m", "slope_deg", "aspect_deg", "prob_landslide"]])

    print("\n=== Example False Negatives (low confidence positives) ===")
    fn = df[(df["label"] == 1)].sort_values("prob_landslide", ascending=True).head(5)
    print(fn[["elev_m", "slope_deg", "aspect_deg", "prob_landslide"]])

    importances = model.feature_importances_
    print("\nFeature Importances:")
    print("Elevation:", importances[0])
    print("Slope:", importances[1])
    print("Aspect_sin:", importances[2])
    print("Aspect_cos:", importances[3])


if __name__ == "__main__":
    main()