"""
Hydro-Topographic Model Evaluation & Performance Benchmarking

GOAL: Trains and validates an expanded Random Forest model incorporating both
terrain geometry and hydrological indices to assess predictive improvement.

RESPONSIBILITIES:
- Constructs an augmented feature matrix including Elevation, Slope,
  Aspect (Sin/Cos), TWI, and Log-transformed SPI/SCA.
- Executes 5-fold Stratified Cross-Validation to evaluate the model's ability
  to generalize across the Nepal study area with the new feature set.
- Compares model performance (ROC-AUC, PR-AUC, F1) against the previous
  topography-only baseline.
- Tracks feature importance across folds to assess the stability and
  contribution of hydrological variables in landslide prediction.
- Maintains strict hyperparameter parity with the baseline model to ensure
  a statistically valid comparative experiment.

INPUT:  data/processed/features_nepal_rxx_realxxx_model_hydro.parquet
OUTPUT: Console report of Hydro-Topographic performance metrics and stability.
"""

from __future__ import annotations
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.ensemble import RandomForestClassifier

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "features_nepal_r05_real001_model_hydro.parquet"


def prepare_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if "label" not in df.columns:
        raise RuntimeError("Expected 'label' column in dataset.")
    y = df["label"].astype(int).values

    # terrain
    elev = df["elev_m"].values
    slope = df["slope_deg"].values
    aspect = df["aspect_deg"].values

    aspect_rad = np.deg2rad(aspect)
    aspect_sin = np.sin(aspect_rad)
    aspect_cos = np.cos(aspect_rad)

    # hydrology
    twi = df["twi"].values
    log_spi = df["log_spi"].values
    log_sca = df["log_sca"].values

    X = np.column_stack([elev, slope, aspect_sin, aspect_cos, twi, log_spi, log_sca])
    return X, y


def main():
    df = pd.read_parquet(DATA_PATH)

    X, y = prepare_features(df)
    print("Feature matrix shape:", X.shape)
    print("Positive ratio:", y.mean())

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    aucs, pr_aucs, f1s = [], [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=42,
        )

        model.fit(X_train, y_train)

        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs > 0.5).astype(int)

        auc = roc_auc_score(y_test, probs)
        pr_auc = average_precision_score(y_test, probs)
        f1 = f1_score(y_test, preds)

        aucs.append(auc)
        pr_aucs.append(pr_auc)
        f1s.append(f1)

        print(f"Fold {fold} | ROC-AUC: {auc:.4f} | PR-AUC: {pr_auc:.4f} | F1: {f1:.4f}")

        importances = model.feature_importances_

        print("importances:", importances)

    print("\n")
    print(f"Mean ROC-AUC : {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"Mean PR-AUC  : {np.mean(pr_aucs):.4f} ± {np.std(pr_aucs):.4f}")
    print(f"Mean F1      : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")


if __name__ == "__main__":
    main()