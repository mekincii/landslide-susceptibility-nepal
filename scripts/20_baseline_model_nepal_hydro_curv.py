"""
Advanced Susceptibility Model: Hydro-Morphometric Random Forest

GOAL: Evaluates the combined predictive power of terrain geometry,
hydrological indices, and slope curvature for landslide susceptibility.

RESPONSIBILITIES:
- Constructs a comprehensive 9-dimensional feature matrix (Elevation, Slope,
  Circular Aspect, TWI, Log-SPI, Log-SCA, Plan Curvature, and Profile Curvature).
- Performs 5-fold Stratified Cross-Validation to benchmark the model against
  previous iterations (Baseline and Hydro-only).
- Calculates and aggregates 'Mean Feature Importance' across all CV folds to
  rank the physical drivers of landslide occurrence.
- Assesses whether the addition of morphometric shape features (curvature)
  provides statistically significant gains in ROC-AUC and PR-AUC.
- Maintains strict experimental control (Random Seed: 42) for valid comparison
  across the 51-script project lifecycle.

INPUT:  data/processed/features_nepal_rxx_realxxx_model_hydro_curv_clean.parquet
OUTPUT: Console report of Mean Performance Metrics and Feature Importance Rankings.
"""

from __future__ import annotations
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.ensemble import RandomForestClassifier

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "features_nepal_r05_real001_model_hydro_curv_clean.parquet"


def prepare_features(df: pd.DataFrame):
    y = df["label"].astype(int).values

    elev = df["elev_m"].values
    slope = df["slope_deg"].values
    aspect = df["aspect_deg"].values
    aspect_rad = np.deg2rad(aspect)
    aspect_sin = np.sin(aspect_rad)
    aspect_cos = np.cos(aspect_rad)

    twi = df["twi"].values
    log_spi = df["log_spi"].values
    log_sca = df["log_sca"].values

    curv_plan = df["curv_plan"].values
    curv_profile = df["curv_profile"].values

    X = np.column_stack([
        elev,
        slope,
        aspect_sin,
        aspect_cos,
        twi,
        log_spi,
        log_sca,
        curv_plan,
        curv_profile
    ])

    return X, y


def main():
    df = pd.read_parquet(DATA_PATH)

    X, y = prepare_features(df)
    print("Feature matrix shape:", X.shape)
    print("Positive ratio:", y.mean())

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    aucs, pr_aucs, f1s = [], [], []
    all_importances = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = RandomForestClassifier(
            n_estimators=300,
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
        all_importances.append(model.feature_importances_)

        print(f"Fold {fold} | ROC-AUC: {auc:.4f} | PR-AUC: {pr_auc:.4f} | F1: {f1:.4f}")

    print("\n")
    print(f"Mean ROC-AUC : {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"Mean PR-AUC  : {np.mean(pr_aucs):.4f} ± {np.std(pr_aucs):.4f}")
    print(f"Mean F1      : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

    mean_importance = np.mean(np.array(all_importances), axis=0)

    feature_names = [
        "elev", "slope", "aspect_sin", "aspect_cos",
        "twi", "log_spi", "log_sca",
        "curv_plan", "curv_profile"
    ]

    print("\nMean Feature Importances")
    for name, imp in sorted(zip(feature_names, mean_importance),
                            key=lambda x: -x[1]):
        print(f"{name:15s} {imp:.4f}")


if __name__ == "__main__":
    main()