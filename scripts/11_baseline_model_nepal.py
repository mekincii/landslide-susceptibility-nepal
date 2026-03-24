"""
Baseline Landslide Susceptibility Model: Topographic Random Forest

GOAL: Trains and evaluates a baseline Random Forest classifier using only
terrain-derived features (Elevation, Slope, Aspect) to establish a performance
benchmark for the Nepal study area.

RESPONSIBILITIES:
- Implements circular feature encoding for 'Aspect' by transforming degrees into
  Sine and Cosine components to preserve geographic continuity.
- Utilizes a 5-fold Stratified Cross-Validation strategy to ensure robust
  performance metrics and handle class distribution consistency.
- Trains a Random Forest ensemble (300 estimators) to capture non-linear
  interactions between topographic variables.
- Reports a comprehensive suite of performance metrics: ROC-AUC, Precision-Recall
  AUC (PR-AUC), and F1-score.
- Establishes the 'Topography-only' baseline to compare against future iterations
  incorporating climate and hydrological data.

INPUT:  data/processed/features_nepal_rxx_realxxx_clean.parquet
OUTPUT: Console summary of cross-validation performance and stability metrics.
"""

from __future__ import annotations
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.ensemble import RandomForestClassifier

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "features_nepal_r05_real001_clean.parquet"


def prepare_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    label_col_candidates = ["label", "y", "is_landslide", "target", "class"]
    label_col = None
    for c in label_col_candidates:
        if c in df.columns:
            label_col = c
            break

    if label_col is None:
        raise RuntimeError("No label column found in dataset.")

    y = df[label_col].astype(int).values

    # Raw terrain features
    elev = df["elev_m"].values
    slope = df["slope_deg"].values
    aspect = df["aspect_deg"].values

    # Aspect is circular (0° == 360°)
    aspect_rad = np.deg2rad(aspect)
    aspect_sin = np.sin(aspect_rad)
    aspect_cos = np.cos(aspect_rad)

    X = np.column_stack([elev, slope, aspect_sin, aspect_cos])

    return X, y


def main():

    df = pd.read_parquet(DATA_PATH)

    X, y = prepare_features(df)

    print("Feature matrix shape:", X.shape)
    print("Positive ratio:", y.mean())

    # 5-fold stratified CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    aucs = []
    pr_aucs = []
    f1s = []

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

    print(f"Mean ROC-AUC : {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"Mean PR-AUC  : {np.mean(pr_aucs):.4f} ± {np.std(pr_aucs):.4f}")
    print(f"Mean F1      : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")


if __name__ == "__main__":
    main()