"""
Combined Hydro-River Model Experiments & Feature Ranking

GOAL: Evaluates the predictive synergy between catchment hydrology and river
proximity to determine their combined impact on landslide susceptibility.

RESPONSIBILITIES:
- Constructs a high-dimensional feature matrix integrating Topography,
  Hydrology (TWI/SPI/SCA), and Fluvial Proximity (Log-Distance to River).
- Benchmarks Random Forest (Default/Balanced) against Logistic Regression
  using a 5-fold Stratified Cross-Validation framework.
- Executes an internal threshold search to optimize the F1-score for each
  architectural configuration.
- Calculates and sorts Mean Feature Importances to quantify the relative
  influence of river proximity compared to traditional terrain drivers.
- Validates data lineage by ensuring all 8 required environmental and
  geomorphic predictors are present before model training.

INPUT:  data/processed/features_nepal_rxx_realxxx_model_hydro_river.parquet
OUTPUT: Console report of model performance (AUC/F1) and prioritized
        feature importance rankings.
"""

from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "features_nepal_r05_real001_model_hydro_river.parquet"


def best_threshold_f1(y_true, probs):
    thresholds = np.linspace(0.05, 0.95, 50)
    best_f1 = -1.0
    best_t = 0.5
    for t in thresholds:
        preds = (probs > t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t, best_f1


def prepare_features(df: pd.DataFrame):
    required = [
        "label",
        "elev_m",
        "slope_deg",
        "aspect_deg",
        "twi",
        "log_spi",
        "log_sca",
        "log_dist_river",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    y = df["label"].astype(int).values

    # Aspect sin/cos
    aspect_rad = np.deg2rad(df["aspect_deg"].values)
    aspect_sin = np.sin(aspect_rad)
    aspect_cos = np.cos(aspect_rad)

    X = np.column_stack([
        df["elev_m"].values,
        df["slope_deg"].values,
        aspect_sin,
        aspect_cos,
        df["twi"].values,
        df["log_spi"].values,
        df["log_sca"].values,
        df["log_dist_river"].values,
    ])

    feature_names = [
        "elev",
        "slope",
        "aspect_sin",
        "aspect_cos",
        "twi",
        "log_spi",
        "log_sca",
        "log_dist_river",
    ]
    return X, y, feature_names


def evaluate_model(model, X, y, name):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    aucs, pr_aucs, f1s, ts = [], [], [], []
    importances = []

    print(f"\n=== {name} ===")
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, probs)
        pr_auc = average_precision_score(y_test, probs)

        best_t, best_f1 = best_threshold_f1(y_test, probs)

        aucs.append(auc)
        pr_aucs.append(pr_auc)
        f1s.append(best_f1)
        ts.append(best_t)

        if hasattr(model, "feature_importances_"):
            importances.append(model.feature_importances_)

        print(f"Fold {fold} | ROC-AUC: {auc:.4f} | PR-AUC: {pr_auc:.4f} | Best F1: {best_f1:.4f} @ t={best_t:.2f}")

    print("\nMean ROC-AUC :", round(float(np.mean(aucs)), 4))
    print("Mean PR-AUC  :", round(float(np.mean(pr_aucs)), 4))
    print("Mean F1      :", round(float(np.mean(f1s)), 4))
    print("Mean best t  :", round(float(np.mean(ts)), 3))

    mean_imp = None
    if importances:
        mean_imp = np.mean(np.vstack(importances), axis=0)
    return mean_imp


def main():
    df = pd.read_parquet(DATA_PATH)

    X, y, feature_names = prepare_features(df)

    rf_default = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=3,
        n_jobs=-1,
        random_state=42,
    )
    imp = evaluate_model(rf_default, X, y, "RandomForest (default)")

    if imp is not None:
        print("\nMean Feature Importances (RF default)")
        order = np.argsort(-imp)
        for idx in order:
            print(f"{feature_names[idx]:>14s}  {imp[idx]:.4f}")

    rf_balanced = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=3,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    evaluate_model(rf_balanced, X, y, "RandomForest (balanced)")

    logreg = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )
    evaluate_model(logreg, X, y, "Logistic Regression (balanced)")


if __name__ == "__main__":
    main()