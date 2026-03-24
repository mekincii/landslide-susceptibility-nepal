"""
Hydro-Topographic Model Benchmarking & Threshold Stability Analysis

GOAL: Evaluates architectural performance (Random Forest vs. Logistic
Regression) specifically within the context of the hydro-augmented dataset.

RESPONSIBILITIES:
- Benchmarks non-linear ensemble methods against linear baselines using
  the 7-dimensional hydro-topographic feature set.
- Analyzes the impact of 'balanced' class weighting on predictive performance
  given the inherent landslide/non-landslide imbalance.
- Executes an internal 50-step threshold search within each CV fold to
  optimize the F1-score and determine the most stable classification cutoff.
- Tracks and reports 'Mean best t' (threshold) to evaluate model calibration
  and decision-point reliability across spatial folds.
- Provides the empirical basis for selecting the optimal model/weight/threshold
  configuration for the final hydrological susceptibility map.

INPUT:  data/processed/features_nepal_rxx_realxxx_model_hydro.parquet
OUTPUT: Comparative performance metrics (ROC-AUC, PR-AUC, F1) and optimal
        threshold stability report.
"""

from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

# Hydro + terrain dataset
DATA_PATH = ROOT / "data" / "processed" / "features_nepal_r05_real001_model_hydro.parquet"


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

    X = np.column_stack([elev, slope, aspect_sin, aspect_cos, twi, log_spi, log_sca])
    return X, y


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


def evaluate_model(model, X, y, name):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    aucs, pr_aucs, f1s, ts = [], [], [], []

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

        print(f"Fold {fold} | ROC-AUC: {auc:.4f} | PR-AUC: {pr_auc:.4f} | Best F1: {best_f1:.4f} @ t={best_t:.2f}")

    print("\nMean ROC-AUC :", round(np.mean(aucs), 4))
    print("Mean PR-AUC  :", round(np.mean(pr_aucs), 4))
    print("Mean F1      :", round(np.mean(f1s), 4))
    print("Mean best t  :", round(float(np.mean(ts)), 3))


def main():
    df = pd.read_parquet(DATA_PATH)

    X, y = prepare_features(df)

    rf_default = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=3,
        n_jobs=-1,
        random_state=42,
    )
    evaluate_model(rf_default, X, y, "RandomForest (default)")

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