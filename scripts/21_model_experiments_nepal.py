"""
Model Architecture Comparison & Threshold Optimization

GOAL: Evaluates different machine learning algorithms and class-balancing
strategies to determine the most robust architecture for landslide prediction.

RESPONSIBILITIES:
- Compares non-linear Random Forest ensembles against linear Logistic
  Regression baselines.
- Tests the impact of internal 'class_weight="balanced"' adjustments to
  handle the 5:1 imbalanced nature of the training data.
- Implements a dynamic threshold-tuning search to maximize the F1-score,
  moving beyond the standard 0.5 probability cutoff.
- Executes 5-fold Stratified Cross-Validation to ensure performance metrics
  (ROC-AUC, PR-AUC, and Optimized F1) are statistically stable.
- Provides a decision-support framework for selecting the final model
  architecture based on predictive gain vs. model complexity.

INPUT:  data/processed/features_nepal_rxx_realxxx_clean.parquet
OUTPUT: Console comparison of algorithm performance and optimal thresholds.
"""

from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "features_nepal_r05_real001_clean.parquet"


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


def best_threshold_f1(y_true, probs):
    thresholds = np.linspace(0.05, 0.95, 50)
    best_f1 = 0
    best_t = 0.5
    for t in thresholds:
        preds = (probs > t).astype(int)
        f1 = f1_score(y_true, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t, best_f1


def evaluate_model(model, X, y, name):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    aucs, pr_aucs, f1s = [], [], []

    print(f"\n=== {name} ===")

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, probs)
        pr_auc = average_precision_score(y_test, probs)

        # threshold tuning inside fold
        best_t, best_f1 = best_threshold_f1(y_test, probs)

        aucs.append(auc)
        pr_aucs.append(pr_auc)
        f1s.append(best_f1)

        print(f"Fold {fold} | ROC-AUC: {auc:.4f} | PR-AUC: {pr_auc:.4f} | Best F1: {best_f1:.4f} @ t={best_t:.2f}")

    print("\nMean ROC-AUC :", round(np.mean(aucs), 4))
    print("Mean PR-AUC  :", round(np.mean(pr_aucs), 4))
    print("Mean F1      :", round(np.mean(f1s), 4))


def main():
    df = pd.read_parquet(DATA_PATH)

    X, y = prepare_features(df)

    # Random Forest default
    rf_default = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=3,
        n_jobs=-1,
        random_state=42,
    )
    evaluate_model(rf_default, X, y, "RandomForest (default)")

    # Random Forest balanced
    rf_balanced = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=3,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    evaluate_model(rf_balanced, X, y, "RandomForest (balanced)")

    # Logistic Regression
    logreg = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )
    evaluate_model(logreg, X, y, "Logistic Regression (balanced)")


if __name__ == "__main__":
    main()