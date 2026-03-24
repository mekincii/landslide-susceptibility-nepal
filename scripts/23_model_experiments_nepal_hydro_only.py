"""
Ablation Study: Hydrology-Only Model Experiment

GOAL: Isolates and evaluates the independent predictive power of
hydrological indices (TWI, SPI, SCA) by removing all primary topographic
features from the model training process.

RESPONSIBILITIES:
- Constructs a restricted feature matrix containing only water-flow
  proxies: Topographic Wetness Index, Log-SPI, and Log-SCA.
- Benchmarks Random Forest and Logistic Regression architectures to
  determine if hydrological predictors exhibit linear or non-linear
  relationships with landslide occurrence.
- Executes 5-fold Stratified Cross-Validation with dynamic threshold
  optimization to find the most effective 'wetness-based' classification cutoff.
- Quantifies the 'information loss' when primary terrain geometry (Slope,
  Elevation) is absent, providing a baseline for feature-importance studies.

INPUT:  data/processed/features_nepal_rxx_realxxx_model_hydro.parquet
OUTPUT: Performance report (ROC-AUC, PR-AUC, F1) for the hydrology-exclusive model.
"""

from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "features_nepal_r05_real001_model_hydro.parquet"


def prepare_features(df: pd.DataFrame):
    y = df["label"].astype(int).values

    # Hydro-only
    X = np.column_stack([
        df["twi"].values,
        df["log_spi"].values,
        df["log_sca"].values,
    ])
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