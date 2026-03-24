"""
Fluvial Ablation Study: River Proximity-Only Model

GOAL: Isolates and evaluates the independent predictive power of river distance
features (linear and logarithmic) to quantify the 'river effect' on landslides.

RESPONSIBILITIES:
- Constructs a restricted feature matrix using only 'dist_river_m' and
  'log_dist_river' to test the model's reliance on fluvial proximity.
- Implements a Scikit-Learn Pipeline for Logistic Regression to automate
  feature scaling (StandardScaler), ensuring numerical stability and convergence.
- Executes 5-fold Stratified Cross-Validation with an optimized threshold
  search biased toward lower probability cutoffs (0.01 - 0.50).
- Benchmarks Random Forest against Scaled Logistic Regression to determine
  the most effective way to model the spatial decay of landslide risk.
- Provides an empirical baseline for understanding how much landslide
  variance is explained specifically by toe-undercutting and river-base erosion.

INPUT:  data/processed/features_nepal_rxx_realxxx_model_hydro_river.parquet
OUTPUT: Performance report (ROC-AUC, PR-AUC, F1) for the river-exclusive model.
"""

from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "features_nepal_r05_real001_model_hydro_river.parquet"


def best_threshold_f1(y_true, probs):
    thresholds = np.linspace(0.01, 0.50, 50)  # low thresholds often matter in imbalanced problems
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

        # Support both pipeline + plain estimators
        probs = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, probs)
        pr_auc = average_precision_score(y_test, probs)
        best_t, best_f1 = best_threshold_f1(y_test, probs)

        aucs.append(auc)
        pr_aucs.append(pr_auc)
        f1s.append(best_f1)
        ts.append(best_t)

        print(f"Fold {fold} | ROC-AUC: {auc:.4f} | PR-AUC: {pr_auc:.4f} | Best F1: {best_f1:.4f} @ t={best_t:.2f}")

    print("\nMean ROC-AUC :", round(float(np.mean(aucs)), 4))
    print("Mean PR-AUC  :", round(float(np.mean(pr_aucs)), 4))
    print("Mean F1      :", round(float(np.mean(f1s)), 4))
    print("Mean best t  :", round(float(np.mean(ts)), 3))


def main():
    df = pd.read_parquet(DATA_PATH)

    if "label" not in df.columns:
        raise RuntimeError("Missing label column.")
    y = df["label"].astype(int).values

    if "dist_river_m" not in df.columns or "log_dist_river" not in df.columns:
        raise RuntimeError("Missing dist_river_m / log_dist_river. Run script 26 first.")

    X = np.column_stack([
        df["dist_river_m"].values,
        df["log_dist_river"].values,
    ])
    print("Feature matrix shape:", X.shape, "(dist_river_m, log_dist_river)")

    rf = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=3,
        n_jobs=-1,
        random_state=42
    )
    evaluate_model(rf, X, y, "RandomForest (river-only)")

    rf_bal = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=3,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )
    evaluate_model(rf_bal, X, y, "RandomForest balanced (river-only)")

    # Logistic regression benefits from scaling; this also avoids convergence warnings
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, class_weight="balanced"))
    ])
    evaluate_model(lr, X, y, "Logistic Regression balanced (river-only, scaled)")


if __name__ == "__main__":
    main()