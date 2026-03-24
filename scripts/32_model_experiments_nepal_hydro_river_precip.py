"""
Multi-Hazard Integrated Model: The Full Feature Suite

GOAL: Trains and compares the final ensemble of models (Random Forest and
Logistic Regression) using the complete array of 10+ topographic,
hydrologic, fluvial, and climatological predictors.

RESPONSIBILITIES:
- Constructs the 'Master Feature Matrix' including Elevation, Slope,
  Circular Aspect, TWI, SPI, SCA, River Proximity, and Annual Precipitation.
- Executes 5-fold Stratified Cross-Validation to determine the maximum
  predictive ceiling of the project.
- Implements on-the-fly Feature Scaling (StandardScaler) for linear models
  to ensure numerical stability across variable magnitudes.
- Conducts automated 'Threshold Optimization' (200-step search) to maximize
  the F1-score, providing a calibrated decision-point for susceptibility mapping.
- Ranks global feature importance to identify which environmental drivers
  (e.g., Rainfall vs. River Proximity) dominate the Nepal landslide regime.
- Provides a direct comparative benchmark between Default RF, Balanced RF,
  and Balanced/Scaled Logistic Regression.

INPUT:  data/processed/features_nepal_rxx_realxxx_model_hydro_river_precip_clean.parquet
OUTPUT: Final performance leaderboard and aggregated feature importance rankings.
"""

from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd


DATA_PATH = Path("data/processed/features_nepal_r05_real001_model_hydro_river_precip_clean.parquet")
LABEL_COL = "label"


def best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray, n: int = 200):
    thresholds = np.linspace(0.01, 0.99, n)
    best_f1 = -1.0
    best_t = 0.5
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_f1, best_t


def run_model(name, model_factory, X, y, feature_names, do_scale=False):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    roc_list, pr_list, f1_list, t_list = [], [], [], []
    importances = []

    for fold, (tr, te) in enumerate(skf.split(X, y), start=1):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]

        if do_scale:
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(Xtr)
            Xte = scaler.transform(Xte)

        model = model_factory()
        model.fit(Xtr, ytr)

        yprob = model.predict_proba(Xte)[:, 1]
        roc = roc_auc_score(yte, yprob)
        pr = average_precision_score(yte, yprob)
        bf1, bt = best_f1_threshold(yte, yprob)

        roc_list.append(roc)
        pr_list.append(pr)
        f1_list.append(bf1)
        t_list.append(bt)

        print(
            f"Fold {fold} | ROC-AUC: {roc:.4f} | PR-AUC: {pr:.4f} | Best F1: {bf1:.4f} @ t={bt:.2f}"
        )

        if hasattr(model, "feature_importances_"):
            importances.append(model.feature_importances_)

    print(f"\nMean ROC-AUC : {np.mean(roc_list):.4f}")
    print(f"Mean PR-AUC  : {np.mean(pr_list):.4f}")
    print(f"Mean F1      : {np.mean(f1_list):.4f}")
    print(f"Mean best t  : {np.mean(t_list):.3f}")

    if importances:
        imp = np.mean(np.vstack(importances), axis=0)
        imp_series = pd.Series(imp, index=feature_names).sort_values(ascending=False)
        print("\n=== Mean Feature Importances (RF) ===")
        print(imp_series.to_string())

    return {
        "roc": roc_list,
        "pr": pr_list,
        "f1": f1_list,
        "t": t_list,
    }


def main():
    df = pd.read_parquet(DATA_PATH)

    if {"aspect_sin", "aspect_cos"}.issubset(df.columns):
        aspect_feats = ["aspect_sin", "aspect_cos"]
    elif "aspect_deg" in df.columns:
        ang = np.deg2rad(df["aspect_deg"].astype("float64"))
        df["aspect_sin"] = np.sin(ang)
        df["aspect_cos"] = np.cos(ang)
        aspect_feats = ["aspect_sin", "aspect_cos"]
    else:
        raise KeyError("Expected aspect_deg or (aspect_sin + aspect_cos).")

    FEATURES = [
        "elev_m",
        "slope_deg",
        *aspect_feats,
        "twi",
        "log_spi",
        "log_sca",
        "log_dist_river",
        "precip_bio12",
        "log_precip_bio12",
    ]

    missing_cols = [c for c in FEATURES + [LABEL_COL] if c not in df.columns]
    if missing_cols:
        raise KeyError(missing_cols)

    before = len(df)
    df = df.dropna(subset=FEATURES + [LABEL_COL]).copy()
    dropped = before - len(df)
    if dropped:
        print(f"Dropped rows with NaNs in FEATURES/label: {dropped} ({(dropped/before)*100:.2f}%)")
    print(f"Final rows used: {len(df)}")

    y = df[LABEL_COL].astype(int).to_numpy()
    X = df[FEATURES].astype("float64").to_numpy()

    print(f"Positive ratio: {y.mean():.4f}")
    print(f"Feature matrix shape: {X.shape}")

    print("\nRandomForest (default)")
    run_model(
        "rf_default",
        lambda: RandomForestClassifier(
            n_estimators=400,
            random_state=42,
            n_jobs=-1,
        ),
        X, y, FEATURES
    )

    print("\nRandomForest (balanced)")
    run_model(
        "rf_balanced",
        lambda: RandomForestClassifier(
            n_estimators=400,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        ),
        X, y, FEATURES
    )

    print("\nLogistic Regression (balanced, scaled)")
    run_model(
        "lr_balanced",
        lambda: LogisticRegression(
            max_iter=4000,
            class_weight="balanced",
            solver="lbfgs",
        ),
        X, y, FEATURES, do_scale=True
    )


if __name__ == "__main__":
    main()