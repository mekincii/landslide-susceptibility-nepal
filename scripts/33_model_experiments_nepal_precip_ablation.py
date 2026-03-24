"""
Precipitation Scaling Ablation: Raw vs. Log-Transformed Rainfall

GOAL: Quantifies the impact of different precipitation scaling methods on
landslide susceptibility model performance in the high-gradient Nepal landscape.

RESPONSIBILITIES:
- Establishes a fixed 'Topographic+Hydrologic' base feature set to ensure
  experimental control across multiple model runs.
- Executes two parallel 5-fold Stratified Cross-Validation experiments:
    - Ablation A: Uses raw annual precipitation (CHELSA Bio12 in mm).
    - Ablation B: Uses log-transformed precipitation log.
- Utilizes high-capacity Random Forest ensembles (500 trees) to capture
  fine-grained interactions between rainfall and slope geometry.
- Optimizes the F1-threshold for each fold to assess if scaling affects
  the model's calibration and classification triggers.
- Reports Mean and Standard Deviation for ROC-AUC, PR-AUC, and F1 to
  identify the mathematically superior feature representation.

INPUT:  data/processed/features_nepal_rxx_realxxx_model_hydro_river_precip_clean.parquet
OUTPUT: Comparative performance metrics and stability audit for A/B testing.
"""

from __future__ import annotations
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

import numpy as np
import pandas as pd


IN_PATH = Path("data/processed/features_nepal_r05_real001_model_hydro_river_precip_clean.parquet")
LABEL_COL = "label"

PRECIP_RAW = "precip_bio12"
PRECIP_LOG = "log_precip_bio12"

# Base features (everything except precip; aspect handled dynamically)
BASE_FEATURES_CORE = [
    "elev_m",
    "slope_deg",
    "log_spi",
    "twi",
    "log_sca",
    "log_dist_river",
]

N_SPLITS = 5
RANDOM_STATE = 42

# Threshold grid for "best F1"
THRESHOLDS = np.linspace(0.01, 0.99, 99)


def _best_f1(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    best = (-1.0, 0.5)  # (f1, threshold)
    for t in THRESHOLDS:
        y_hat = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_hat, zero_division=0)
        if f1 > best[0]:
            best = (f1, float(t))
    return best  # (best_f1, best_threshold)


def run_cv(
    df: pd.DataFrame,
    features: list[str],
    title: str,
) -> dict:
    missing = [c for c in features + [LABEL_COL] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for run '{title}': {missing}")

    dfx = df.dropna(subset=features + [LABEL_COL]).copy()

    X = dfx[features].to_numpy(dtype=float)
    y = dfx[LABEL_COL].to_numpy(dtype=int)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    roc_list, pr_list, f1_list, t_list = [], [], [], []

    print(f"\n{title}")
    print(f"Rows used: {len(dfx)} | Features: {len(features)}")

    for fold, (tr, te) in enumerate(skf.split(X, y), start=1):
        model = RandomForestClassifier(
            n_estimators=500,
            random_state=RANDOM_STATE + fold,
            n_jobs=-1,
            class_weight=None,
        )
        model.fit(X[tr], y[tr])

        prob = model.predict_proba(X[te])[:, 1]
        roc = roc_auc_score(y[te], prob)
        pr = average_precision_score(y[te], prob)
        best_f1, best_t = _best_f1(y[te], prob)

        roc_list.append(roc)
        pr_list.append(pr)
        f1_list.append(best_f1)
        t_list.append(best_t)

        print(
            f"Fold {fold} | ROC-AUC: {roc:.4f} | PR-AUC: {pr:.4f} | Best F1: {best_f1:.4f} @ t={best_t:.2f}"
        )

    out = {
        "title": title,
        "features": features,
        "mean_roc": float(np.mean(roc_list)),
        "mean_pr": float(np.mean(pr_list)),
        "mean_f1": float(np.mean(f1_list)),
        "mean_t": float(np.mean(t_list)),
        "std_roc": float(np.std(roc_list)),
        "std_pr": float(np.std(pr_list)),
        "std_f1": float(np.std(f1_list)),
    }

    print(f"\nMean ROC-AUC : {out['mean_roc']:.4f} ± {out['std_roc']:.4f}")
    print(f"Mean PR-AUC  : {out['mean_pr']:.4f} ± {out['std_pr']:.4f}")
    print(f"Mean F1      : {out['mean_f1']:.4f} ± {out['std_f1']:.4f}")
    print(f"Mean best t  : {out['mean_t']:.3f}")

    return out


def main() -> None:
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Input not found: {IN_PATH.resolve()}")

    df = pd.read_parquet(IN_PATH)

    cols = set(df.columns)

    if {"aspect_sin", "aspect_cos"}.issubset(cols):
        aspect_feats = ["aspect_sin", "aspect_cos"]
        print("Using existing aspect_sin/aspect_cos")
    elif "aspect_deg" in cols:
        rad = np.deg2rad(df["aspect_deg"].astype(float).to_numpy())
        df["aspect_sin"] = np.sin(rad)
        df["aspect_cos"] = np.cos(rad)
        aspect_feats = ["aspect_sin", "aspect_cos"]
    else:
        aspect_feats = []
        print("WARNING: No aspect columns found. Running without aspect.")

    base = BASE_FEATURES_CORE + aspect_feats

    missing_base = [c for c in base if c not in df.columns]
    if missing_base:
        raise KeyError(
            f"Base feature(s) missing from dataframe: {missing_base}\n"
            f"Available cols include: {sorted(list(cols))[:40]} ..."
        )

    # Run ablations
    _ = run_cv(
        df,
        features=base + [PRECIP_RAW],
        title="Ablation A — base + precip_bio12 (raw)",
    )

    _ = run_cv(
        df,
        features=base + [PRECIP_LOG],
        title="Ablation B — base + log_precip_bio12",
    )


if __name__ == "__main__":
    main()