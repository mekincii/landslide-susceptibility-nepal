"""
Knowledge-Driven Baseline: Weighted Fuzzy Membership Model

GOAL: Establishes a non-machine-learning benchmark based on expert-weighted
environmental thresholds to quantify the 'value-add' of complex ML ensembles.

RESPONSIBILITIES:
- Implements linear fuzzy membership functions (0.0 to 1.0) based on
  statistical quantiles (10th/90th) for key environmental predictors.
- Applies 'Risk Orientation' logic to handle inverse relationships (e.g.,
  shorter distances to rivers increasing landslide susceptibility).
- Combines fuzzified features using a Weighted Linear Combination (WLC)
  based on previously observed feature importance rankings.
- Evaluates the fuzzy model within the standard Spatial GroupKFold framework
  to provide a direct performance comparison (AUC/F1) against the Random Forest.
- Logs the baseline results to the master experiment tracker for final
  project synthesis and scientific validation.

INPUT:  data/processed/features_nepal_rxx_realxxx_model_hydro_river_precip_clean.parquet
OUTPUT: outputs/experiments/fuzzy_baseline_spatial_cv_reduced.csv
"""

from __future__ import annotations
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.model_selection import GroupKFold

import argparse
import json
import numpy as np
import pandas as pd


DATA_DEFAULT = r"data\processed\features_nepal_r05_real001_model_hydro_river_precip_clean.parquet"
LABEL_COL = "label"
GEOM_COL = "geometry"


def ensure_aspect_sin_cos(df: pd.DataFrame) -> pd.DataFrame:
    if "aspect_sin" in df.columns and "aspect_cos" in df.columns:
        return df

    if "aspect_deg" not in df.columns:
        raise KeyError("Need either aspect_sin/aspect_cos or aspect_deg")

    rad = np.deg2rad(df["aspect_deg"].astype(float).to_numpy())
    df = df.copy()
    df["aspect_sin"] = np.sin(rad)
    df["aspect_cos"] = np.cos(rad)

    return df


def extract_xy_from_geometry(df: pd.DataFrame):
    try:
        from shapely import wkb
    except Exception as e:
        raise ImportError("Install shapely (pip install shapely) to decode WKB geometry.") from e

    geom = df[GEOM_COL].values
    pts = [wkb.loads(g) if isinstance(g, (bytes, bytearray)) else g for g in geom]
    x = np.array([p.x for p in pts], dtype=float)
    y = np.array([p.y for p in pts], dtype=float)
    return x, y


def grid_groups(x, y, grid_km: float):
    g = grid_km * 1000.0
    gx = np.floor(x / g).astype(int)
    gy = np.floor(y / g).astype(int)
    return np.char.add(gx.astype(str), np.char.add("_", gy.astype(str)))


def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray):
    thresholds = np.linspace(0.01, 0.99, 99)
    best_f1 = -1.0
    best_t = 0.5
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return float(best_f1), float(best_t)


def fuzzy_membership_quantile(series: pd.Series, high_risk_when_high: bool = True):
    x = series.astype(float).to_numpy()
    q10 = np.nanquantile(x, 0.10)
    q90 = np.nanquantile(x, 0.90)

    if np.isclose(q10, q90):
        mu = np.full_like(x, 0.5, dtype=float)
    else:
        mu = (x - q10) / (q90 - q10)
        mu = np.clip(mu, 0.0, 1.0)

    if not high_risk_when_high:
        mu = 1.0 - mu

    return mu


def build_fuzzy_score(df: pd.DataFrame, feature_weights: dict[str, float]) -> np.ndarray:
    memberships = {}

    # Orientation:
    # True  -> higher value = higher risk
    # False -> lower value = higher risk
    orientations = {
        "elev_m": True,
        "slope_deg": True,
        "aspect_sin": True,
        "aspect_cos": True,
        "log_spi": True,
        "twi": True,
        "log_sca": True,
        "log_dist_river": False,
        "log_precip_bio12": True,
    }

    for feat, weight in feature_weights.items():
        if feat not in df.columns:
            raise KeyError(f"Missing feature for fuzzy model: {feat}")
        memberships[feat] = fuzzy_membership_quantile(
            df[feat],
            high_risk_when_high=orientations[feat]
        )

    num = np.zeros(len(df), dtype=float)
    den = 0.0

    for feat, weight in feature_weights.items():
        num += weight * memberships[feat]
        den += weight

    score = num / den
    return score


def append_row(csv_path: Path, row: dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_row = pd.DataFrame([row])

    if csv_path.exists():
        df_row.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(csv_path, mode="w", header=True, index=False)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", default=DATA_DEFAULT)
    parser.add_argument("--grid-km", type=float, default=20.0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--out", default=r"outputs\experiments\fuzzy_baseline_spatial_cv_reduced.csv")
    parser.add_argument("--log-experiments", default=r"outputs\experiments\experiments.csv")

    args = parser.parse_args()

    df = pd.read_parquet(args.data)
    df = ensure_aspect_sin_cos(df)

    features = [
        "elev_m",
        "log_precip_bio12",
        "log_spi",
        "twi",
        "slope_deg",
    ]

    required = features + [LABEL_COL, GEOM_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df = df.dropna(subset=required).copy()

    feature_weights = {
        "elev_m": 1.00,
        "log_precip_bio12": 0.68,
        "log_spi": 0.12,
        "twi": 0.11,
        "slope_deg": 0.08,
    }

    x, y = extract_xy_from_geometry(df)
    groups = grid_groups(x, y, args.grid_km)

    y_true = df[LABEL_COL].to_numpy(int)

    gkf = GroupKFold(n_splits=args.folds)

    rows = []
    roc_list, pr_list, f1_list, t_list = [], [], [], []

    for fold, (_, te) in enumerate(gkf.split(df, y_true, groups), start=1):
        df_te = df.iloc[te].copy()
        y_te = y_true[te]

        score = build_fuzzy_score(df_te, feature_weights)

        roc = roc_auc_score(y_te, score)
        pr = average_precision_score(y_te, score)
        best_f1, best_t = best_f1_threshold(y_te, score)

        roc_list.append(roc)
        pr_list.append(pr)
        f1_list.append(best_f1)
        t_list.append(best_t)

        rows.append({
            "fold": fold,
            "grid_km": args.grid_km,
            "roc_auc": roc,
            "pr_auc": pr,
            "best_f1": best_f1,
            "best_t": best_t,
        })

        print(f"Fold {fold} | ROC-AUC: {roc:.4f} | PR-AUC: {pr:.4f} | Best F1: {best_f1:.4f} @ t={best_t:.2f}")

    mean_roc = float(np.mean(roc_list))
    std_roc = float(np.std(roc_list))
    mean_pr = float(np.mean(pr_list))
    std_pr = float(np.std(pr_list))
    mean_f1 = float(np.mean(f1_list))
    std_f1 = float(np.std(f1_list))
    mean_t = float(np.mean(t_list))

    print("\nFuzzy Baseline Spatial CV Results")
    print(f"Mean ROC-AUC : {mean_roc:.4f} ± {std_roc:.4f}")
    print(f"Mean PR-AUC  : {mean_pr:.4f} ± {std_pr:.4f}")
    print(f"Mean F1      : {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"Mean best t  : {mean_t:.3f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)

    log_row = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "run_id": pd.Timestamp.now().strftime("%Y%m%d_%H%M%S"),
        "dataset": "terrain_hydro_river_precip_log_fuzzy_reduced",
        "input_path": str(Path(args.data)),
        "rows": int(len(df)),
        "rows_used": int(len(df)),
        "pos_ratio": float(df[LABEL_COL].mean()),
        "pos_ratio_used": float(df[LABEL_COL].mean()),
        "label_col": LABEL_COL,
        "features": json.dumps(features),
        "features_signature": "|".join(features),
        "grid_km": float(args.grid_km),
        "folds": int(args.folds),
        "seed": "",
        "model": "Fuzzy_baseline_quantile_weighted_reduced",
        "roc_auc_mean": mean_roc,
        "roc_auc_std": std_roc,
        "pr_auc_mean": mean_pr,
        "pr_auc_std": std_pr,
        "f1_mean": mean_f1,
        "f1_std": std_f1,
        "best_t_mean": mean_t,
        "best_t_std": float(np.std(t_list)),
        "brier_mean": "",
        "brier_std": "",
    }

    append_row(Path(args.log_experiments), log_row)

    print("Saved fold results:", out_path)
    print("Logged fuzzy baseline to:", args.log_experiments)


if __name__ == "__main__":
    main()