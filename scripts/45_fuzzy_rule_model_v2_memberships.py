"""
Granular Expert Baseline: 5-Membership Fuzzy Inference System

GOAL: Refines the physical baseline model by increasing the resolution of
linguistic variables, allowing for better separation of extreme landslide triggers.

RESPONSIBILITIES:
- Fuzzifies environmental features into five distinct membership sets
  (Very Low to Very High) using an adaptive 5-quantile distribution ($P_{10}$ to $P_{90}$).
- Implements an expanded 12-rule logic base that prioritizes synergistic
  extremes (e.g., Very High Slope + Very High Precipitation).
- Defuzzifies the qualitative inference into a continuous susceptibility
  index (0.0 - 1.0) using class-center aggregation.
- Benchmarks this high-resolution 'Expert Logic' against the Random Forest
  within the standardized 20km Spatial CV framework.
- Quantifies whether increased granularity in human-coded rules can
  narrow the performance gap between fuzzy logic and machine learning.

INPUT:  data/processed/features_nepal_rxx_realxxx_model_hydro_river_precip_clean.parquet
OUTPUT: outputs/experiments/fuzzy_rule_spatial_cv_v2_memberships.csv
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



# 5-membership fuzzification
def quantile_params_5(series: pd.Series):
    x = series.astype(float).to_numpy()
    q10 = np.nanquantile(x, 0.10)
    q30 = np.nanquantile(x, 0.30)
    q50 = np.nanquantile(x, 0.50)
    q70 = np.nanquantile(x, 0.70)
    q90 = np.nanquantile(x, 0.90)
    return q10, q30, q50, q70, q90


def very_low_membership(x, q10, q30):
    x = np.asarray(x, dtype=float)
    mu = np.zeros_like(x)
    mu[x <= q10] = 1.0
    mask = (x > q10) & (x < q30)
    mu[mask] = (q30 - x[mask]) / (q30 - q10 + 1e-9)
    return np.clip(mu, 0, 1)


def low_membership(x, q10, q30, q50):
    x = np.asarray(x, dtype=float)
    mu = np.zeros_like(x)
    left = (x > q10) & (x < q30)
    right = (x >= q30) & (x < q50)
    mu[x == q30] = 1.0
    mu[left] = (x[left] - q10) / (q30 - q10 + 1e-9)
    mu[right] = (q50 - x[right]) / (q50 - q30 + 1e-9)
    return np.clip(mu, 0, 1)


def medium_membership(x, q30, q50, q70):
    x = np.asarray(x, dtype=float)
    mu = np.zeros_like(x)
    left = (x > q30) & (x < q50)
    right = (x >= q50) & (x < q70)
    mu[x == q50] = 1.0
    mu[left] = (x[left] - q30) / (q50 - q30 + 1e-9)
    mu[right] = (q70 - x[right]) / (q70 - q50 + 1e-9)
    return np.clip(mu, 0, 1)


def high_membership(x, q50, q70, q90):
    x = np.asarray(x, dtype=float)
    mu = np.zeros_like(x)
    left = (x > q50) & (x < q70)
    right = (x >= q70) & (x < q90)
    mu[x == q70] = 1.0
    mu[left] = (x[left] - q50) / (q70 - q50 + 1e-9)
    mu[right] = (q90 - x[right]) / (q90 - q70 + 1e-9)
    return np.clip(mu, 0, 1)


def very_high_membership(x, q70, q90):
    x = np.asarray(x, dtype=float)
    mu = np.zeros_like(x)
    mu[x >= q90] = 1.0
    mask = (x > q70) & (x < q90)
    mu[mask] = (x[mask] - q70) / (q90 - q70 + 1e-9)
    return np.clip(mu, 0, 1)


def fuzzify_feature_5(series: pd.Series):
    q10, q30, q50, q70, q90 = quantile_params_5(series)
    x = series.astype(float).to_numpy()

    return {
        "very_low": very_low_membership(x, q10, q30),
        "low": low_membership(x, q10, q30, q50),
        "med": medium_membership(x, q30, q50, q70),
        "high": high_membership(x, q50, q70, q90),
        "very_high": very_high_membership(x, q70, q90),
    }


# Rule-based fuzzy score
def build_fuzzy_rule_score_v2(df: pd.DataFrame) -> np.ndarray:
    elev = fuzzify_feature_5(df["elev_m"])
    precip = fuzzify_feature_5(df["log_precip_bio12"])
    slope = fuzzify_feature_5(df["slope_deg"])
    spi = fuzzify_feature_5(df["log_spi"])
    twi = fuzzify_feature_5(df["twi"])

    low_out = np.zeros(len(df), dtype=float)
    med_out = np.zeros(len(df), dtype=float)
    high_out = np.zeros(len(df), dtype=float)
    vhigh_out = np.zeros(len(df), dtype=float)

    # 1 IF slope very_high AND precip very_high THEN very high
    r1 = np.minimum(slope["very_high"], precip["very_high"])
    vhigh_out = np.maximum(vhigh_out, r1)

    # 2 IF slope high AND precip high THEN very high
    r2 = np.minimum(slope["high"], precip["high"])
    vhigh_out = np.maximum(vhigh_out, r2)

    # 3 IF slope high AND spi high THEN high
    r3 = np.minimum(slope["high"], spi["high"])
    high_out = np.maximum(high_out, r3)

    # 4 IF twi high AND precip high THEN high
    r4 = np.minimum(twi["high"], precip["high"])
    high_out = np.maximum(high_out, r4)

    # 5 IF elev high AND precip high THEN high
    r5 = np.minimum(elev["high"], precip["high"])
    high_out = np.maximum(high_out, r5)

    # 6 IF slope very_low AND precip very_low THEN low
    r6 = np.minimum(slope["very_low"], precip["very_low"])
    low_out = np.maximum(low_out, r6)

    # 7 IF slope low AND precip low THEN low
    r7 = np.minimum(slope["low"], precip["low"])
    low_out = np.maximum(low_out, r7)

    # 8 IF slope med AND precip med AND twi med THEN medium
    r8 = np.minimum(np.minimum(slope["med"], precip["med"]), twi["med"])
    med_out = np.maximum(med_out, r8)

    # 9 IF slope high AND twi high THEN high
    r9 = np.minimum(slope["high"], twi["high"])
    high_out = np.maximum(high_out, r9)

    # 10 IF elev very_low AND precip very_low THEN low
    r10 = np.minimum(elev["very_low"], precip["very_low"])
    low_out = np.maximum(low_out, r10)

    # 11 IF precip very_high AND spi high THEN very high
    r11 = np.minimum(precip["very_high"], spi["high"])
    vhigh_out = np.maximum(vhigh_out, r11)

    # 12 IF slope med AND precip high THEN high
    r12 = np.minimum(slope["med"], precip["high"])
    high_out = np.maximum(high_out, r12)

    # Defuzzification
    c_low = 0.20
    c_med = 0.45
    c_high = 0.70
    c_vhigh = 0.90

    num = (
        low_out * c_low
        + med_out * c_med
        + high_out * c_high
        + vhigh_out * c_vhigh
    )
    den = low_out + med_out + high_out + vhigh_out

    score = np.divide(num, den + 1e-9)
    score = np.clip(score, 0.0, 1.0)
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
    parser.add_argument("--out", default=r"outputs\experiments\fuzzy_rule_spatial_cv_v2_memberships.csv")
    parser.add_argument("--log-experiments", default=r"outputs\experiments\experiments.csv")
    args = parser.parse_args()

    df = pd.read_parquet(args.data)

    required = [
        "elev_m",
        "log_precip_bio12",
        "slope_deg",
        "log_spi",
        "twi",
        LABEL_COL,
        GEOM_COL,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df = df.dropna(subset=required).copy()

    x, y = extract_xy_from_geometry(df)
    groups = grid_groups(x, y, args.grid_km)

    y_true = df[LABEL_COL].to_numpy(int)
    gkf = GroupKFold(n_splits=args.folds)

    rows = []
    roc_list, pr_list, f1_list, t_list = [], [], [], []

    for fold, (_, te) in enumerate(gkf.split(df, y_true, groups), start=1):
        df_te = df.iloc[te].copy()
        y_te = y_true[te]

        score = build_fuzzy_rule_score_v2(df_te)

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

    print("\nFuzzy Rule Based V2 (5 Memberships) Spatial CV Results")
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
        "dataset": "terrain_hydro_river_precip_log_fuzzy_rule_v2_memberships",
        "input_path": str(Path(args.data)),
        "rows": int(len(df)),
        "rows_used": int(len(df)),
        "pos_ratio": float(df[LABEL_COL].mean()),
        "pos_ratio_used": float(df[LABEL_COL].mean()),
        "label_col": LABEL_COL,
        "features": json.dumps(["elev_m", "log_precip_bio12", "slope_deg", "log_spi", "twi"]),
        "features_signature": "elev_m|log_precip_bio12|slope_deg|log_spi|twi",
        "grid_km": float(args.grid_km),
        "folds": int(args.folds),
        "seed": "",
        "model": "Fuzzy_rule_based_v2_memberships",
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
    print("Logged fuzzy rule model to:", args.log_experiments)


if __name__ == "__main__":
    main()