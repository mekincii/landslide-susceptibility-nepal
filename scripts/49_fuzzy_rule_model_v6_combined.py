"""
Unified Expert Baseline: Combined Fuzzy Inference System (V6)

GOAL: Synthesizes all previous fuzzy logic iterations into a single,
comprehensive expert-knowledge baseline that combines precipitation
triggers with topographic and hydrologic pre-conditions.

RESPONSIBILITIES:
- Implements a 22-rule logic engine using Mamdani Min-Max inference
  to model the synergistic relationship between monsoon rainfall and
  mountainous terrain.
- Utilizes adaptive quantile-based membership functions to normalize
  Elevation, Slope, SPI, and TWI across the Nepal landscape.
- Conducts 'Center of Gravity' defuzzification to produce a high-fidelity
  susceptibility score (0.0 - 1.0) comparable to ML outputs.
- Benchmarks this 'Human-Logic Peak' against machine learning ensembles
  within a 20km Spatial GroupKFold framework.
- Persists final comparative metrics to the master experiment tracker to
  conclude the knowledge-driven baseline phase of the project.

INPUT:  data/processed/features_nepal_rxx_realxxx_model_hydro_river_precip_clean.parquet
OUTPUT: outputs/experiments/fuzzy_rule_spatial_cv_v6_combined.csv
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
    from shapely import wkb

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
            best_t = t

    return float(best_f1), float(best_t)


# 3-membership fuzzification
def quantile_membership_params(series: pd.Series):
    x = series.astype(float).to_numpy()
    q20 = np.nanquantile(x, 0.20)
    q50 = np.nanquantile(x, 0.50)
    q80 = np.nanquantile(x, 0.80)
    return q20, q50, q80


def low_membership(x, q20, q50):
    x = np.asarray(x)
    mu = np.zeros_like(x)

    mu[x <= q20] = 1

    mask = (x > q20) & (x < q50)
    mu[mask] = (q50 - x[mask]) / (q50 - q20 + 1e-9)

    return np.clip(mu, 0, 1)


def medium_membership(x, q20, q50, q80):
    x = np.asarray(x)
    mu = np.zeros_like(x)

    left = (x > q20) & (x < q50)
    right = (x >= q50) & (x < q80)

    mu[x == q50] = 1
    mu[left] = (x[left] - q20) / (q50 - q20 + 1e-9)
    mu[right] = (q80 - x[right]) / (q80 - q50 + 1e-9)

    return np.clip(mu, 0, 1)


def high_membership(x, q50, q80):
    x = np.asarray(x)
    mu = np.zeros_like(x)

    mu[x >= q80] = 1

    mask = (x > q50) & (x < q80)
    mu[mask] = (x[mask] - q50) / (q80 - q50 + 1e-9)

    return np.clip(mu, 0, 1)


def fuzzify_feature(series: pd.Series):

    q20, q50, q80 = quantile_membership_params(series)
    x = series.astype(float).to_numpy()

    return {
        "low": low_membership(x, q20, q50),
        "med": medium_membership(x, q20, q50, q80),
        "high": high_membership(x, q50, q80),
    }


# Combined fuzzy model
def build_fuzzy_rule_score_v6(df: pd.DataFrame):

    elev = fuzzify_feature(df["elev_m"])
    precip = fuzzify_feature(df["log_precip_bio12"])
    slope = fuzzify_feature(df["slope_deg"])
    spi = fuzzify_feature(df["log_spi"])
    twi = fuzzify_feature(df["twi"])

    low_out = np.zeros(len(df))
    med_out = np.zeros(len(df))
    high_out = np.zeros(len(df))
    vhigh_out = np.zeros(len(df))

    # Precipitation-triggered rules
    r1 = np.minimum(precip["high"], slope["high"])
    vhigh_out = np.maximum(vhigh_out, r1)

    r2 = np.minimum(precip["high"], spi["high"])
    vhigh_out = np.maximum(vhigh_out, r2)

    r3 = np.minimum(precip["high"], twi["high"])
    high_out = np.maximum(high_out, r3)

    r4 = np.minimum(precip["high"], elev["high"])
    high_out = np.maximum(high_out, r4)

    r5 = np.minimum(precip["high"], slope["med"])
    high_out = np.maximum(high_out, r5)

    r6 = np.minimum(precip["med"], slope["high"])
    high_out = np.maximum(high_out, r6)

    r7 = np.minimum(precip["high"], twi["med"])
    high_out = np.maximum(high_out, r7)

    r8 = np.minimum(precip["high"], spi["med"])
    high_out = np.maximum(high_out, r8)

    r9 = precip["high"]
    high_out = np.maximum(high_out, 0.8 * r9)

    # Expanded terrain rules
    r10 = np.minimum(slope["high"], twi["high"])
    high_out = np.maximum(high_out, r10)

    r11 = np.minimum(slope["high"], spi["high"])
    high_out = np.maximum(high_out, r11)

    r12 = np.minimum(elev["high"], slope["high"])
    high_out = np.maximum(high_out, r12)

    r13 = np.minimum(twi["high"], spi["high"])
    high_out = np.maximum(high_out, r13)

    r14 = np.minimum(elev["high"], precip["med"])
    high_out = np.maximum(high_out, r14)

    r15 = np.minimum(slope["med"], precip["med"])
    med_out = np.maximum(med_out, r15)

    r16 = np.minimum(twi["med"], spi["med"])
    med_out = np.maximum(med_out, r16)

    r17 = np.minimum(elev["med"], precip["med"])
    med_out = np.maximum(med_out, r17)

    # Low risk rules
    r18 = np.minimum(precip["low"], slope["low"])
    low_out = np.maximum(low_out, r18)

    r19 = precip["low"]
    low_out = np.maximum(low_out, r19)

    r20 = slope["low"]
    low_out = np.maximum(low_out, r20)

    r21 = np.minimum(twi["low"], spi["low"])
    low_out = np.maximum(low_out, r21)

    r22 = np.minimum(elev["low"], precip["low"])
    low_out = np.maximum(low_out, r22)

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
    score = np.clip(score, 0, 1)

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
    parser.add_argument("--grid-km", type=float, default=20)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--out", default=r"outputs\experiments\fuzzy_rule_spatial_cv_v6_combined.csv")
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

    df = df.dropna(subset=required).copy()

    x, y = extract_xy_from_geometry(df)
    groups = grid_groups(x, y, args.grid_km)

    y_true = df[LABEL_COL].to_numpy(int)

    gkf = GroupKFold(n_splits=args.folds)

    roc_list = []
    pr_list = []
    f1_list = []
    t_list = []

    rows = []

    for fold, (_, te) in enumerate(gkf.split(df, y_true, groups), start=1):

        df_te = df.iloc[te]

        score = build_fuzzy_rule_score_v6(df_te)

        y_te = y_true[te]

        roc = roc_auc_score(y_te, score)
        pr = average_precision_score(y_te, score)
        best_f1, best_t = best_f1_threshold(y_te, score)

        roc_list.append(roc)
        pr_list.append(pr)
        f1_list.append(best_f1)
        t_list.append(best_t)

        rows.append(
            {
                "fold": fold,
                "roc_auc": roc,
                "pr_auc": pr,
                "best_f1": best_f1,
                "best_t": best_t,
            }
        )

        print(
            f"Fold {fold} | ROC-AUC: {roc:.4f} | PR-AUC: {pr:.4f} | Best F1: {best_f1:.4f} @ t={best_t:.2f}"
        )

    mean_roc = float(np.mean(roc_list))
    std_roc = float(np.std(roc_list))
    mean_pr = float(np.mean(pr_list))
    std_pr = float(np.std(pr_list))
    mean_f1 = float(np.mean(f1_list))
    std_f1 = float(np.std(f1_list))
    mean_t = float(np.mean(t_list))

    print("\nFuzzy Rule Based V6 (Combined) Spatial CV Results ===")
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
        "dataset": "terrain_hydro_river_precip_log_fuzzy_rule_v6_combined",
        "input_path": str(Path(args.data)),
        "rows": int(len(df)),
        "rows_used": int(len(df)),
        "pos_ratio": float(df[LABEL_COL].mean()),
        "pos_ratio_used": float(df[LABEL_COL].mean()),
        "label_col": LABEL_COL,
        "features": json.dumps(["elev_m", "log_precip_bio12", "slope_deg", "log_spi", "twi"]),
        "grid_km": float(args.grid_km),
        "folds": int(args.folds),
        "model": "Fuzzy_rule_based_v6_combined",
        "roc_auc_mean": mean_roc,
        "roc_auc_std": std_roc,
        "pr_auc_mean": mean_pr,
        "pr_auc_std": std_pr,
        "f1_mean": mean_f1,
        "f1_std": std_f1,
        "best_t_mean": mean_t,
        "best_t_std": float(np.std(t_list)),
    }

    append_row(Path(args.log_experiments), log_row)

    print("Saved fold results:", out_path)
    print("Logged fuzzy rule model to:", args.log_experiments)


if __name__ == "__main__":
    main()