"""
Spatial Cross-Validation: Geographic Generalization Audit

GOAL: Evaluates the model's ability to predict landslides in entirely unseen
geographic regions by enforcing a strict spatial separation between training
and testing folds.

RESPONSIBILITIES:
- Implements a grid-based spatial partitioning strategy (default: 20km x 20km)
  to group training observations by geographic proximity.
- Utilizes 'GroupKFold' to ensure that entire geographic blocks are withheld
  from training, preventing data leakage via spatial autocorrelation.
- Enforces a projected metric Coordinate Reference System (UTM 45N) to
  ensure consistent grid dimensions across the Nepal study area.
- Benchmarks Random Forest and Logistic Regression architectures under
  spatially constrained conditions to detect geographic overfitting.
- Logs detailed performance metrics (ROC-AUC, PR-AUC, F1) to a central
  experiment database for long-term model versioning and auditability.

INPUT:  data/processed/features_nepal_rxx_realxxx_model_hydro_river_precip_clean.parquet
OUTPUT: Performance report on geographic generalization and logged results in outputs/experiments/.
"""

from __future__ import annotations
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from _utils.exp_logger import log_experiment

import argparse
import numpy as np
import pandas as pd
import geopandas as gpd


LABEL_COL = "label"
GEOM_COL = "geometry"
CRS_EXPECTED = "EPSG:32645"

FEATURES = [
    "elev_m",
    "slope_deg",
    "aspect_sin",
    "aspect_cos",
    "twi",
    "log_spi",
    "log_sca",
    "log_dist_river",
    "precip_bio12",
    "log_precip_bio12",
]


def ensure_aspect_sin_cos(df: pd.DataFrame) -> pd.DataFrame:
    if "aspect_sin" in df.columns and "aspect_cos" in df.columns:
        return df
    if "aspect_deg" not in df.columns:
        raise KeyError("Missing aspect features: need (aspect_sin, aspect_cos) or aspect_deg")
    ang = np.deg2rad(df["aspect_deg"].astype(float).to_numpy())
    df = df.copy()
    df["aspect_sin"] = np.sin(ang)
    df["aspect_cos"] = np.cos(ang)
    return df


def read_geoparquet_any(path: Path) -> gpd.GeoDataFrame:
    try:
        gdf = gpd.read_parquet(path)
        if GEOM_COL not in gdf.columns:
            raise ValueError("GeoDataFrame read succeeded but no geometry column found.")
        return gdf
    except Exception:
        # Fallback: pandas + WKB decode
        df = pd.read_parquet(path)
        if GEOM_COL not in df.columns:
            raise KeyError(f"Parquet missing '{GEOM_COL}' column.")
        from shapely import wkb
        geom = df[GEOM_COL].apply(lambda b: wkb.loads(b) if isinstance(b, (bytes, bytearray)) else b)
        gdf = gpd.GeoDataFrame(df.drop(columns=[GEOM_COL]), geometry=geom, crs=None)
        return gdf


def make_grid_groups(gdf: gpd.GeoDataFrame, grid_km: float) -> np.ndarray:
    if gdf.geometry is None:
        raise ValueError("GeoDataFrame has no geometry.")
    if gdf.crs is None:
        gdf = gdf.set_crs(CRS_EXPECTED, allow_override=True)

    # Enforce projected CRS
    if str(gdf.crs).upper() != CRS_EXPECTED:
        gdf = gdf.to_crs(CRS_EXPECTED)

    grid_m = grid_km * 1000.0
    xs = gdf.geometry.x.to_numpy()
    ys = gdf.geometry.y.to_numpy()

    minx = float(xs.min())
    miny = float(ys.min())

    ix = np.floor((xs - minx) / grid_m).astype(np.int64)
    iy = np.floor((ys - miny) / grid_m).astype(np.int64)

    groups = ix * 10_000_000 + iy
    return groups


def best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    thresholds = np.linspace(0.01, 0.99, 99)
    best_f1 = -1.0
    best_t = 0.5
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return float(best_f1), float(best_t)


def run_spatial_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model,
    n_splits: int,
    seed: int,
) -> dict:
    gkf = GroupKFold(n_splits=n_splits)

    rocs, prs, f1s, ts = [], [], [], []
    fold_rows = []

    for fold_idx, (tr, te) in enumerate(gkf.split(X, y, groups=groups), start=1):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]

        model.fit(Xtr, ytr)

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(Xte)[:, 1]
        else:
            dec = model.decision_function(Xte)
            prob = 1.0 / (1.0 + np.exp(-dec))

        roc = roc_auc_score(yte, prob)
        pr = average_precision_score(yte, prob)
        f1, t = best_f1_threshold(yte, prob)

        rocs.append(roc)
        prs.append(pr)
        f1s.append(f1)
        ts.append(t)

        fold_rows.append(
            {
                "fold": fold_idx,
                "roc_auc": roc,
                "pr_auc": pr,
                "best_f1": f1,
                "best_t": t,
                "n_test": int(len(te)),
                "pos_test": int(yte.sum()),
            }
        )

        print(f"Fold {fold_idx} | ROC-AUC: {roc:.4f} | PR-AUC: {pr:.4f} | Best F1: {f1:.4f} @ t={t:.2f}")

    return {
        "roc_mean": float(np.mean(rocs)),
        "roc_std": float(np.std(rocs)),
        "pr_mean": float(np.mean(prs)),
        "pr_std": float(np.std(prs)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "t_mean": float(np.mean(ts)),
        "t_std": float(np.std(ts)),
        "folds": fold_rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        type=str,
        default="data/processed/features_nepal_r05_real001_model_hydro_river_precip_clean.parquet",
        help="Modeling parquet to use (must include label + geometry + features).",
    )
    ap.add_argument("--grid-km", type=float, default=20.0, help="Spatial grid size in kilometers.")
    ap.add_argument("--folds", type=int, default=5, help="Number of spatial folds.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (for models).")
    ap.add_argument("--log-csv", type=str, default="outputs/experiments/experiments.csv", help="Log CSV output path.")
    args = ap.parse_args()

    data_path = Path(args.data)

    gdf = read_geoparquet_any(data_path)

    if gdf.crs is None:
        gdf = gdf.set_crs(CRS_EXPECTED, allow_override=True)
    elif str(gdf.crs).upper() != CRS_EXPECTED:
        gdf = gdf.to_crs(CRS_EXPECTED)

    gdf = ensure_aspect_sin_cos(gdf)

    missing_cols = [c for c in ([LABEL_COL] + FEATURES) if c not in gdf.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns: {missing_cols}")

    before = len(gdf)
    gdf = gdf.dropna(subset=FEATURES + [LABEL_COL]).copy()
    after = len(gdf)
    if after != before:
        print(f"Dropped rows with NaNs in features/label: {before - after} -> remaining {after}")

    groups = make_grid_groups(gdf, grid_km=args.grid_km)
    n_groups = int(pd.Series(groups).nunique())
    print(f"Spatial groups (grid cells): {n_groups}")

    folds = int(min(args.folds, n_groups))
    if folds < 2:
        raise RuntimeError(f"Not enough spatial groups ({n_groups}) for CV folds.")
    if folds != args.folds:
        print(f"Adjusted folds to {folds} (because only {n_groups} spatial groups exist)")

    X = gdf[FEATURES].astype(float).to_numpy()
    y = gdf[LABEL_COL].astype(int).to_numpy()

    pos_ratio = float(y.mean())
    print(f"Positive ratio: {pos_ratio:.4f}")
    print(f"Feature matrix shape: {X.shape}")

    rf_default = RandomForestClassifier(
        n_estimators=500,
        random_state=args.seed,
        n_jobs=-1,
    )
    rf_balanced = RandomForestClassifier(
        n_estimators=500,
        random_state=args.seed,
        n_jobs=-1,
        class_weight="balanced",
    )
    lr_balanced = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=1)),
        ]
    )

    models = [
        ("RF_default", rf_default),
        ("RF_balanced", rf_balanced),
        ("LR_balanced_scaled", lr_balanced),
    ]

    for model_name, model in models:
        print(f"\n=== {model_name} | Spatial CV (grid={args.grid_km}km, folds={folds}) ===")
        res = run_spatial_cv(X, y, groups, model, n_splits=folds, seed=args.seed)

        print("\nFinal Results")
        print(f"Mean ROC-AUC : {res['roc_mean']:.4f} ± {res['roc_std']:.4f}")
        print(f"Mean PR-AUC  : {res['pr_mean']:.4f} ± {res['pr_std']:.4f}")
        print(f"Mean F1      : {res['f1_mean']:.4f} ± {res['f1_std']:.4f}")
        print(f"Mean best t  : {res['t_mean']:.3f}")

        log_row = {
            "project": "landslide-nepal",
            "script": "34_spatial_cv_nepal.py",
            "dataset": str(data_path).replace("\\", "/"),
            "region": "nepal",
            "cv_type": "spatial_grid_groupkfold",
            "grid_km": float(args.grid_km),
            "n_folds": int(folds),
            "n_rows": int(len(gdf)),
            "pos_ratio": pos_ratio,
            "model": model_name,
            "features": "|".join(FEATURES),
            "roc_auc_mean": res["roc_mean"],
            "roc_auc_std": res["roc_std"],
            "pr_auc_mean": res["pr_mean"],
            "pr_auc_std": res["pr_std"],
            "f1_best_mean": res["f1_mean"],
            "f1_best_std": res["f1_std"],
            "best_t_mean": res["t_mean"],
            "seed": int(args.seed),
        }
        out_csv = log_experiment(log_row, out_csv=args.log_csv)
        print(f"Logged to: {out_csv.resolve()}")


if __name__ == "__main__":
    main()