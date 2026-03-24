"""
Systematic Experimental Orchestrator: Multi-Scale Spatial Validation

GOAL: Executes a comprehensive sweep across multiple dataset variants, model
architectures, and spatial grid scales to identify the optimal landslide
susceptibility configuration for Nepal.

RESPONSIBILITIES:
- Iterates through hierarchical DatasetSpecs (Terrain -> Hydro -> River -> Precip)
  to quantify the incremental value of each environmental feature layer.
- Performs Multi-Scale Spatial Cross-Validation by varying the geographic
  block size (10km, 20km, 50km), testing the model's spatial extrapolation limits.
- Evaluates model performance using a robust metric suite: ROC-AUC, PR-AUC,
  Optimized F1, and Brier Score (for probability calibration).
- Implements a resilient spatial grouping engine that handles diverse
  coordinate and geometry formats (WKB, X/Y, Point).
- Maintains a persistent experimental 'Leaderboard' (summary_best.csv),
  automatically aggregating and sorting results from multiple runs to track
  project progress.

INPUTS:  All cumulative Parquet feature sets in data/processed/
OUTPUTS: outputs/experiments/experiments.csv (Detailed Log)
         outputs/experiments/summary_best.csv (Leaderboard)
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from shapely.geometry import Point
from shapely import wkb as shapely_wkb
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    brier_score_loss,
)

import argparse
import json
import numpy as np
import pandas as pd

try:
    import geopandas as gpd
except ImportError:
    gpd = None


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs" / "experiments"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENTS_CSV = OUT_DIR / "experiments.csv"
SUMMARY_CSV = OUT_DIR / "summary_best.csv"


@dataclass
class DatasetSpec:
    name: str
    path: Path
    base_features: List[str]


def _default_dataset_specs(precip_variant: str) -> List[DatasetSpec]:
    precip_feats: List[str]
    if precip_variant == "raw":
        precip_feats = ["precip_bio12"]
    elif precip_variant == "log":
        precip_feats = ["log_precip_bio12"]
    elif precip_variant == "both":
        precip_feats = ["precip_bio12", "log_precip_bio12"]
    else:
        raise ValueError(f"Unknown precip_variant={precip_variant}")

    specs = [
        DatasetSpec(
            name="terrain_only",
            path=DATA / "features_nepal_r05_real001_clean.parquet",
            base_features=["elev_m", "slope_deg", "aspect_deg"],  # aspect gets expanded to sin/cos
        ),
        DatasetSpec(
            name="terrain_hydro",
            path=DATA / "features_nepal_r05_real001_model_hydro.parquet",
            base_features=["elev_m", "slope_deg", "aspect_deg", "log_spi", "log_sca", "twi"],
        ),
        DatasetSpec(
            name="terrain_hydro_river",
            path=DATA / "features_nepal_r05_real001_model_hydro_river.parquet",
            base_features=["elev_m", "slope_deg", "aspect_deg", "log_spi", "log_sca", "twi", "log_dist_river"],
        ),
        DatasetSpec(
            name=f"terrain_hydro_river_precip_{precip_variant}",
            path=DATA / "features_nepal_r05_real001_model_hydro_river_precip_clean.parquet",
            base_features=[
                "elev_m",
                "slope_deg",
                "aspect_deg",
                "log_spi",
                "log_sca",
                "twi",
                "log_dist_river",
                *precip_feats,
            ],
        ),
    ]
    return specs


def _now_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_geometry(df: pd.DataFrame, crs: Optional[str] = None):
    if "geometry" not in df.columns:
        for xcol, ycol in [("x", "y"), ("easting", "northing"), ("lon", "lat")]:
            if xcol in df.columns and ycol in df.columns:
                x = df[xcol].to_numpy()
                y = df[ycol].to_numpy()
                df = df.copy()
                df["geometry"] = [Point(float(xx), float(yy)) for xx, yy in zip(x, y)]
                return df, x, y
        raise ValueError("No 'geometry' column found and could not infer x/y columns.")

    geom = df["geometry"].iloc[0]
    if isinstance(geom, (bytes, bytearray, memoryview)):
        geoms = df["geometry"].apply(lambda b: shapely_wkb.loads(bytes(b)))
        x = geoms.apply(lambda g: g.x).to_numpy()
        y = geoms.apply(lambda g: g.y).to_numpy()
        df = df.copy()
        df["geometry"] = geoms
        return df, x, y

    try:
        x = df["geometry"].apply(lambda g: g.x).to_numpy()
        y = df["geometry"].apply(lambda g: g.y).to_numpy()
        return df, x, y
    except Exception as e:
        raise TypeError(f"Unsupported geometry type in column 'geometry': {type(geom)}") from e


def _find_label_column(df: pd.DataFrame) -> str:
    candidates = ["label", "y", "target", "class", "is_landslide", "landslide"]
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Could not find label column. Tried: {candidates}. Available cols: {list(df.columns)[:50]}...")


def _make_aspect_sin_cos(df: pd.DataFrame) -> pd.DataFrame:
    if "aspect_sin" in df.columns and "aspect_cos" in df.columns:
        return df
    if "aspect_deg" in df.columns:
        a = np.deg2rad(df["aspect_deg"].astype(float).to_numpy())
        df = df.copy()
        df["aspect_sin"] = np.sin(a)
        df["aspect_cos"] = np.cos(a)
    return df


def _resolve_features(df: pd.DataFrame, base_features: List[str]) -> List[str]:
    feats: List[str] = []
    for f in base_features:
        if f == "aspect_deg":
            df2 = _make_aspect_sin_cos(df)
            feats.extend(["aspect_sin", "aspect_cos"])
        else:
            feats.append(f)
    return feats


def _grid_groups(x: np.ndarray, y: np.ndarray, grid_km: float) -> np.ndarray:
    cell = grid_km * 1000.0
    gx = np.floor(x / cell).astype(np.int64)
    gy = np.floor(y / cell).astype(np.int64)
    return gx.astype(str) + "_" + gy.astype(str)


def _best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
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


def _make_models(random_state: int = 42):
    rf_default = RandomForestClassifier(
        n_estimators=500,
        random_state=random_state,
        n_jobs=-1,
    )
    rf_balanced = RandomForestClassifier(
        n_estimators=500,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )
    lr_bal_scaled = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=4000,
                class_weight="balanced",
                solver="lbfgs",
                n_jobs=None,  # avoid the warning you saw
                random_state=random_state,
            )),
        ]
    )
    return {
        "RF_default": rf_default,
        "RF_balanced": rf_balanced,
        "LR_balanced_scaled": lr_bal_scaled,
    }


def _append_log_row(csv_path: Path, row: Dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_row = pd.DataFrame([row])

    if csv_path.exists():
        df_row.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(csv_path, mode="w", header=True, index=False)


def _refresh_summary(experiments_csv: Path, summary_csv: Path):
    if not experiments_csv.exists():
        return

    df = pd.read_csv(experiments_csv)
    group_cols = ["dataset", "model", "grid_km", "folds", "features_signature"]

    agg = df.groupby(group_cols).agg(
        rows=("rows", "max"),
        pos_ratio=("pos_ratio", "mean"),
        roc_auc_mean=("roc_auc_mean", "mean"),
        roc_auc_std=("roc_auc_std", "mean"),
        pr_auc_mean=("pr_auc_mean", "mean"),
        pr_auc_std=("pr_auc_std", "mean"),
        f1_mean=("f1_mean", "mean"),
        f1_std=("f1_std", "mean"),
        best_t_mean=("best_t_mean", "mean"),
        brier_mean=("brier_mean", "mean"),
        run_id=("run_id", "last"),
        timestamp=("timestamp", "last"),
        input_path=("input_path", "last"),
    ).reset_index()

    agg = agg.sort_values(["pr_auc_mean", "roc_auc_mean"], ascending=False)
    agg.to_csv(summary_csv, index=False)


def _features_signature(features: List[str]) -> str:
    return "|".join(features)


def run_spatial_cv(
    *,
    df: pd.DataFrame,
    features: List[str],
    label_col: str,
    x: np.ndarray,
    y: np.ndarray,
    grid_km: float,
    folds: int,
    model_name: str,
    model,
    random_state: int,
) -> Dict:
    use_cols = features + [label_col]
    df2 = df.dropna(subset=use_cols).copy()
    X = df2[features].to_numpy(dtype=float)
    y_true_all = df2[label_col].to_numpy(dtype=int)

    idx = df2.index.to_numpy()
    x2 = x[idx]
    y2 = y[idx]

    groups = _grid_groups(x2, y2, grid_km=grid_km)
    if len(np.unique(groups)) < folds:
        raise RuntimeError(
            f"Not enough spatial groups ({len(np.unique(groups))}) for folds={folds}. "
            f"Increase grid_km or reduce folds."
        )

    gkf = GroupKFold(n_splits=folds)

    roc_list, pr_list, f1_list, t_list, brier_list = [], [], [], [], []

    for i, (tr, te) in enumerate(gkf.split(X, y_true_all, groups=groups), start=1):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y_true_all[tr], y_true_all[te]

        clf = model
        clf.fit(Xtr, ytr)

        if hasattr(clf, "predict_proba"):
            p = clf.predict_proba(Xte)[:, 1]
        else:
            s = clf.decision_function(Xte)
            p = 1.0 / (1.0 + np.exp(-s))

        roc = roc_auc_score(yte, p)
        pr = average_precision_score(yte, p)
        best_f1, best_t = _best_f1_threshold(yte, p)
        brier = brier_score_loss(yte, p)

        roc_list.append(float(roc))
        pr_list.append(float(pr))
        f1_list.append(float(best_f1))
        t_list.append(float(best_t))
        brier_list.append(float(brier))

        print(f"Fold {i} | ROC-AUC: {roc:.4f} | PR-AUC: {pr:.4f} | Best F1: {best_f1:.4f} @ t={best_t:.2f}")

    out = {
        "rows_used": int(X.shape[0]),
        "pos_ratio_used": float(y_true_all.mean()),
        "roc_auc_mean": float(np.mean(roc_list)),
        "roc_auc_std": float(np.std(roc_list, ddof=1) if len(roc_list) > 1 else 0.0),
        "pr_auc_mean": float(np.mean(pr_list)),
        "pr_auc_std": float(np.std(pr_list, ddof=1) if len(pr_list) > 1 else 0.0),
        "f1_mean": float(np.mean(f1_list)),
        "f1_std": float(np.std(f1_list, ddof=1) if len(f1_list) > 1 else 0.0),
        "best_t_mean": float(np.mean(t_list)),
        "best_t_std": float(np.std(t_list, ddof=1) if len(t_list) > 1 else 0.0),
        "brier_mean": float(np.mean(brier_list)),
        "brier_std": float(np.std(brier_list, ddof=1) if len(brier_list) > 1 else 0.0),
        "folds": int(folds),
        "grid_km": float(grid_km),
        "model": model_name,
        "random_state": int(random_state),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--grid-km", type=float, nargs="+", default=[10.0, 20.0, 50.0])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--precip-variant", type=str, default="log", choices=["raw", "log", "both"])
    args = ap.parse_args()

    run_id = _now_run_id()
    timestamp = datetime.now().isoformat(timespec="seconds")

    specs = _default_dataset_specs(args.precip_variant)

    for spec in specs:
        if not spec.path.exists():
            print(f"Skipping dataset (file missing): {spec.name} -> {spec.path}")
            continue

        print(f"\n")
        print(f"Dataset: {spec.name}")
        print(f"Path: {spec.path}")
        df = pd.read_parquet(spec.path)
        print(f"Rows: {len(df)} | Cols: {df.shape[1]}")

        label_col = _find_label_column(df)

        df, x, y = _ensure_geometry(df)

        df = _make_aspect_sin_cos(df)

        features = _resolve_features(df, spec.base_features)

        missing = [f for f in features if f not in df.columns]
        if missing:
            print(f"Missing features in {spec.name}: {missing}")
            print(f"Available columns (sample): {list(df.columns)[:40]}")
            continue

        feat_sig = _features_signature(features)

        models = _make_models(random_state=args.seed)

        for grid_km in args.grid_km:
            groups = _grid_groups(x, y, grid_km=float(grid_km))
            print(f"\nGrid={grid_km} km | spatial groups: {len(np.unique(groups))} | label={label_col}")
            print(f"Features ({len(features)}): {features}")

            for model_name, model in models.items():
                print(f"\n=== {model_name} | Spatial CV (grid={grid_km}km, folds={args.folds}) ===")

                results = run_spatial_cv(
                    df=df,
                    features=features,
                    label_col=label_col,
                    x=x,
                    y=y,
                    grid_km=float(grid_km),
                    folds=int(args.folds),
                    model_name=model_name,
                    model=model,
                    random_state=int(args.seed),
                )

                print(f"\nFinal Spatial CV Results: {results}")
                print(f"Mean ROC-AUC : {results['roc_auc_mean']:.4f} ± {results['roc_auc_std']:.4f}")
                print(f"Mean PR-AUC  : {results['pr_auc_mean']:.4f} ± {results['pr_auc_std']:.4f}")
                print(f"Mean F1      : {results['f1_mean']:.4f} ± {results['f1_std']:.4f}")
                print(f"Mean best t  : {results['best_t_mean']:.3f}")
                print(f"Mean Brier   : {results['brier_mean']:.4f}")

                row = {
                    "timestamp": timestamp,
                    "run_id": run_id,
                    "dataset": spec.name,
                    "input_path": str(spec.path),
                    "rows": int(len(df)),
                    "rows_used": int(results["rows_used"]),
                    "pos_ratio": float(df[label_col].mean()),
                    "pos_ratio_used": float(results["pos_ratio_used"]),
                    "label_col": label_col,
                    "features": json.dumps(features),
                    "features_signature": feat_sig,
                    "grid_km": float(grid_km),
                    "folds": int(args.folds),
                    "seed": int(args.seed),
                    "model": model_name,
                    "roc_auc_mean": results["roc_auc_mean"],
                    "roc_auc_std": results["roc_auc_std"],
                    "pr_auc_mean": results["pr_auc_mean"],
                    "pr_auc_std": results["pr_auc_std"],
                    "f1_mean": results["f1_mean"],
                    "f1_std": results["f1_std"],
                    "best_t_mean": results["best_t_mean"],
                    "best_t_std": results["best_t_std"],
                    "brier_mean": results["brier_mean"],
                    "brier_std": results["brier_std"],
                }
                _append_log_row(EXPERIMENTS_CSV, row)
                print(f"Logged to: {EXPERIMENTS_CSV}")

    _refresh_summary(EXPERIMENTS_CSV, SUMMARY_CSV)
    print(f"\nSummary written to: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()