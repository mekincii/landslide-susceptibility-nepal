"""
Spatial Permutation Importance: Model Interpretability Audit

GOAL: Quantifies the contribution of each environmental feature to the model's
predictive performance by measuring the accuracy loss when feature
information is destroyed via random shuffling.

RESPONSIBILITIES:
- Executes a 'Model-Agnostic' Permutation Importance routine within a
  Spatial GroupKFold (20km grid) validation framework.
- Evaluates feature strength based on the 'Drop' in both ROC-AUC and
  Precision-Recall AUC (PR-AUC) when tested on spatially unseen regions.
- Distinguishes between globally robust predictors (consistent importance
  across folds) and regionally overfitted features (variable importance).
- Implements a resilient geometry-to-coordinate extractor (WKB/XY) to
  support grid-based spatial grouping.
- Exports both a raw 'per-fold' importance log and a statistical summary
  (Mean/Std) to identify the primary physical drivers of landslides in Nepal.

INPUT:  data/processed/features_nepal_rxx_realxxx_model_hydro_river_precip_clean.parquet
OUTPUTS: outputs/experiments/spatial_perm_importance.csv (Raw)
         outputs/experiments/spatial_perm_importance_summary.csv (Aggregated)
"""

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import argparse
import numpy as np
import pandas as pd


DATA_DEFAULT = r"data\processed\features_nepal_r05_real001_model_hydro_river_precip_clean.parquet"
LABEL_COL = "label"
GEOM_COL = "geometry"


def _ensure_aspect_sin_cos(df: pd.DataFrame) -> pd.DataFrame:
    if "aspect_sin" in df.columns and "aspect_cos" in df.columns:
        return df
    if "aspect_deg" not in df.columns:
        raise ValueError("Need either (aspect_sin, aspect_cos) or aspect_deg in dataset.")
    rad = np.deg2rad(df["aspect_deg"].astype(float))
    df["aspect_sin"] = np.sin(rad)
    df["aspect_cos"] = np.cos(rad)
    return df


def _extract_xy_from_geometry(df: pd.DataFrame):
    for xcol, ycol in [("x", "y"), ("easting", "northing"), ("lon", "lat")]:
        if xcol in df.columns and ycol in df.columns:
            return df[xcol].to_numpy(), df[ycol].to_numpy()

    try:
        from shapely import wkb
    except Exception as e:
        raise ImportError(
            "Need shapely to decode WKB geometry, or store x/y columns in the parquet."
        ) from e

    geom = df[GEOM_COL].values
    pts = [wkb.loads(g) if isinstance(g, (bytes, bytearray)) else g for g in geom]
    x = np.array([p.x for p in pts], dtype=float)
    y = np.array([p.y for p in pts], dtype=float)
    return x, y


def _grid_groups(x, y, grid_km: float):
    g = grid_km * 1000.0
    gx = np.floor(x / g).astype(int)
    gy = np.floor(y / g).astype(int)
    return np.char.add(gx.astype(str), np.char.add("_", gy.astype(str)))


def build_model(model_name: str, seed: int):
    if model_name == "rf_default":
        return RandomForestClassifier(
            n_estimators=400,
            random_state=seed,
            n_jobs=-1,
            class_weight=None,
        )
    if model_name == "rf_balanced":
        return RandomForestClassifier(
            n_estimators=400,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced",
        )
    if model_name == "lr_balanced_scaled":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    n_jobs=1
                ))
            ]
        )
    raise ValueError(f"Unknown model_name: {model_name}")


def perm_importance_on_fold(X_test, y_test, probs_test, feature_names, metric_fn, rng: np.random.Generator):
    base = metric_fn(y_test, probs_test)

    drops = {}
    for j, feat in enumerate(feature_names):
        Xp = X_test.copy()
        col = Xp[:, j].copy()
        rng.shuffle(col)
        Xp[:, j] = col

        drops[feat] = base  # placeholder
    return base, drops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA_DEFAULT)
    ap.add_argument("--grid-km", type=float, default=20.0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--model", type=str, default="rf_balanced",
                    choices=["rf_default", "rf_balanced", "lr_balanced_scaled"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=r"outputs\experiments\spatial_perm_importance.csv")
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    df = _ensure_aspect_sin_cos(df)

    FEATURES = [
        "elev_m", "slope_deg", "aspect_sin", "aspect_cos",
        "twi", "log_spi", "log_sca",
        "log_dist_river",
        "log_precip_bio12",
    ]
    missing = [c for c in FEATURES + [LABEL_COL, GEOM_COL] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    df = df.dropna(subset=FEATURES + [LABEL_COL]).copy()

    x, y = _extract_xy_from_geometry(df)
    groups = _grid_groups(x, y, grid_km=args.grid_km)

    X = df[FEATURES].to_numpy(dtype=float)
    y = df[LABEL_COL].to_numpy(dtype=int)

    gkf = GroupKFold(n_splits=args.folds)
    rng = np.random.default_rng(args.seed)

    model = build_model(args.model, args.seed)

    rows = []
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups), start=1):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]

        model.fit(Xtr, ytr)
        probs = model.predict_proba(Xte)[:, 1]

        base_roc = roc_auc_score(yte, probs)
        base_pr = average_precision_score(yte, probs)

        for j, feat in enumerate(FEATURES):
            Xp = Xte.copy()
            col = Xp[:, j].copy()
            rng.shuffle(col)
            Xp[:, j] = col

            probs_p = model.predict_proba(Xp)[:, 1]
            roc_p = roc_auc_score(yte, probs_p)
            pr_p = average_precision_score(yte, probs_p)

            rows.append({
                "grid_km": args.grid_km,
                "fold": fold,
                "model": args.model,
                "feature": feat,
                "base_roc": base_roc,
                "base_pr": base_pr,
                "roc_drop": base_roc - roc_p,
                "pr_drop": base_pr - pr_p,
            })

        print(f"Fold {fold} done | base ROC={base_roc:.4f} | base PR={base_pr:.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows)

    summary = (out_df
               .groupby(["model", "feature"], as_index=False)
               .agg(pr_drop_mean=("pr_drop", "mean"),
                    pr_drop_std=("pr_drop", "std"),
                    roc_drop_mean=("roc_drop", "mean"),
                    roc_drop_std=("roc_drop", "std"))
               .sort_values("pr_drop_mean", ascending=False))

    out_df.to_csv(out_path, index=False)
    summary_path = out_path.with_name(out_path.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("Wrote:", str(out_path))
    print("Wrote:", str(summary_path))
    print("\nTop features by PR-AUC drop:")
    print(summary.head(10).to_string(index=False))


if __name__ == "__main__":
    main()