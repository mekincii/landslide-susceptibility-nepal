"""
Spatial Probability Calibration & Reliability Audit

GOAL: Evaluates the statistical reliability of predicted landslide
probabilities to ensure that risk scores are physically meaningful
and well-calibrated across different geographic regions.

RESPONSIBILITIES:
- Computes Calibration Curves (Reliability Diagrams) within a spatial
  GroupKFold framework to compare predicted risk vs. observed frequency.
- Implements the 'Expected Calibration Error' (ECE) metric to quantify the
  weighted average gap between model confidence and actual outcomes.
- Calculates the Brier Score Loss to assess the overall probabilistic accuracy
  and discriminative power of the ensemble.
- Exports granular bin-level data for external visualization of the
  reliability gap across different susceptibility levels (0.0 to 1.0).
- Validates that the 'Balanced' Random Forest correctly maps probability
  thresholds to real-world hazard levels across the 20km spatial grid.

INPUT:  data/processed/features_nepal_rxx_realxxx_model_hydro_river_precip_clean.parquet
OUTPUTS: outputs/experiments/spatial_reliability.csv (Bin data)
         outputs/experiments/spatial_reliability_summary.txt (Global ECE/Brier)
"""

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import GroupKFold

import argparse
import numpy as np
import pandas as pd


DATA_DEFAULT = r"data\processed\features_nepal_r05_real001_model_hydro_river_precip_clean.parquet"
LABEL_COL = "label"
GEOM_COL = "geometry"


def _ensure_aspect_sin_cos(df: pd.DataFrame) -> pd.DataFrame:
    if "aspect_sin" in df.columns and "aspect_cos" in df.columns:
        return df
    rad = np.deg2rad(df["aspect_deg"].astype(float))
    df["aspect_sin"] = np.sin(rad)
    df["aspect_cos"] = np.cos(rad)
    return df


def _extract_xy_from_geometry(df: pd.DataFrame):
    try:
        from shapely import wkb
    except Exception as e:
        raise ImportError("Install shapely (pip install shapely) to decode WKB geometry.") from e

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


def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_prob, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.sum() / len(y_true)) * abs(acc - conf)
    return float(ece)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA_DEFAULT)
    ap.add_argument("--grid-km", type=float, default=20.0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=r"outputs\experiments\spatial_reliability.csv")
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    df = _ensure_aspect_sin_cos(df)

    FEATURES = [
        "elev_m", "slope_deg", "aspect_sin", "aspect_cos",
        "twi", "log_spi", "log_sca",
        "log_dist_river",
        "log_precip_bio12",
    ]
    df = df.dropna(subset=FEATURES + [LABEL_COL, GEOM_COL]).copy()

    x, y = _extract_xy_from_geometry(df)
    groups = _grid_groups(x, y, args.grid_km)

    X = df[FEATURES].to_numpy(float)
    y = df[LABEL_COL].to_numpy(int)

    gkf = GroupKFold(n_splits=args.folds)

    model = RandomForestClassifier(
        n_estimators=500,
        random_state=args.seed,
        n_jobs=-1,
        class_weight="balanced",
    )

    all_rows = []
    all_probs = []
    all_true = []

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups), start=1):
        model.fit(X[tr], y[tr])
        p = model.predict_proba(X[te])[:, 1]
        yt = y[te]

        brier = brier_score_loss(yt, p)
        ece = expected_calibration_error(yt, p, n_bins=args.bins)

        frac_pos, mean_pred = calibration_curve(yt, p, n_bins=args.bins, strategy="uniform")

        for i in range(len(frac_pos)):
            all_rows.append({
                "grid_km": args.grid_km,
                "fold": fold,
                "bin": i,
                "mean_pred": float(mean_pred[i]),
                "frac_pos": float(frac_pos[i]),
            })

        all_probs.append(p)
        all_true.append(yt)

        print(f"Fold {fold} | Brier={brier:.4f} | ECE={ece:.4f}")

    all_probs = np.concatenate(all_probs)
    all_true = np.concatenate(all_true)

    overall_brier = brier_score_loss(all_true, all_probs)
    overall_ece = expected_calibration_error(all_true, all_probs, n_bins=args.bins)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(out_path, index=False)

    summary_path = out_path.with_name(out_path.stem + "_summary.txt")
    summary_path.write_text(
        f"grid_km={args.grid_km}, folds={args.folds}, bins={args.bins}\n"
        f"overall_brier={overall_brier:.6f}\n"
        f"overall_ece={overall_ece:.6f}\n"
    )

    print("Wrote:", str(out_path))
    print("Wrote:", str(summary_path))
    print(f"Overall | Brier={overall_brier:.4f} | ECE={overall_ece:.4f}")


if __name__ == "__main__":
    main()