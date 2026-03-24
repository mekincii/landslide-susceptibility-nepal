"""
Climatological Dataset Sanitization & Quality Audit

GOAL: Finalizes the multi-hazard feature set by removing observations
lacking precipitation data and verifying the statistical integrity of
the remaining training samples.

RESPONSIBILITIES:
- Loads the integrated dataset containing Topographic, Hydrologic,
  Fluvial, and Climatological (CHELSA Bio12) features.
- Executes a 'Complete Case' filter, dropping any points where
  precipitation values are missing (NaN), typically at the study area
  margins or in extreme high-altitude zones.
- Quantifies the data loss incurred by the precipitation integration
  to ensure the training set remains representative of the Nepal landscape.
- Audits the post-cleaning class balance (Positive vs. Negative samples)
  to confirm the Neg:Pos ratio remains stable for machine learning.
- Persists the high-fidelity 'ML-ready' dataset to an optimized Parquet
  file, ready for the final ensemble model run.

INPUT:  data/processed/features_nepal_rxx_realxxx_model_hydro_river_precip.parquet
OUTPUT: data/processed/features_nepal_rxx_realxxx_model_hydro_river_precip_clean.parquet
"""

from pathlib import Path

import pandas as pd

IN_PATH = Path("data/processed/features_nepal_r05_real001_model_hydro_river_precip.parquet")
OUT_PATH = Path("data/processed/features_nepal_r05_real001_model_hydro_river_precip_clean.parquet")

REQ_COLS = ["precip_bio12", "log_precip_bio12"]


def main():
    in_abs = IN_PATH.resolve()
    out_abs = OUT_PATH.resolve()

    print(f"Loading: {in_abs}")
    df = pd.read_parquet(IN_PATH)
    print(f"Rows: {len(df)}")

    print("\nMissingness (%) before")
    miss = df[REQ_COLS].isna().mean() * 100
    print(miss)

    before = len(df)
    df2 = df.dropna(subset=REQ_COLS).copy()
    dropped = before - len(df2)

    print(f"\nDropped rows (NaN in {REQ_COLS}): {dropped} / {before} ({(dropped/before)*100:.2f}%)")
    print(f"Remaining rows: {len(df2)}")

    if "label" in df2.columns:
        pos = int((df2["label"] == 1).sum())
        neg = int((df2["label"] == 0).sum())
        print("\nClass balance (after)")
        print(f"Positives: {pos}")
        print(f"Negatives: {neg}")
        if pos > 0:
            print(f"Neg:Pos ratio: {neg/pos:.2f} :1")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting: {out_abs}")
    df2.to_parquet(OUT_PATH, index=False)


if __name__ == "__main__":
    main()