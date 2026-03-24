"""
Fuzzy Susceptibility Inference: Expert-Knowledge Probability Mapping

GOAL: Applies the finalized Mamdani Fuzzy Inference System (FIS) to the entire
study area to generate a knowledge-driven landslide susceptibility map.

RESPONSIBILITIES:
- Executes the fuzzy logic engine across the full multidimensional feature
  set (Elevation, Precipitation, Slope, SPI, and TWI).
- Fuzzifies continuous environmental data into linguistic sets using
  statistically derived quantile memberships.
- Applies 10 geomorphological 'If-Then' rules to determine hazard activation
  levels based on expert-identified environmental triggers.
- Defuzzifies the qualitative results into a 0.0-1.0 'fuzzy_prob' index
  using the Center of Gravity (CoG) method.
- Exports a spatially referenced CSV for comparison against the Random
  Forest model and final cartographic production.

INPUT:  data/processed/features_nepal_rxx_realxxx_model_hydro_river_precip_clean.parquet
OUTPUT: outputs/maps/nepal_fuzzy_probabilities.csv
"""

from __future__ import annotations
from pathlib import Path

import argparse
import numpy as np
import pandas as pd


DATA_DEFAULT = r"data\processed\features_nepal_r05_real001_model_hydro_river_precip_clean.parquet"
OUTPUT_DEFAULT = r"outputs\maps\nepal_fuzzy_probabilities.csv"

LABEL_COL = "label"

# Fuzzy memberships
def quantile_membership_params(series: pd.Series):
    x = series.astype(float).to_numpy()
    q20 = np.nanquantile(x, 0.20)
    q50 = np.nanquantile(x, 0.50)
    q80 = np.nanquantile(x, 0.80)
    return q20, q50, q80


def low_membership(x, q20, q50):
    x = np.asarray(x, dtype=float)
    mu = np.zeros_like(x)

    mu[x <= q20] = 1.0
    mask = (x > q20) & (x < q50)
    mu[mask] = (q50 - x[mask]) / (q50 - q20 + 1e-9)

    return np.clip(mu, 0.0, 1.0)


def medium_membership(x, q20, q50, q80):
    x = np.asarray(x, dtype=float)
    mu = np.zeros_like(x)

    left = (x > q20) & (x < q50)
    right = (x >= q50) & (x < q80)

    mu[left] = (x[left] - q20) / (q50 - q20 + 1e-9)
    mu[right] = (q80 - x[right]) / (q80 - q50 + 1e-9)

    mu[x == q50] = 1.0

    return np.clip(mu, 0.0, 1.0)


def high_membership(x, q50, q80):
    x = np.asarray(x, dtype=float)
    mu = np.zeros_like(x)

    mu[x >= q80] = 1.0
    mask = (x > q50) & (x < q80)
    mu[mask] = (x[mask] - q50) / (q80 - q50 + 1e-9)

    return np.clip(mu, 0.0, 1.0)


def fuzzify_feature(series: pd.Series):
    q20, q50, q80 = quantile_membership_params(series)
    x = series.astype(float).to_numpy()

    return {
        "low": low_membership(x, q20, q50),
        "med": medium_membership(x, q20, q50, q80),
        "high": high_membership(x, q50, q80),
    }


# Final fuzzy rule model
def fuzzy_rule_score(df: pd.DataFrame) -> np.ndarray:

    elev = fuzzify_feature(df["elev_m"])
    precip = fuzzify_feature(df["log_precip_bio12"])
    slope = fuzzify_feature(df["slope_deg"])
    spi = fuzzify_feature(df["log_spi"])
    twi = fuzzify_feature(df["twi"])

    low_out = np.zeros(len(df))
    med_out = np.zeros(len(df))
    high_out = np.zeros(len(df))
    vhigh_out = np.zeros(len(df))

    r1 = np.minimum(slope["high"], precip["high"])
    vhigh_out = np.maximum(vhigh_out, r1)

    r2 = np.minimum(slope["high"], spi["high"])
    high_out = np.maximum(high_out, r2)

    r3 = np.minimum(twi["high"], precip["high"])
    high_out = np.maximum(high_out, r3)

    r4 = np.minimum(elev["high"], precip["high"])
    high_out = np.maximum(high_out, r4)

    r5 = np.minimum(slope["low"], precip["low"])
    low_out = np.maximum(low_out, r5)

    r6 = np.minimum(np.minimum(slope["med"], precip["med"]), twi["med"])
    med_out = np.maximum(med_out, r6)

    r7 = np.minimum(slope["high"], twi["high"])
    high_out = np.maximum(high_out, r7)

    r8 = np.minimum(elev["low"], precip["low"])
    low_out = np.maximum(low_out, r8)

    r9 = np.minimum(precip["high"], spi["high"])
    vhigh_out = np.maximum(vhigh_out, r9)

    r10 = np.minimum(slope["med"], precip["high"])
    high_out = np.maximum(high_out, r10)

    # Defuzzification
    c_low = 0.20
    c_med = 0.45
    c_high = 0.70
    c_vhigh = 0.90

    numerator = (
        low_out * c_low +
        med_out * c_med +
        high_out * c_high +
        vhigh_out * c_vhigh
    )

    denominator = low_out + med_out + high_out + vhigh_out + 1e-9

    score = numerator / denominator
    score = np.clip(score, 0.0, 1.0)

    return score


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data", default=DATA_DEFAULT)
    parser.add_argument("--out", default=OUTPUT_DEFAULT)

    args = parser.parse_args()

    df = pd.read_parquet(args.data)

    required = [
        "elev_m",
        "log_precip_bio12",
        "slope_deg",
        "log_spi",
        "twi",
    ]

    df = df.dropna(subset=required).copy()

    df["fuzzy_prob"] = fuzzy_rule_score(df)

    output_cols = [
        "longitude",
        "latitude",
        "fuzzy_prob",
        LABEL_COL,
    ]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    df_map = df.dropna(subset=["longitude", "latitude"]).copy()
    df_map[output_cols].to_csv(args.out, index=False)

    print("Saved:", args.out)


if __name__ == "__main__":
    main()