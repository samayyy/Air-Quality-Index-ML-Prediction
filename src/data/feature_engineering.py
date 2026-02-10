"""
Feature engineering module for AQI prediction.
Creates temporal, lag, rolling, and interaction features from cleaned data.
"""
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))
from config import INDIA_SEASONS, LAG_DAYS, ROLLING_WINDOWS, POLLUTANT_COLS


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract temporal features from the Date column."""
    df = df.copy()
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["day_of_year"] = df["Date"].dt.dayofyear
    df["quarter"] = df["Date"].dt.quarter
    df["is_weekend"] = (df["Date"].dt.dayofweek >= 5).astype(int)
    df["season"] = df["month"].map(INDIA_SEASONS)
    return df


def add_lag_features(df: pd.DataFrame, target_cols: list = None,
                      group_col: str = "City") -> pd.DataFrame:
    """
    Create lag features for specified columns, grouped by city.
    Uses shift() which naturally prevents future data leakage.
    """
    if target_cols is None:
        target_cols = [c for c in POLLUTANT_COLS if c in df.columns]

    df = df.sort_values([group_col, "Date"]).copy()
    for col in target_cols:
        for lag in LAG_DAYS:
            df[f"{col}_lag_{lag}"] = df.groupby(group_col)[col].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, target_cols: list = None,
                          group_col: str = "City") -> pd.DataFrame:
    """
    Create rolling mean and std features for specified columns.
    min_periods=1 ensures we get values even at the start of each city's data.
    """
    if target_cols is None:
        # Use key AQI pollutants only (not all 12) to limit feature explosion
        target_cols = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
        target_cols = [c for c in target_cols if c in df.columns]

    df = df.sort_values([group_col, "Date"]).copy()
    for col in target_cols:
        for window in ROLLING_WINDOWS:
            rolled = df.groupby(group_col)[col].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
            )
            df[f"{col}_rolling_mean_{window}"] = rolled

            rolled_std = df.groupby(group_col)[col].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).std()
            )
            df[f"{col}_rolling_std_{window}"] = rolled_std
    return df


def add_pollutant_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create ratio features between related pollutants.
    PM2.5/PM10 ratio indicates combustion vs dust source.
    """
    df = df.copy()
    if "PM2.5" in df.columns and "PM10" in df.columns:
        df["PM25_PM10_ratio"] = df["PM2.5"] / df["PM10"].replace(0, np.nan)

    if "NO2" in df.columns and "NO" in df.columns:
        df["NO2_NO_ratio"] = df["NO2"] / df["NO"].replace(0, np.nan)

    return df


def encode_city(df: pd.DataFrame, method: str = "label") -> pd.DataFrame:
    """Encode the City column. Default: label encoding (ordinal)."""
    df = df.copy()
    if method == "label":
        df["City_encoded"] = df["City"].astype("category").cat.codes
    elif method == "onehot":
        dummies = pd.get_dummies(df["City"], prefix="City", drop_first=True)
        df = pd.concat([df, dummies], axis=1)
    return df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering pipeline.
    Returns DataFrame with all features added.
    """
    print("Adding temporal features...")
    df = add_temporal_features(df)

    print("Adding lag features...")
    df = add_lag_features(df)

    print("Adding rolling features...")
    df = add_rolling_features(df)

    print("Adding pollutant ratios...")
    df = add_pollutant_ratios(df)

    print("Encoding city...")
    df = encode_city(df)

    # Drop rows with NaN from lag/rolling (first few rows per city)
    initial_len = len(df)
    df = df.dropna(subset=[c for c in df.columns if "lag_30" in c]).reset_index(drop=True)
    print(f"Dropped {initial_len - len(df)} rows due to insufficient lag history")

    print(f"Final feature matrix: {df.shape}")
    return df
