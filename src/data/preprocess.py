"""
Data preprocessing module for India AQI dataset.
Handles missing values, outliers, and data validation.
"""
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))
from config import POLLUTANT_COLS, AQI_POLLUTANTS, RANDOM_STATE


def load_city_day(filepath: str) -> pd.DataFrame:
    """Load city_day.csv with proper dtypes and date parsing.
    Handles both 'Date' (old dataset) and 'Datetime' (2015-2024 dataset) column names."""
    df = pd.read_csv(filepath)
    # Normalize date column name
    if "Datetime" in df.columns and "Date" not in df.columns:
        df = df.rename(columns={"Datetime": "Date"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["City", "Date"]).reset_index(drop=True)
    return df


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Generate a report of missing values per column."""
    missing = df.isnull().sum()
    pct = (missing / len(df) * 100).round(2)
    report = pd.DataFrame({"Missing": missing, "Pct_Missing": pct})
    return report[report["Missing"] > 0].sort_values("Pct_Missing", ascending=False)


def interpolate_within_groups(df: pd.DataFrame, group_col: str = "City",
                               cols: list = None, limit: int = 3) -> pd.DataFrame:
    """
    Tier 1 imputation: Linear interpolation within city groups.
    Only fills gaps of <= `limit` consecutive NaN values.
    """
    if cols is None:
        cols = [c for c in POLLUTANT_COLS if c in df.columns]

    df = df.copy()
    for col in cols:
        df[col] = df.groupby(group_col)[col].transform(
            lambda x: x.interpolate(method="linear", limit=limit, limit_direction="both")
        )
    return df


def knn_impute(df: pd.DataFrame, cols: list = None, n_neighbors: int = 5) -> pd.DataFrame:
    """
    Tier 2 imputation: KNN imputation using correlated pollutant columns.
    Works on the principle that pollutants are inter-correlated.
    """
    if cols is None:
        cols = [c for c in POLLUTANT_COLS if c in df.columns]

    df = df.copy()
    imputer = KNNImputer(n_neighbors=n_neighbors, weights="distance")
    df[cols] = imputer.fit_transform(df[cols])
    return df


def drop_sparse_rows(df: pd.DataFrame, cols: list = None,
                      threshold: float = 0.5) -> pd.DataFrame:
    """
    Tier 3: Drop rows where more than `threshold` fraction of
    pollutant columns are NaN (after imputation attempts).
    """
    if cols is None:
        cols = [c for c in POLLUTANT_COLS if c in df.columns]

    nan_frac = df[cols].isnull().sum(axis=1) / len(cols)
    return df[nan_frac <= threshold].reset_index(drop=True)


def handle_outliers_iqr(df: pd.DataFrame, cols: list = None,
                         factor: float = 1.5) -> pd.DataFrame:
    """
    Winsorize outliers: cap values at Q1 - factor*IQR and Q3 + factor*IQR.
    Preserves data points (doesn't drop rows), just caps extreme values.
    """
    if cols is None:
        cols = [c for c in POLLUTANT_COLS if c in df.columns]

    df = df.copy()
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        # Pollutant concentrations can't be negative
        lower = max(lower, 0)
        df[col] = df[col].clip(lower=lower, upper=upper)
    return df


def full_preprocessing_pipeline(filepath: str) -> pd.DataFrame:
    """
    Run the complete preprocessing pipeline:
    1. Load data
    2. Linear interpolation (gap <= 3)
    3. KNN imputation for remaining NaN
    4. Drop rows with > 50% still missing
    5. Winsorize outliers
    6. Drop rows with no AQI (either original or calculable)
    """
    print("Loading data...")
    df = load_city_day(filepath)
    print(f"  Raw shape: {df.shape}")

    print("Step 1: Linear interpolation (within city groups, gap <= 3)...")
    df = interpolate_within_groups(df)

    print("Step 2: KNN imputation (k=5, distance-weighted)...")
    df = knn_impute(df)

    print("Step 3: Dropping rows with > 50% pollutants still missing...")
    before = len(df)
    df = drop_sparse_rows(df)
    print(f"  Dropped {before - len(df)} rows")

    print("Step 4: Winsorizing outliers (IQR x 1.5)...")
    df = handle_outliers_iqr(df)

    print("Step 5: Dropping rows without valid AQI...")
    df = df.dropna(subset=["AQI"]).reset_index(drop=True)
    print(f"  Final shape: {df.shape}")

    return df
