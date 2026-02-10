"""
CPCB (Central Pollution Control Board) AQI Calculator for India.

The AQI is calculated using piecewise linear interpolation within breakpoint
ranges defined by CPCB. The final AQI = max(all sub-indices).
Minimum 3 pollutants required, at least one must be PM2.5 or PM10.
"""
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))
from config import AQI_BREAKPOINTS, AQI_CATEGORIES


def compute_sub_index(pollutant: str, concentration: float) -> float | None:
    """
    Calculate AQI sub-index for a single pollutant using CPCB breakpoints.
    Returns None if concentration is out of range or pollutant not recognized.
    """
    if pd.isna(concentration) or pollutant not in AQI_BREAKPOINTS:
        return None

    for c_low, c_high, i_low, i_high in AQI_BREAKPOINTS[pollutant]:
        if c_low <= concentration <= c_high:
            return i_low + (concentration - c_low) * (i_high - i_low) / (c_high - c_low)
    return None  # Out of defined range


def categorize_aqi(aqi_value: float) -> str | None:
    """Map AQI numeric value to CPCB category string."""
    if pd.isna(aqi_value):
        return None
    for category, (low, high) in AQI_CATEGORIES.items():
        if low <= aqi_value <= high:
            return category
    return "Severe" if aqi_value > 500 else None


def compute_aqi(row: pd.Series) -> tuple:
    """
    Compute AQI from pollutant concentrations in a DataFrame row.

    Returns:
        (aqi_value, aqi_bucket, dominant_pollutant) or (None, None, None)
    """
    from config import AQI_POLLUTANTS

    sub_indices = {}
    for pollutant in AQI_POLLUTANTS:
        if pollutant in row and pd.notna(row[pollutant]):
            si = compute_sub_index(pollutant, row[pollutant])
            if si is not None:
                sub_indices[pollutant] = si

    # CPCB rule: need >= 3 pollutants, one must be PM2.5 or PM10
    pm_present = any(p in sub_indices for p in ["PM2.5", "PM10"])
    if len(sub_indices) < 3 or not pm_present:
        return None, None, None

    dominant = max(sub_indices, key=sub_indices.get)
    aqi_value = sub_indices[dominant]
    aqi_bucket = categorize_aqi(aqi_value)

    return round(aqi_value, 2), aqi_bucket, dominant


def compute_aqi_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add computed AQI columns to a DataFrame."""
    results = df.apply(compute_aqi, axis=1, result_type="expand")
    results.columns = ["AQI_Calculated", "AQI_Bucket_Calculated", "Dominant_Pollutant"]
    return pd.concat([df, results], axis=1)
