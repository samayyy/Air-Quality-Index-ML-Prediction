"""
Cache manager for real-time AQI data.
Stores recent readings for LSTM sliding window and reduces API calls.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import DATA_EXTERNAL


CACHE_FILE = DATA_EXTERNAL / "realtime_cache.parquet"
MAX_AGE_DAYS = 60


def load_cache() -> pd.DataFrame:
    """Load cached readings from parquet file."""
    if CACHE_FILE.exists():
        df = pd.read_parquet(CACHE_FILE)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    return pd.DataFrame()


def save_cache(df: pd.DataFrame):
    """Save cache to parquet, creating directory if needed."""
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE_FILE, index=False)


def add_reading(city: str, pollutants: dict, aqi: float = None,
                timestamp: datetime = None):
    """
    Add a new reading to the cache.

    Args:
        city: City name
        pollutants: Dict of pollutant name -> value
        aqi: AQI value (from API or calculated)
        timestamp: Reading time (defaults to now)
    """
    if timestamp is None:
        timestamp = datetime.now()

    row = {"city": city, "timestamp": timestamp, "aqi": aqi}
    row.update(pollutants)

    cache = load_cache()
    new_row = pd.DataFrame([row])
    cache = pd.concat([cache, new_row], ignore_index=True)

    # Deduplicate: keep latest reading per city per day
    cache["date"] = pd.to_datetime(cache["timestamp"]).dt.date
    cache = cache.sort_values("timestamp").drop_duplicates(
        subset=["city", "date"], keep="last"
    )
    cache = cache.drop(columns=["date"])

    save_cache(cache)


def get_city_history(city: str, days: int = 14) -> pd.DataFrame:
    """
    Get recent cached readings for a city.
    Returns DataFrame sorted by timestamp, last `days` readings.
    """
    cache = load_cache()
    if cache.empty:
        return pd.DataFrame()

    city_data = cache[cache["city"] == city].copy()
    if city_data.empty:
        return pd.DataFrame()

    cutoff = datetime.now() - timedelta(days=days)
    city_data = city_data[city_data["timestamp"] >= cutoff]
    return city_data.sort_values("timestamp").reset_index(drop=True)


def cleanup_old_entries():
    """Remove entries older than MAX_AGE_DAYS."""
    cache = load_cache()
    if cache.empty:
        return

    cutoff = datetime.now() - timedelta(days=MAX_AGE_DAYS)
    cache = cache[cache["timestamp"] >= cutoff]
    save_cache(cache)


def get_cache_stats() -> dict:
    """Return summary statistics about the cache."""
    cache = load_cache()
    if cache.empty:
        return {"total_readings": 0, "cities": 0, "date_range": None}

    return {
        "total_readings": len(cache),
        "cities": cache["city"].nunique(),
        "city_list": sorted(cache["city"].unique().tolist()),
        "date_range": (
            cache["timestamp"].min().isoformat(),
            cache["timestamp"].max().isoformat(),
        ),
        "readings_per_city": cache["city"].value_counts().to_dict(),
    }
