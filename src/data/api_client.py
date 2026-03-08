"""
WAQI (World Air Quality Index) API client for real-time AQI data.
Fetches current pollutant readings for Indian cities.
"""
import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import WAQI_BASE_URL, WAQI_CITY_MAP, AQI_POLLUTANTS


# === US-EPA AQI Breakpoints (for reverse: AQI sub-index → concentration) ===
# Format: (C_low, C_high, I_low, I_high)
# WAQI returns AQI sub-indices; our models need raw concentrations.
EPA_BREAKPOINTS = {
    "PM2.5": [  # μg/m³ (24-hr)
        (0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400), (350.5, 500.4, 401, 500),
    ],
    "PM10": [  # μg/m³ (24-hr)
        (0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150),
        (255, 354, 151, 200), (355, 424, 201, 300),
        (425, 504, 301, 400), (505, 604, 401, 500),
    ],
    "O3": [  # ppb (8-hr) → convert to μg/m³ (* 2.0)
        (0, 54, 0, 50), (55, 70, 51, 100), (71, 85, 101, 150),
        (86, 105, 151, 200), (106, 200, 201, 300),
    ],
    "NO2": [  # ppb → convert to μg/m³ (* 1.88)
        (0, 53, 0, 50), (54, 100, 51, 100), (101, 360, 101, 150),
        (361, 649, 151, 200), (650, 1249, 201, 300),
        (1250, 1649, 301, 400), (1650, 2049, 401, 500),
    ],
    "SO2": [  # ppb → convert to μg/m³ (* 2.62)
        (0, 35, 0, 50), (36, 75, 51, 100), (76, 185, 101, 150),
        (186, 304, 151, 200), (305, 604, 201, 300),
        (605, 804, 301, 400), (805, 1004, 401, 500),
    ],
    "CO": [  # ppm → convert to mg/m³ (* 1.145)
        (0, 4.4, 0, 50), (4.5, 9.4, 51, 100), (9.5, 12.4, 101, 150),
        (12.5, 15.4, 151, 200), (15.5, 30.4, 201, 300),
        (30.5, 40.4, 301, 400), (40.5, 50.4, 401, 500),
    ],
}

# ppb/ppm → μg/m³ or mg/m³ conversion factors (at 25°C, 1 atm)
_UNIT_CONVERSION = {
    "O3": 2.0,      # ppb → μg/m³
    "NO2": 1.88,    # ppb → μg/m³
    "SO2": 2.62,    # ppb → μg/m³
    "CO": 1.145,    # ppm → mg/m³
}


def aqi_to_concentration(pollutant: str, aqi_value: float) -> float | None:
    """
    Reverse-calculate raw concentration from a US-EPA AQI sub-index value.

    WAQI API returns AQI sub-indices in iaqi, but our CPCB formula and ML
    models expect raw concentrations (μg/m³ for PM/gases, mg/m³ for CO).

    Returns concentration in CPCB-compatible units, or None if out of range.
    """
    if pollutant not in EPA_BREAKPOINTS or aqi_value is None:
        return None

    breakpoints = EPA_BREAKPOINTS[pollutant]

    # Find the matching AQI range
    for c_low, c_high, i_low, i_high in breakpoints:
        if i_low <= aqi_value <= i_high:
            # Reverse linear interpolation: C = (I - I_low) / (I_high - I_low) * (C_high - C_low) + C_low
            concentration = (aqi_value - i_low) / (i_high - i_low) * (c_high - c_low) + c_low
            # Convert gaseous pollutant units (ppb/ppm → μg/m³ or mg/m³)
            if pollutant in _UNIT_CONVERSION:
                concentration *= _UNIT_CONVERSION[pollutant]
            return round(concentration, 2)

    # Above max breakpoint — extrapolate from last range
    c_low, c_high, i_low, i_high = breakpoints[-1]
    concentration = (aqi_value - i_low) / (i_high - i_low) * (c_high - c_low) + c_low
    if pollutant in _UNIT_CONVERSION:
        concentration *= _UNIT_CONVERSION[pollutant]
    return round(concentration, 2)


def concentration_to_epa_aqi(pollutant: str, concentration: float) -> float | None:
    """
    Forward-calculate US-EPA AQI sub-index from raw concentration.

    Concentration units expected: μg/m³ for PM/gases, mg/m³ for CO.
    For gaseous pollutants, converts from μg/m³ back to ppb/ppm before lookup.

    Returns US-EPA AQI sub-index value, or None if out of range.
    """
    if pollutant not in EPA_BREAKPOINTS or concentration is None:
        return None

    # Convert gaseous pollutant units back to EPA units (μg/m³ → ppb/ppm)
    conc_epa = concentration
    if pollutant in _UNIT_CONVERSION:
        conc_epa = concentration / _UNIT_CONVERSION[pollutant]

    breakpoints = EPA_BREAKPOINTS[pollutant]

    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= conc_epa <= c_high:
            aqi = (i_high - i_low) / (c_high - c_low) * (conc_epa - c_low) + i_low
            return round(aqi, 1)

    # Above max breakpoint — extrapolate
    c_low, c_high, i_low, i_high = breakpoints[-1]
    if conc_epa > c_high:
        aqi = (i_high - i_low) / (c_high - c_low) * (conc_epa - c_low) + i_low
        return round(aqi, 1)

    return None


def compute_epa_aqi(pollutants: dict) -> tuple[float | None, str | None]:
    """
    Compute overall US-EPA AQI from a dict of pollutant concentrations.
    Returns (aqi_value, dominant_pollutant) or (None, None).

    This produces values on the same scale as WAQI/AQICN/IQAir/Google,
    so users can directly compare with what they see online.
    """
    sub_indices = {}
    for p in ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]:
        if p in pollutants and pollutants[p] is not None:
            si = concentration_to_epa_aqi(p, pollutants[p])
            if si is not None:
                sub_indices[p] = si

    if not sub_indices:
        return None, None

    dominant = max(sub_indices, key=sub_indices.get)
    return round(sub_indices[dominant], 1), dominant


def get_token() -> str:
    """Get WAQI API token from environment."""
    token = os.environ.get("WAQI_API_TOKEN", "")
    if not token:
        raise ValueError(
            "WAQI_API_TOKEN not set. Get one at https://aqicn.org/data-platform/token/ "
            "and add it to your .env file."
        )
    return token


def fetch_city_realtime(city: str, token: str = None) -> dict | None:
    """
    Fetch real-time AQI data for a city from WAQI API.

    Returns dict with keys: city, aqi, time, pollutants (dict of readings),
    or None on failure.
    """
    if token is None:
        token = get_token()

    search_term = WAQI_CITY_MAP.get(city, city.lower())
    url = f"{WAQI_BASE_URL}/feed/{search_term}/?token={token}"

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") != "ok":
            print(f"WAQI API error for {city}: {data.get('data', 'Unknown error')}")
            return None

        feed = data["data"]
        iaqi = feed.get("iaqi", {})

        # Map WAQI pollutant keys to our column names
        pollutant_map = {
            "pm25": "PM2.5",
            "pm10": "PM10",
            "no2": "NO2",
            "so2": "SO2",
            "co": "CO",
            "o3": "O3",
        }

        pollutants = {}
        pollutants_raw_aqi = {}  # Store original sub-index values too
        for waqi_key, our_key in pollutant_map.items():
            if waqi_key in iaqi:
                aqi_val = iaqi[waqi_key].get("v")
                pollutants_raw_aqi[our_key] = aqi_val
                # Convert AQI sub-index → raw concentration
                conc = aqi_to_concentration(our_key, aqi_val)
                pollutants[our_key] = conc if conc is not None else aqi_val

        # NH3 is rarely available from WAQI (no EPA breakpoint, keep as-is)
        if "nh3" in iaqi:
            pollutants["NH3"] = iaqi["nh3"].get("v")
            pollutants_raw_aqi["NH3"] = iaqi["nh3"].get("v")

        return {
            "city": city,
            "aqi": feed.get("aqi"),
            "dominant_pollutant": feed.get("dominentpol"),
            "time": feed.get("time", {}).get("iso"),
            "station": feed.get("city", {}).get("name"),
            "pollutants": pollutants,
            "pollutants_aqi": pollutants_raw_aqi,
        }

    except requests.RequestException as e:
        print(f"Request failed for {city}: {e}")
        return None


def fetch_all_cities(token: str = None, delay: float = 0.3) -> pd.DataFrame:
    """
    Fetch real-time data for all CPCB cities.
    Returns DataFrame with one row per city.
    """
    if token is None:
        token = get_token()

    records = []
    for city in WAQI_CITY_MAP:
        result = fetch_city_realtime(city, token)
        if result:
            row = {"City": result["city"], "WAQI_AQI": result["aqi"],
                   "Station": result["station"], "Time": result["time"],
                   "Dominant_Pollutant": result["dominant_pollutant"]}
            row.update(result["pollutants"])
            records.append(row)
        time.sleep(delay)  # Rate limiting

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    return df


def fetch_city_history(city: str, token: str = None) -> pd.DataFrame | None:
    """
    Fetch recent forecast/history data for a city (limited by WAQI free tier).
    Returns DataFrame or None.
    """
    if token is None:
        token = get_token()

    search_term = WAQI_CITY_MAP.get(city, city.lower())
    url = f"{WAQI_BASE_URL}/feed/{search_term}/?token={token}"

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") != "ok":
            return None

        feed = data["data"]
        forecast = feed.get("forecast", {}).get("daily", {})

        if not forecast:
            return None

        # Map WAQI forecast keys to our pollutant names for conversion
        forecast_pollutant_map = {
            "pm25": "PM2.5", "pm10": "PM10", "o3": "O3",
            "no2": "NO2", "so2": "SO2", "co": "CO",
        }

        records = []
        for pollutant, days in forecast.items():
            our_name = forecast_pollutant_map.get(pollutant)
            for day_data in days:
                avg_aqi = day_data.get("avg")
                # Convert forecast AQI sub-indices → concentrations
                avg_conc = aqi_to_concentration(our_name, avg_aqi) if our_name else avg_aqi
                records.append({
                    "date": day_data.get("day"),
                    "pollutant": pollutant,
                    "avg": avg_conc if avg_conc is not None else avg_aqi,
                    "min": day_data.get("min"),
                    "max": day_data.get("max"),
                })

        return pd.DataFrame(records) if records else None

    except requests.RequestException:
        return None


def get_city_medians() -> dict:
    """
    Return historical median pollutant values per city from the training data.
    Used as fallback when API data is incomplete (e.g., NH3 not available).
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from config import DATA_PROCESSED

    parquet_path = DATA_PROCESSED / "city_day_clean.parquet"
    if not parquet_path.exists():
        return {}

    df = pd.read_parquet(parquet_path)
    medians = {}
    for city in df["City"].unique():
        city_data = df[df["City"] == city]
        medians[city] = {}
        for col in AQI_POLLUTANTS:
            if col in city_data.columns:
                medians[city][col] = city_data[col].median()
    return medians
