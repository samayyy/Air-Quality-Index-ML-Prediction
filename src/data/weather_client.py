"""
Open-Meteo weather forecast client for 7-day weather data.
Free API, no key required. Provides weather context for AQI forecasting.
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import OPEN_METEO_BASE_URL, CITY_COORDINATES, FORECAST_DAYS


def fetch_weather_forecast(city: str, days: int = FORECAST_DAYS) -> pd.DataFrame | None:
    """
    Fetch 7-day weather forecast from Open-Meteo for a given city.

    Args:
        city: City name (must be in CITY_COORDINATES)
        days: Number of forecast days (1-7)

    Returns:
        DataFrame with columns: date, temp_max, temp_min, humidity, wind_speed,
        precipitation, pressure. Returns None on failure.
    """
    coords = CITY_COORDINATES.get(city)
    if coords is None:
        print(f"No coordinates found for city: {city}")
        return None

    lat, lon = coords

    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ",".join([
            "temperature_2m_max",
            "temperature_2m_min",
            "relative_humidity_2m_mean",
            "wind_speed_10m_max",
            "precipitation_sum",
            "surface_pressure_mean",
        ]),
        "timezone": "Asia/Kolkata",
        "forecast_days": min(days, 7),
    }

    try:
        resp = requests.get(OPEN_METEO_BASE_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        daily = data.get("daily", {})
        if not daily or not daily.get("time"):
            return None

        df = pd.DataFrame({
            "date": pd.to_datetime(daily["time"]),
            "temp_max": daily.get("temperature_2m_max"),
            "temp_min": daily.get("temperature_2m_min"),
            "humidity": daily.get("relative_humidity_2m_mean"),
            "wind_speed": daily.get("wind_speed_10m_max"),
            "precipitation": daily.get("precipitation_sum"),
            "pressure": daily.get("surface_pressure_mean"),
        })

        return df

    except requests.RequestException as e:
        print(f"Weather API request failed for {city}: {e}")
        return None


def get_weather_adjustment_factors(weather_row: dict) -> dict:
    """
    Compute weather-based pollutant adjustment multipliers.

    Logic based on atmospheric science:
    - High humidity + low wind → pollutants trapped (higher PM)
    - Rain → washes out particulates (lower PM)
    - Temperature inversions (low temp_min) → trapping layer
    - Strong wind → disperses pollutants

    Returns dict of multiplier factors for PM, gaseous pollutants.
    """
    pm_factor = 1.0
    gas_factor = 1.0

    humidity = weather_row.get("humidity", 50)
    wind = weather_row.get("wind_speed", 10)
    precip = weather_row.get("precipitation", 0)
    temp_min = weather_row.get("temp_min", 15)

    # High humidity traps pollutants
    if humidity and humidity > 80:
        pm_factor *= 1.15
        gas_factor *= 1.05
    elif humidity and humidity < 40:
        pm_factor *= 0.9

    # Low wind = stagnation
    if wind is not None and wind < 5:
        pm_factor *= 1.2
        gas_factor *= 1.1
    elif wind is not None and wind > 20:
        pm_factor *= 0.75
        gas_factor *= 0.85

    # Rain washes out particulates
    if precip is not None and precip > 5:
        pm_factor *= 0.7
        gas_factor *= 0.9
    elif precip is not None and precip > 1:
        pm_factor *= 0.85

    # Cold temperature inversions trap pollutants
    if temp_min is not None and temp_min < 8:
        pm_factor *= 1.1
        gas_factor *= 1.05

    return {
        "pm_factor": round(pm_factor, 3),
        "gas_factor": round(gas_factor, 3),
    }
