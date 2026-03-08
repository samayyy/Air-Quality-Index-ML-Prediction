"""
AQI Forecaster: combines WAQI forecast data, Open-Meteo weather forecasts,
and trained ML models to produce 1-7 day AQI predictions.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import FORECAST_DAYS, AQI_POLLUTANTS
from src.models.predict import AQIPredictor
from src.data.api_client import (
    fetch_city_realtime, fetch_city_history, get_city_medians, compute_epa_aqi,
)
from src.data.weather_client import fetch_weather_forecast, get_weather_adjustment_factors

# Maximum allowed day-to-day AQI change (30% cap prevents wild spikes)
MAX_DAY_CHANGE_RATIO = 0.30


class AQIForecaster:
    """
    Produces 1-7 day AQI forecasts by combining:
    - WAQI pollutant forecasts (pm25, pm10, o3)
    - Open-Meteo weather forecasts
    - Historical city medians for missing pollutants
    - Weather-based adjustment factors
    - Trained ML models (LightGBM regression + XGBoost classification)
    """

    WAQI_POLLUTANT_MAP = {
        "pm25": "PM2.5",
        "pm10": "PM10",
        "o3": "O3",
    }

    def __init__(self):
        self.predictor = AQIPredictor()
        self._city_medians = None

    @property
    def city_medians(self):
        if self._city_medians is None:
            self._city_medians = get_city_medians()
        return self._city_medians

    def forecast(self, city: str, days: int = FORECAST_DAYS,
                 token: str = None) -> list[dict]:
        """
        Generate AQI forecast for a city for the next 1-7 days.

        Returns list of daily forecast dicts with predicted_aqi, category,
        pollutants, weather, confidence, dominant_pollutant.
        """
        days = min(days, 7)

        # Fetch data sources
        waqi_forecast = self._get_waqi_forecast(city, token)
        weather_forecast = fetch_weather_forecast(city, days)
        current = fetch_city_realtime(city, token)
        medians = self.city_medians.get(city, {})

        # Build per-day raw predictions
        raw_forecasts = []
        for day_offset in range(1, days + 1):
            target_date = datetime.now().date() + timedelta(days=day_offset)
            date_str = target_date.isoformat()

            # 1. Get WAQI pollutant forecasts for this day
            pollutants = self._get_waqi_day_pollutants(waqi_forecast, date_str)
            has_waqi = len(pollutants) > 0

            # 2. Get weather for this day
            weather = self._get_weather_day(weather_forecast, target_date)

            # 3. Fill missing pollutants (medians preferred over current to avoid anomaly propagation)
            adjustment = get_weather_adjustment_factors(weather) if weather else {
                "pm_factor": 1.0, "gas_factor": 1.0
            }
            pollutants = self._fill_missing_pollutants(
                pollutants, medians, current, adjustment
            )

            # 4. Run through ML predictor + EPA AQI
            prediction = self.predictor.predict_from_pollutants(pollutants)
            epa_aqi, epa_dominant = compute_epa_aqi(pollutants)

            # 5. Compute confidence
            confidence = self._compute_confidence(day_offset, has_waqi, pollutants)

            # Use EPA AQI as primary (matches what users see on Google/AQICN)
            primary_aqi = epa_aqi or prediction.get("cpcb_aqi")

            raw_forecasts.append({
                "date": target_date,
                "predicted_aqi": primary_aqi,
                "epa_aqi": epa_aqi,
                "cpcb_aqi": prediction.get("cpcb_aqi"),
                "ml_aqi": prediction.get("ml_aqi"),
                "category": prediction.get("ml_category") or prediction.get("cpcb_category"),
                "pollutants": pollutants,
                "weather": weather or {},
                "confidence": confidence,
                "dominant_pollutant": epa_dominant or prediction.get("cpcb_dominant"),
            })

        # 6. Smooth forecast — dampen unrealistic day-to-day jumps
        smoothed = self._smooth_forecast(raw_forecasts, current)
        return smoothed

    def _smooth_forecast(self, forecasts: list[dict],
                         current: dict | None) -> list[dict]:
        """
        Dampen unrealistic day-to-day AQI jumps.
        Uses the current AQI as anchor for day 1, then caps change between
        consecutive days to MAX_DAY_CHANGE_RATIO.
        """
        if not forecasts:
            return forecasts

        # Use current WAQI AQI as the anchor point
        anchor_aqi = None
        if current and current.get("aqi"):
            try:
                anchor_aqi = float(current["aqi"])
            except (ValueError, TypeError):
                pass

        # If we have a current anchor, blend day 1 toward it
        if anchor_aqi and forecasts[0].get("predicted_aqi"):
            raw_day1 = forecasts[0]["predicted_aqi"]
            # Weighted blend: 40% current anchor + 60% model prediction
            # This prevents huge jumps from "today" to "tomorrow"
            blended = anchor_aqi * 0.4 + raw_day1 * 0.6
            forecasts[0]["predicted_aqi"] = round(blended, 2)

        # Cap day-to-day changes
        for i in range(1, len(forecasts)):
            prev_aqi = forecasts[i - 1].get("predicted_aqi")
            curr_aqi = forecasts[i].get("predicted_aqi")

            if prev_aqi and curr_aqi and prev_aqi > 0:
                change_ratio = (curr_aqi - prev_aqi) / prev_aqi
                if abs(change_ratio) > MAX_DAY_CHANGE_RATIO:
                    # Dampen: allow at most MAX_DAY_CHANGE_RATIO change
                    direction = 1 if change_ratio > 0 else -1
                    dampened = prev_aqi * (1 + direction * MAX_DAY_CHANGE_RATIO)
                    forecasts[i]["predicted_aqi"] = round(dampened, 2)
                    # Reduce confidence for dampened predictions
                    forecasts[i]["confidence"] = max(
                        0.2, forecasts[i]["confidence"] - 0.15
                    )

        # Recompute categories after smoothing
        from src.data.aqi_calculator import categorize_aqi
        for f in forecasts:
            aqi_val = f.get("predicted_aqi")
            if aqi_val:
                f["category"] = categorize_aqi(aqi_val)

        return forecasts

    def _get_waqi_forecast(self, city: str, token: str = None) -> dict:
        """
        Fetch WAQI forecast data and organize by date and pollutant.
        Returns: {date_str: {pollutant: avg_value}}
        """
        df = fetch_city_history(city, token)
        if df is None or df.empty:
            return {}

        result = {}
        for _, row in df.iterrows():
            date_str = str(row.get("date", ""))
            pollutant_key = row.get("pollutant", "")
            our_name = self.WAQI_POLLUTANT_MAP.get(pollutant_key)
            if not our_name or not date_str:
                continue

            if date_str not in result:
                result[date_str] = {}
            avg_val = row.get("avg")
            if avg_val is not None and not (isinstance(avg_val, float) and np.isnan(avg_val)):
                result[date_str][our_name] = float(avg_val)

        return result

    def _get_waqi_day_pollutants(self, waqi_forecast: dict,
                                  date_str: str) -> dict:
        """Get WAQI forecast pollutants for a specific date."""
        return dict(waqi_forecast.get(date_str, {}))

    def _get_weather_day(self, weather_df: pd.DataFrame | None,
                         target_date) -> dict | None:
        """Extract weather data for a specific date from forecast DataFrame."""
        if weather_df is None or weather_df.empty:
            return None

        target = pd.Timestamp(target_date)
        day_row = weather_df[weather_df["date"].dt.date == target.date()]
        if day_row.empty:
            return None

        row = day_row.iloc[0]
        return {
            "temp_max": row.get("temp_max"),
            "temp_min": row.get("temp_min"),
            "humidity": row.get("humidity"),
            "wind_speed": row.get("wind_speed"),
            "precipitation": row.get("precipitation"),
            "pressure": row.get("pressure"),
        }

    def _fill_missing_pollutants(self, pollutants: dict, medians: dict,
                                  current: dict | None,
                                  adjustment: dict) -> dict:
        """
        Fill missing pollutant values using a weighted blend of:
        - Historical city medians (stable baseline, preferred)
        - Current readings (only as minor supplement)
        Then apply weather adjustment factors.

        Uses 70% median + 30% current to avoid propagating anomalous
        current readings into the forecast.
        """
        pm_factor = adjustment.get("pm_factor", 1.0)
        gas_factor = adjustment.get("gas_factor", 1.0)

        current_pollutants = {}
        if current and current.get("pollutants"):
            current_pollutants = current["pollutants"]

        for p in AQI_POLLUTANTS:
            if p in pollutants and pollutants[p] is not None:
                # Apply weather adjustment to existing WAQI forecast values
                factor = pm_factor if p in ("PM2.5", "PM10") else gas_factor
                pollutants[p] = round(pollutants[p] * factor, 2)
            else:
                # Blend median (stable) + current (recent) for missing pollutants
                median_val = medians.get(p)
                current_val = current_pollutants.get(p)

                # Clean up NaN values
                if median_val is not None and isinstance(median_val, float) and np.isnan(median_val):
                    median_val = None
                if current_val is not None and isinstance(current_val, float) and np.isnan(current_val):
                    current_val = None

                if median_val is not None and current_val is not None:
                    # Weighted blend: 70% median + 30% current
                    base = float(median_val) * 0.7 + float(current_val) * 0.3
                elif median_val is not None:
                    base = float(median_val)
                elif current_val is not None:
                    base = float(current_val)
                else:
                    continue

                factor = pm_factor if p in ("PM2.5", "PM10") else gas_factor
                pollutants[p] = round(base * factor, 2)

        return pollutants

    def _compute_confidence(self, day_offset: int, has_waqi: bool,
                            pollutants: dict) -> float:
        """
        Compute confidence score for a forecast day.
        Decreases with horizon and missing data.
        """
        # Base confidence decreases with forecast horizon
        base = max(0.3, 1.0 - (day_offset - 1) * 0.1)

        # Bonus for having WAQI forecast data for this day
        if has_waqi:
            base += 0.05

        # Penalty for many missing pollutants
        total_available = sum(1 for p in AQI_POLLUTANTS if p in pollutants and pollutants[p])
        if total_available < 4:
            base -= 0.1

        confidence = min(0.95, max(0.2, base))
        return round(confidence, 2)
