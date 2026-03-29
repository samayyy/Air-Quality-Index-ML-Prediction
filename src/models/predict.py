"""
Unified prediction interface for AQI models.
Loads best models and provides consistent prediction API.
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (
    MODELS_DIR, AQI_POLLUTANTS, POLLUTANT_COLS,
    AQI_CATEGORIES, INDIA_SEASONS, LAG_DAYS, ROLLING_WINDOWS,
)
from src.data.aqi_calculator import compute_sub_index, categorize_aqi


class AQIPredictor:
    """Unified predictor that loads best regression and classification models."""

    def __init__(self):
        self.reg_model = None
        self.cls_model = None
        self.scaler = None
        self.feature_cols = None
        self._loaded = False

    def load_models(self):
        """Load the best regression, classification, and optionally tuned models."""
        reg_dir = MODELS_DIR / "regression"
        cls_dir = MODELS_DIR / "classification"
        tuned_dir = MODELS_DIR / "tuned"

        # Try tuned first, fall back to default
        if (tuned_dir / "best_regression.joblib").exists():
            self.reg_model = joblib.load(tuned_dir / "best_regression.joblib")
            self.reg_model_name = "Tuned LightGBM (Optuna, R²=0.9065)"
        elif (reg_dir / "lightgbm.joblib").exists():
            self.reg_model = joblib.load(reg_dir / "lightgbm.joblib")
            self.reg_model_name = "LightGBM"

        if (tuned_dir / "best_classification.joblib").exists():
            self.cls_model = joblib.load(tuned_dir / "best_classification.joblib")
            self.cls_model_name = "Tuned XGBoost (Optuna, Acc=84.16%)"
        elif (cls_dir / "xgboost.joblib").exists():
            self.cls_model = joblib.load(cls_dir / "xgboost.joblib")
            self.cls_model_name = "XGBoost"

        if (reg_dir / "scaler.joblib").exists():
            self.scaler = joblib.load(reg_dir / "scaler.joblib")

        self._loaded = True

    @staticmethod
    def _cpcb_to_epa(cpcb_aqi: float, dominant_pollutant: str) -> float | None:
        """
        Convert a CPCB-scale AQI to EPA-scale AQI via the dominant pollutant.

        Steps:
        1. Reverse CPCB AQI → concentration (using CPCB breakpoints)
        2. Forward concentration → EPA AQI (using EPA breakpoints)

        This gives an accurate cross-scale conversion because both scales
        are piecewise linear functions of the same underlying concentration.
        """
        from config import AQI_BREAKPOINTS as CPCB_BP
        from src.data.api_client import concentration_to_epa_aqi

        if dominant_pollutant not in CPCB_BP:
            return None

        # Step 1: Reverse CPCB AQI → concentration
        breakpoints = CPCB_BP[dominant_pollutant]
        concentration = None
        for c_low, c_high, i_low, i_high in breakpoints:
            if i_low <= cpcb_aqi <= i_high:
                concentration = (cpcb_aqi - i_low) / (i_high - i_low) * (c_high - c_low) + c_low
                break

        if concentration is None:
            # Above max breakpoint — extrapolate from last range
            c_low, c_high, i_low, i_high = breakpoints[-1]
            if cpcb_aqi > i_high:
                concentration = (cpcb_aqi - i_low) / (i_high - i_low) * (c_high - c_low) + c_low

        if concentration is None:
            return None

        # Step 2: Forward concentration → EPA AQI
        # Note: CPCB concentrations are in μg/m³ (PM, gases) or mg/m³ (CO)
        # EPA breakpoints for gases use ppb/ppm, but concentration_to_epa_aqi()
        # expects μg/m³ input and handles the conversion internally
        epa_aqi = concentration_to_epa_aqi(dominant_pollutant, concentration)
        return round(epa_aqi, 1) if epa_aqi else None

    def _compute_cpcb_aqi(self, pollutants: dict) -> tuple:
        """Compute AQI using CPCB formula from pollutant dict."""
        sub_indices = {}
        for p in AQI_POLLUTANTS:
            if p in pollutants and pollutants[p] is not None and not np.isnan(pollutants[p]):
                si = compute_sub_index(p, pollutants[p])
                if si is not None:
                    sub_indices[p] = si

        pm_present = any(p in sub_indices for p in ["PM2.5", "PM10"])
        if len(sub_indices) < 3 or not pm_present:
            return None, None, None

        dominant = max(sub_indices, key=sub_indices.get)
        aqi_val = round(sub_indices[dominant], 2)
        category = categorize_aqi(aqi_val)
        return aqi_val, category, dominant

    def predict_from_pollutants(self, pollutants: dict,
                                city_medians: dict = None) -> dict:
        """
        Predict AQI from a dict of pollutant concentrations.

        Args:
            pollutants: e.g. {"PM2.5": 85, "PM10": 120, "NO2": 40, ...}
            city_medians: optional historical medians for the city (improves
                          lag/rolling feature estimation for ML models)

        Returns:
            dict with cpcb_aqi, ml_regression, ml_classification, etc.
        """
        if not self._loaded:
            self.load_models()

        result = {
            "pollutants": pollutants,
            "cpcb_aqi": None,
            "cpcb_category": None,
            "cpcb_dominant": None,
            "epa_aqi": None,
            "epa_dominant": None,
            "ml_aqi": None,
            "ml_category": None,
            "model_agreement": None,
        }

        # CPCB formula
        aqi_val, category, dominant = self._compute_cpcb_aqi(pollutants)
        result["cpcb_aqi"] = aqi_val
        result["cpcb_category"] = category
        result["cpcb_dominant"] = dominant

        # US-EPA AQI (matches WAQI/AQICN/Google scale)
        from src.data.api_client import compute_epa_aqi
        epa_aqi, epa_dominant = compute_epa_aqi(pollutants)
        result["epa_aqi"] = epa_aqi
        result["epa_dominant"] = epa_dominant

        # ML predictions need a feature vector
        ml_aqi_cpcb = None
        if self.reg_model is not None:
            try:
                features = self._build_feature_vector(pollutants, city_medians)
                if features is not None:
                    pred = self.reg_model.predict(features.reshape(1, -1))[0]
                    ml_aqi_cpcb = round(float(pred), 2)
                    result["ml_aqi"] = ml_aqi_cpcb
            except Exception as e:
                result["ml_error"] = str(e)

        if self.cls_model is not None:
            try:
                features = self._build_feature_vector(pollutants, city_medians)
                if features is not None:
                    pred = self.cls_model.predict(features.reshape(1, -1))[0]
                    bucket_order = list(AQI_CATEGORIES.keys())
                    result["ml_category"] = bucket_order[int(pred)] if isinstance(pred, (int, np.integer)) else str(pred)
            except Exception as e:
                result["cls_error"] = str(e)

        # Convert ML output from CPCB scale → EPA scale
        # Use the EPA dominant pollutant (not CPCB dominant) for conversion,
        # since we want to match the EPA scale. Fall back to CPCB dominant.
        if ml_aqi_cpcb:
            # Prefer EPA dominant (what WAQI uses), then CPCB dominant
            conv_pollutant = result.get("epa_dominant") or result.get("cpcb_dominant")
            if conv_pollutant:
                ml_epa = self._cpcb_to_epa(ml_aqi_cpcb, conv_pollutant)
                if ml_epa is not None:
                    result["ml_aqi_epa"] = ml_epa

        # Model agreement
        if result["cpcb_category"] and result["ml_category"]:
            result["model_agreement"] = result["cpcb_category"] == result["ml_category"]

        return result

    def _build_feature_vector(self, pollutants: dict,
                               city_medians: dict = None) -> np.ndarray | None:
        """
        Build a feature vector from pollutant values.

        For lag and rolling features, uses current values as proxy for
        short-term lags (1-3 days) and blends toward city medians for
        longer lags (7-30 days) when available. This prevents the model
        from seeing unrealistic "30-day rolling mean" values that exactly
        equal the current reading.
        """
        # Start with raw pollutant values
        raw = []
        for col in POLLUTANT_COLS:
            raw.append(pollutants.get(col, 0.0) or 0.0)

        # Temporal features (current time)
        from datetime import datetime
        now = datetime.now()
        temporal = [
            now.year, now.month, now.weekday(), now.timetuple().tm_yday,
            (now.month - 1) // 3 + 1, int(now.weekday() >= 5),
        ]

        # City encoded (default 0)
        city_encoded = [0]

        # Lag features: short lags use current; longer lags blend with median
        # LAG_DAYS = [1, 2, 3, 7, 14, 30]
        lag_features = []
        for col in POLLUTANT_COLS:
            val = pollutants.get(col, 0.0) or 0.0
            median = None
            if city_medians:
                m = city_medians.get(col)
                if m is not None and not (isinstance(m, float) and np.isnan(m)):
                    median = float(m)
            if median is None:
                median = val  # fallback: use current for all lags
            for lag in LAG_DAYS:
                if lag <= 3:
                    lag_features.append(val)  # recent: trust current
                else:
                    # Longer lags: blend toward median
                    lag_features.append(val * 0.5 + median * 0.5)

        # Rolling features: short windows use current; longer blend with median
        rolling_cols = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
        rolling_features = []
        for col in rolling_cols:
            val = pollutants.get(col, 0.0) or 0.0
            median = None
            if city_medians:
                m = city_medians.get(col)
                if m is not None and not (isinstance(m, float) and np.isnan(m)):
                    median = float(m)
            if median is None:
                median = val
            for window in ROLLING_WINDOWS:
                if window <= 7:
                    rolling_features.append(val)  # mean
                else:
                    rolling_features.append(val * 0.5 + median * 0.5)
                # Std: small positive value (not zero — zero std is unrealistic)
                rolling_features.append(abs(val - median) * 0.3 if val != median else val * 0.1)

        # Ratios
        pm25 = pollutants.get("PM2.5", 0.0) or 0.0
        pm10 = pollutants.get("PM10", 1.0) or 1.0
        no2 = pollutants.get("NO2", 0.0) or 0.0
        no = pollutants.get("NO", 1.0) or 1.0
        ratios = [pm25 / pm10 if pm10 > 0 else 0, no2 / no if no > 0 else 0]

        features = np.array(
            raw + temporal + lag_features + rolling_features + ratios + city_encoded,
            dtype=np.float64
        )
        features = np.nan_to_num(features, nan=0.0)
        return features

    def predict_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict AQI for each row of a DataFrame with pollutant columns.
        Adds prediction columns to the DataFrame.
        """
        if not self._loaded:
            self.load_models()

        results = []
        for _, row in df.iterrows():
            pollutants = {col: row.get(col) for col in AQI_POLLUTANTS if col in row.index}
            pred = self.predict_from_pollutants(pollutants)
            results.append(pred)

        pred_df = pd.DataFrame(results)
        for col in ["cpcb_aqi", "cpcb_category", "ml_aqi", "ml_category", "model_agreement"]:
            if col in pred_df.columns:
                df[col] = pred_df[col].values
        return df
