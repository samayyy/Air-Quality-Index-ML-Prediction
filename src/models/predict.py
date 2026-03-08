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

    def predict_from_pollutants(self, pollutants: dict) -> dict:
        """
        Predict AQI from a dict of pollutant concentrations.

        Args:
            pollutants: e.g. {"PM2.5": 85, "PM10": 120, "NO2": 40, ...}

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
        # For single-point prediction, we create a minimal feature vector
        # with zeros for lag/rolling features (best effort)
        if self.reg_model is not None:
            try:
                features = self._build_feature_vector(pollutants)
                if features is not None:
                    pred = self.reg_model.predict(features.reshape(1, -1))[0]
                    result["ml_aqi"] = round(float(pred), 2)
            except Exception as e:
                result["ml_error"] = str(e)

        if self.cls_model is not None:
            try:
                features = self._build_feature_vector(pollutants)
                if features is not None:
                    pred = self.cls_model.predict(features.reshape(1, -1))[0]
                    bucket_order = list(AQI_CATEGORIES.keys())
                    result["ml_category"] = bucket_order[int(pred)] if isinstance(pred, (int, np.integer)) else str(pred)
            except Exception as e:
                result["cls_error"] = str(e)

        # Model agreement
        if result["cpcb_category"] and result["ml_category"]:
            result["model_agreement"] = result["cpcb_category"] == result["ml_category"]

        return result

    def _build_feature_vector(self, pollutants: dict) -> np.ndarray | None:
        """
        Build a feature vector from pollutant values.
        Uses zeros for lag/rolling features (single-point prediction).
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

        # Lag features: use current pollutant values as proxy
        lag_features = []
        for col in POLLUTANT_COLS:
            val = pollutants.get(col, 0.0) or 0.0
            for _ in LAG_DAYS:
                lag_features.append(val)

        # Rolling features: use current value as proxy
        rolling_cols = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
        rolling_features = []
        for col in rolling_cols:
            val = pollutants.get(col, 0.0) or 0.0
            for _ in ROLLING_WINDOWS:
                rolling_features.append(val)  # mean
                rolling_features.append(0.0)  # std

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
