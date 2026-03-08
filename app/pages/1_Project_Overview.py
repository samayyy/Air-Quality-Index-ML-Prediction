"""Project Overview page."""
import streamlit as st
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from app.components.sidebar import render_sidebar

st.set_page_config(page_title="Project Overview", page_icon="📋", layout="wide")
render_sidebar()

st.title("Project Overview")

st.markdown("""
## Objective
Predict India's Air Quality Index (AQI) using machine learning, trained on CPCB-verified
pollutant measurements from 26 cities across India (2015-2020).

### Two Prediction Tasks
1. **Regression**: Predict the exact AQI numeric value
2. **Classification**: Predict the AQI category (Good → Severe)

---

## Dataset
| Property | Value |
|----------|-------|
| **Source** | CPCB (Central Pollution Control Board) via Kaggle |
| **Period** | January 2015 – July 2020 |
| **Cities** | 26 major Indian cities |
| **Raw Rows** | 29,531 daily observations |
| **Pollutants** | PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene |
| **AQI Pollutants** | 7 (PM2.5, PM10, NO2, SO2, CO, O3, NH3) |

---

## CPCB AQI Calculation
The AQI is computed using **piecewise linear interpolation** within CPCB-defined breakpoint
ranges for each pollutant. The final AQI = maximum sub-index across all pollutants.

**Requirements**: At least 3 pollutants available, including PM2.5 or PM10.

---

## Pipeline Overview
1. **Data Collection**: Kaggle dataset download
2. **EDA**: Missing value analysis, distributions, correlations, temporal trends
3. **Preprocessing**: 3-tier imputation (interpolation → KNN → drop sparse), winsorization
4. **Feature Engineering**: 146 features (temporal, lag, rolling, ratios, city encoding)
5. **Regression Models**: 8 models compared, LightGBM best (R²=0.902)
6. **Classification Models**: 7 models with SMOTE, XGBoost best (Accuracy=84.1%)
7. **Deep Learning**: 5 architectures (LSTM, GRU variants) on raw sequences
8. **Hyperparameter Tuning**: Optuna Bayesian optimization (50 trials)
9. **Explainability**: SHAP analysis for feature importance
10. **Real-Time**: WAQI API integration for live predictions

---

## Technology Stack
- **Languages**: Python 3.11
- **ML**: scikit-learn, XGBoost, LightGBM, CatBoost
- **Deep Learning**: TensorFlow / Keras
- **Tuning**: Optuna
- **Explainability**: SHAP
- **Web App**: Streamlit
- **Data**: pandas, NumPy, PyArrow
- **Visualization**: Matplotlib, Seaborn, Plotly
""")
