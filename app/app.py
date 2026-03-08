"""
India AQI Prediction - Streamlit Web Panel
Main entry point with landing page and key metrics.
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_PROCESSED, MODELS_DIR, FIGURES_DIR
from app.components.sidebar import render_sidebar

st.set_page_config(
    page_title="India AQI Predictor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load custom CSS
css_path = Path(__file__).parent / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

render_sidebar()

# === Landing Page ===
st.markdown(
    '<h1 class="page-title">India Air Quality Index Prediction</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    "**ML-powered AQI prediction** using CPCB-verified data from 26 Indian cities (2015-2020). "
    "Explore models, view explanations, and get live predictions."
)

st.divider()

# Key metrics row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Cities Analyzed", "26", help="CPCB monitoring stations across India")
with col2:
    st.metric("Data Points", "29,531", help="Raw observations (2015-2020)")
with col3:
    best_r2 = "0.902"
    try:
        reg = pd.read_csv(DATA_PROCESSED / "regression_results.csv", index_col=0)
        best_r2 = f"{reg['R2'].max():.3f}"
    except FileNotFoundError:
        pass
    st.metric("Best R² Score", best_r2, help="LightGBM regression model")
with col4:
    best_acc = "84.1%"
    try:
        cls = pd.read_csv(DATA_PROCESSED / "classification_results.csv", index_col=0)
        best_acc = f"{cls['Accuracy'].max()*100:.1f}%"
    except FileNotFoundError:
        pass
    st.metric("Best Accuracy", best_acc, help="XGBoost classification model")

st.divider()

# Project highlights
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Project Highlights")
    st.markdown("""
    - **8 Regression Models**: Ridge, Lasso, Decision Tree, Random Forest, SVR, XGBoost, LightGBM, CatBoost
    - **7 Classification Models**: With SMOTE for class imbalance handling
    - **5 Deep Learning Models**: LSTM, Stacked LSTM, BiLSTM, Conv1D-LSTM, GRU
    - **Optuna Tuning**: 50-trial Bayesian optimization for top models
    - **SHAP Explainability**: Feature importance and per-prediction explanations
    - **Real-Time Predictions**: WAQI API integration for live city data
    """)

with col_right:
    st.subheader("AQI Categories (CPCB Standard)")
    aqi_data = pd.DataFrame({
        "Category": ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"],
        "AQI Range": ["0-50", "51-100", "101-200", "201-300", "301-400", "401-500"],
        "Health Impact": [
            "Minimal impact",
            "Minor breathing discomfort",
            "Breathing discomfort for sensitive people",
            "Breathing discomfort for most people",
            "Respiratory illness on prolonged exposure",
            "Severe health effects, avoid outdoor activity",
        ],
    })
    st.dataframe(aqi_data, use_container_width=True, hide_index=True)

st.divider()

# Quick navigation
st.subheader("Explore the Project")
cols = st.columns(3)
with cols[0]:
    st.page_link("pages/1_Project_Overview.py", label="Project Overview", icon="📋")
    st.page_link("pages/2_EDA_Visualizations.py", label="EDA & Visualizations", icon="📊")
with cols[1]:
    st.page_link("pages/3_Model_Comparison.py", label="Model Comparison", icon="🤖")
    st.page_link("pages/4_Live_Prediction.py", label="Live Prediction", icon="🎯")
with cols[2]:
    st.page_link("pages/5_City_Analysis.py", label="City Analysis", icon="🏙️")
    st.page_link("pages/6_Explainability.py", label="Explainability (SHAP)", icon="🔍")
    st.page_link("pages/7_Forecast_Insights.py", label="Forecast & Insights", icon="🔮")
