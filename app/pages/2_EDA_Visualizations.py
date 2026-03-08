"""EDA Visualizations page - displays the 20 existing figures."""
import streamlit as st
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config import FIGURES_DIR
from app.components.sidebar import render_sidebar

st.set_page_config(page_title="EDA & Visualizations", page_icon="📊", layout="wide")
render_sidebar()

st.title("Exploratory Data Analysis & Visualizations")

# Group figures by section
sections = {
    "Missing Value Analysis": [
        ("01_missing_values.png", "Missing value percentages across all columns"),
        ("02_missing_by_city.png", "Missing data patterns by city"),
    ],
    "Pollutant Distributions": [
        ("03_pollutant_distributions.png", "Distribution of key pollutant concentrations"),
        ("04_aqi_distribution.png", "AQI value distribution and category breakdown"),
    ],
    "Correlation Analysis": [
        ("05_correlations.png", "Pearson and Spearman correlation heatmaps"),
    ],
    "Temporal Patterns": [
        ("06_temporal_trends.png", "Yearly, monthly, seasonal, and day-of-week AQI patterns"),
        ("07_top5_cities_timeseries.png", "Monthly AQI time series for 5 most polluted cities"),
    ],
    "City-wise Analysis": [
        ("08_city_mean_aqi.png", "Mean AQI ranking across all 26 cities"),
        ("09_city_month_heatmap.png", "City × Month AQI heatmap"),
    ],
    "Class Imbalance & Outliers": [
        ("10_class_imbalance.png", "AQI category class distribution"),
        ("11_outlier_scatter.png", "Outlier detection across pollutants"),
    ],
    "Preprocessing Results": [
        ("12_winsorization_pm25.png", "PM2.5 before/after winsorization"),
        ("13_aqi_validation.png", "Original vs CPCB-calculated AQI validation"),
        ("14_split_distributions.png", "Train/Val/Test split AQI distributions"),
    ],
}

for section_name, figures in sections.items():
    st.subheader(section_name)
    if len(figures) == 1:
        fig_file, caption = figures[0]
        fig_path = FIGURES_DIR / fig_file
        if fig_path.exists():
            st.image(str(fig_path), caption=caption, use_container_width=True)
        else:
            st.warning(f"Figure not found: {fig_file}")
    else:
        cols = st.columns(len(figures))
        for col, (fig_file, caption) in zip(cols, figures):
            fig_path = FIGURES_DIR / fig_file
            with col:
                if fig_path.exists():
                    st.image(str(fig_path), caption=caption, use_container_width=True)
                else:
                    st.warning(f"Figure not found: {fig_file}")
    st.divider()

# Additional figures from modeling (if they exist)
st.subheader("Model Results Visualizations")
model_figs = [
    ("15_regression_comparison.png", "Regression model comparison (R², RMSE, Training Time)"),
    ("16_predicted_vs_actual.png", "Predicted vs Actual AQI scatter (top 4 models)"),
    ("17_residuals.png", "Residual distributions (top 4 models)"),
    ("18_feature_importance.png", "Top 25 feature importances (best model)"),
    ("19_classification_comparison.png", "Classification model comparison"),
    ("20_confusion_matrices.png", "Confusion matrices (top 3 classifiers)"),
]

for fig_file, caption in model_figs:
    fig_path = FIGURES_DIR / fig_file
    if fig_path.exists():
        st.image(str(fig_path), caption=caption, use_container_width=True)
        st.divider()

# Deep learning figures
dl_figs = [
    ("21_dl_comparison.png", "Deep Learning model comparison"),
    ("22_dl_training_history.png", "Training history curves"),
    ("23_dl_predicted_vs_actual.png", "DL Predicted vs Actual"),
    ("24_all_models_comparison.png", "ML vs Deep Learning overall comparison"),
]

has_dl = any((FIGURES_DIR / f).exists() for f, _ in dl_figs)
if has_dl:
    st.subheader("Deep Learning Visualizations")
    for fig_file, caption in dl_figs:
        fig_path = FIGURES_DIR / fig_file
        if fig_path.exists():
            st.image(str(fig_path), caption=caption, use_container_width=True)
            st.divider()

# Optuna and SHAP figures
extra_figs = [
    ("25_optuna_comparison.png", "Optuna hyperparameter tuning results"),
    ("26_shap_summary.png", "SHAP summary (beeswarm) plot"),
    ("27_shap_bar.png", "SHAP feature importance (mean |SHAP|)"),
    ("28_shap_dependence.png", "SHAP dependence plots for top features"),
    ("29_shap_classification.png", "SHAP analysis for classification model"),
    ("30_final_comparison.png", "Final comprehensive model comparison"),
]

has_extra = any((FIGURES_DIR / f).exists() for f, _ in extra_figs)
if has_extra:
    st.subheader("Advanced Analysis Visualizations")
    for fig_file, caption in extra_figs:
        fig_path = FIGURES_DIR / fig_file
        if fig_path.exists():
            st.image(str(fig_path), caption=caption, use_container_width=True)
            st.divider()
