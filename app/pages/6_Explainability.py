"""Explainability page - SHAP plots and feature importance."""
import streamlit as st
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config import FIGURES_DIR
from app.components.sidebar import render_sidebar

st.set_page_config(page_title="Explainability", page_icon="🔍", layout="wide")
render_sidebar()

st.title("Model Explainability (SHAP)")

st.markdown("""
**SHAP (SHapley Additive exPlanations)** provides consistent, interpretable explanations
for model predictions. Each feature gets a SHAP value indicating its contribution to moving
the prediction away from the baseline.

- **Positive SHAP value**: Feature pushes prediction higher (worse AQI)
- **Negative SHAP value**: Feature pushes prediction lower (better AQI)
""")

st.divider()

# Feature Importance (from training)
st.subheader("Feature Importance (Best Regression Model)")
fig_path = FIGURES_DIR / "18_feature_importance.png"
if fig_path.exists():
    st.image(str(fig_path), caption="Top 25 features by model importance", use_container_width=True)

st.divider()

# SHAP Plots
shap_figs = [
    ("26_shap_summary.png", "SHAP Summary Plot (Beeswarm)",
     "Each dot is a single prediction. Position on X-axis = SHAP value (impact). "
     "Color = feature value (red=high, blue=low)."),
    ("27_shap_bar.png", "SHAP Feature Importance (Mean |SHAP|)",
     "Average absolute SHAP value per feature. Higher = more influential overall."),
    ("28_shap_dependence.png", "SHAP Dependence Plots",
     "How individual feature values affect predictions. Shows interaction effects."),
    ("29_shap_classification.png", "SHAP for Classification Model",
     "Feature importance for the AQI category classification task."),
]

for fig_name, title, description in shap_figs:
    fig_path = FIGURES_DIR / fig_name
    if fig_path.exists():
        st.subheader(title)
        st.caption(description)
        st.image(str(fig_path), use_container_width=True)
        st.divider()

# Check if any SHAP figures exist
shap_exists = any((FIGURES_DIR / f).exists() for f, _, _ in shap_figs)
if not shap_exists:
    st.info(
        "SHAP analysis figures not found. Run notebook 08 (SHAP analysis) to generate them."
    )

# Key Insights
st.subheader("Key Insights")
st.markdown("""
### Expected Feature Importance Patterns

Based on the model training results:

1. **PM2.5 and PM10** are typically the most influential features for AQI prediction,
   as they are the dominant pollutants in most Indian cities.

2. **Lag features** (especially 1-day and 7-day lags) capture temporal persistence -
   pollution today is strongly correlated with pollution yesterday.

3. **Rolling averages** (7-day and 14-day windows) capture medium-term trends
   and seasonal patterns.

4. **Temporal features** (month, season) capture annual cycles -
   winter months consistently show higher AQI.

5. **City encoding** captures the baseline pollution level differences
   between industrial/metro cities and cleaner coastal cities.

### Interpreting SHAP Values
- A high PM2.5 value with a high positive SHAP value means: "High PM2.5 concentration
  is pushing the predicted AQI upward (worse air quality)."
- A monsoon season feature with a negative SHAP value means: "Being in monsoon season
  pushes the predicted AQI downward (better air quality)."
""")
