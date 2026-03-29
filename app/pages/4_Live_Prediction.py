"""Live Prediction page - Manual input + WAQI API live data."""
import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config import WAQI_CITY_MAP, AQI_POLLUTANTS
from app.components.sidebar import render_sidebar
from app.components.charts import aqi_gauge, pollutant_bar
from app.components.metrics_display import aqi_category_badge, metric_card, model_agreement_indicator

st.set_page_config(page_title="Live Prediction", page_icon="🎯", layout="wide")
render_sidebar()

st.title("Live AQI Prediction")

# Load predictor
@st.cache_resource
def load_predictor():
    from src.models.predict import AQIPredictor
    predictor = AQIPredictor()
    try:
        predictor.load_models()
        return predictor
    except Exception as e:
        st.error(f"Could not load models: {e}")
        return None

predictor = load_predictor()

mode = st.radio("Select Mode", ["Manual Input", "Live City Data"], horizontal=True)

if mode == "Manual Input":
    st.subheader("Enter Pollutant Concentrations")
    st.caption("Adjust the sliders or use preset values to predict AQI.")

    # Presets
    col_presets = st.columns(4)
    with col_presets[0]:
        if st.button("Delhi Winter"):
            st.session_state.update({
                "pm25": 250.0, "pm10": 350.0, "no2": 80.0,
                "so2": 18.0, "co": 4.5, "o3": 25.0, "nh3": 35.0,
            })
    with col_presets[1]:
        if st.button("Clean Monsoon"):
            st.session_state.update({
                "pm25": 25.0, "pm10": 45.0, "no2": 15.0,
                "so2": 8.0, "co": 0.8, "o3": 30.0, "nh3": 10.0,
            })
    with col_presets[2]:
        if st.button("Moderate City"):
            st.session_state.update({
                "pm25": 80.0, "pm10": 130.0, "no2": 45.0,
                "so2": 15.0, "co": 2.0, "o3": 40.0, "nh3": 20.0,
            })
    with col_presets[3]:
        if st.button("Reset"):
            for k in ["pm25", "pm10", "no2", "so2", "co", "o3", "nh3"]:
                st.session_state.pop(k, None)

    # Pollutant sliders
    col1, col2, col3 = st.columns(3)
    with col1:
        pm25 = st.slider("PM2.5 (μg/m³)", 0.0, 500.0,
                          st.session_state.get("pm25", 60.0), 1.0)
        pm10 = st.slider("PM10 (μg/m³)", 0.0, 600.0,
                          st.session_state.get("pm10", 100.0), 1.0)
        nh3 = st.slider("NH3 (μg/m³)", 0.0, 400.0,
                         st.session_state.get("nh3", 20.0), 1.0)
    with col2:
        no2 = st.slider("NO2 (μg/m³)", 0.0, 300.0,
                         st.session_state.get("no2", 40.0), 1.0)
        so2 = st.slider("SO2 (μg/m³)", 0.0, 200.0,
                         st.session_state.get("so2", 15.0), 1.0)
    with col3:
        co = st.slider("CO (mg/m³)", 0.0, 50.0,
                        st.session_state.get("co", 2.0), 0.1)
        o3 = st.slider("O3 (μg/m³)", 0.0, 300.0,
                        st.session_state.get("o3", 35.0), 1.0)

    pollutants = {
        "PM2.5": pm25, "PM10": pm10, "NO2": no2,
        "SO2": so2, "CO": co, "O3": o3, "NH3": nh3,
    }

    if st.button("Predict AQI", type="primary"):
        st.divider()

        if predictor:
            result = predictor.predict_from_pollutants(pollutants)
        else:
            # Fallback: just compute CPCB
            from src.data.aqi_calculator import compute_sub_index, categorize_aqi
            sub_indices = {}
            for p in AQI_POLLUTANTS:
                if p in pollutants:
                    si = compute_sub_index(p, pollutants[p])
                    if si is not None:
                        sub_indices[p] = si
            if sub_indices:
                dominant = max(sub_indices, key=sub_indices.get)
                aqi_val = round(sub_indices[dominant], 2)
                result = {
                    "cpcb_aqi": aqi_val, "cpcb_category": categorize_aqi(aqi_val),
                    "cpcb_dominant": dominant, "ml_aqi": None, "ml_category": None,
                    "model_agreement": None,
                }
            else:
                result = {"cpcb_aqi": None, "cpcb_category": None, "ml_aqi": None}

        # Display results
        # Show which models are being used
        if predictor and predictor._loaded:
            st.info(
                f"**Models Used** — Regression: `{predictor.reg_model_name}` | "
                f"Classification: `{predictor.cls_model_name}`"
            )

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            if result.get("cpcb_aqi") is not None:
                st.plotly_chart(aqi_gauge(result["cpcb_aqi"], "CPCB Formula AQI"),
                                use_container_width=True)
        with col_g2:
            if result.get("ml_aqi") is not None:
                ml_title = f"ML Prediction ({predictor.reg_model_name.split('(')[0].strip()})" if predictor else "ML Model AQI"
                st.plotly_chart(aqi_gauge(result["ml_aqi"], ml_title),
                                use_container_width=True)
            else:
                st.info("ML model prediction not available (models not loaded)")

        # Details
        col_d1, col_d2, col_d3 = st.columns(3)
        with col_d1:
            if result.get("cpcb_category"):
                st.markdown(f"**CPCB Category**: {aqi_category_badge(result['cpcb_category'])}",
                            unsafe_allow_html=True)
            if result.get("cpcb_dominant"):
                st.markdown(f"**Dominant Pollutant**: {result['cpcb_dominant']}")
        with col_d2:
            if result.get("ml_category"):
                st.markdown(f"**ML Category**: {aqi_category_badge(result['ml_category'])}",
                            unsafe_allow_html=True)
        with col_d3:
            model_agreement_indicator(result.get("model_agreement"))

        # Pollutant chart
        st.plotly_chart(pollutant_bar(pollutants), use_container_width=True)

else:  # Live City Data
    st.subheader("Fetch Live Data from WAQI API")

    city = st.selectbox("Select City", sorted(WAQI_CITY_MAP.keys()))

    if st.button("Fetch & Predict", type="primary"):
        with st.spinner(f"Fetching data for {city}..."):
            try:
                from src.data.api_client import fetch_city_realtime
                data = fetch_city_realtime(city)

                if data is None:
                    st.error(f"Could not fetch data for {city}. Check your WAQI API token in .env file.")
                else:
                    st.success(f"Data fetched from: {data.get('station', 'Unknown station')}")
                    st.caption(f"Last updated: {data.get('time', 'Unknown')}")

                    pollutants = data.get("pollutants", {})
                    waqi_aqi = data.get("aqi")

                    # Show pollutant readings
                    st.subheader("Current Readings")
                    poll_cols = st.columns(len(pollutants) if pollutants else 1)
                    for i, (p, v) in enumerate(pollutants.items()):
                        with poll_cols[i % len(poll_cols)]:
                            st.metric(p, f"{v:.1f}" if v else "N/A")

                    if not pollutants:
                        st.warning("No pollutant data available from WAQI")
                    else:
                        # Fill missing NH3 with historical median
                        if "NH3" not in pollutants:
                            try:
                                from src.data.api_client import get_city_medians
                                medians = get_city_medians()
                                if city in medians and "NH3" in medians[city]:
                                    pollutants["NH3"] = medians[city]["NH3"]
                                    st.caption(f"NH3 not available from WAQI; using historical median: {pollutants['NH3']:.1f}")
                            except Exception:
                                pass

                        # Run predictions (with city medians for better lag estimation)
                        if predictor:
                            city_meds = None
                            try:
                                from src.data.api_client import get_city_medians
                                all_meds = get_city_medians()
                                city_meds = all_meds.get(city)
                            except Exception:
                                pass
                            result = predictor.predict_from_pollutants(pollutants, city_medians=city_meds)

                            st.divider()
                            if predictor._loaded:
                                st.info(
                                    f"**Models Used** — Regression: `{predictor.reg_model_name}` | "
                                    f"Classification: `{predictor.cls_model_name}`"
                                )

                            # Show gauges: reported vs ML predicted
                            col_g1, col_g2, col_g3 = st.columns(3)
                            with col_g1:
                                if waqi_aqi and str(waqi_aqi).replace("-", "").isdigit():
                                    st.plotly_chart(aqi_gauge(int(waqi_aqi), "WAQI Station AQI"),
                                                    use_container_width=True)
                                    st.caption("Live value (US-EPA scale)")
                            with col_g2:
                                ml_cpcb = result.get("ml_aqi")
                                if ml_cpcb:
                                    ml_title = f"ML Predicted ({predictor.reg_model_name.split('(')[0].strip()})"
                                    st.plotly_chart(aqi_gauge(ml_cpcb, ml_title),
                                                    use_container_width=True)
                                    st.caption("ML prediction (Indian CPCB scale)")
                                else:
                                    st.info("ML model prediction not available")
                            with col_g3:
                                ml_epa = result.get("ml_aqi_epa")
                                if ml_epa:
                                    st.plotly_chart(aqi_gauge(ml_epa, "ML Predicted (EPA)"),
                                                    use_container_width=True)
                                    st.caption("Converted to EPA scale via dominant pollutant")

                            st.caption(
                                "**WAQI Station** shows the live reported value on the US-EPA scale (same as Google/AQICN). "
                                "**ML Predicted** shows our LightGBM model's prediction on the Indian CPCB scale. "
                                "**ML (EPA)** converts the prediction to EPA scale for comparison — accurate when the same "
                                "pollutant dominates on both scales (PM2.5/PM10 cities), approximate otherwise."
                            )

                            model_agreement_indicator(result.get("model_agreement"))

                    st.plotly_chart(pollutant_bar(pollutants, f"{city} - Current Pollutant Levels"),
                                    use_container_width=True)

                    # Cache the reading
                    try:
                        from src.data.cache_manager import add_reading
                        add_reading(city, pollutants, waqi_aqi)
                    except Exception:
                        pass

            except ValueError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Error: {e}")
                st.info("Make sure you have a valid WAQI_API_TOKEN in your .env file. "
                        "Get one at https://aqicn.org/data-platform/token/")
