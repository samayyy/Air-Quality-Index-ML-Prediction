"""
Page 7: AQI Forecast & Insights
7-day forecasting with causal analysis and AI-powered narratives.
"""
import streamlit as st
import pandas as pd
import os
from datetime import datetime
from pathlib import Path

import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import WAQI_CITY_MAP
from app.components.sidebar import render_sidebar
from app.components.charts import (
    aqi_gauge, pollutant_bar, forecast_line_chart, cause_breakdown_chart,
    AQI_COLORS,
)
from src.data.api_client import fetch_city_realtime, get_token
from src.models.forecast import AQIForecaster
from src.analysis.causal import analyze_causes, get_recommendations
from src.analysis.llm_insights import generate_forecast_narrative, generate_health_advisory

render_sidebar()

st.markdown(
    '<h1 class="page-title">AQI Forecast & Insights</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    "7-day AQI forecasting with weather context, causal analysis, "
    "and AI-powered recommendations."
)

# ============================================================
# Section 1: City Selection + Current Status
# ============================================================
st.divider()

cities = sorted(WAQI_CITY_MAP.keys())
col_select, col_status = st.columns([1, 3])

with col_select:
    city = st.selectbox("Select City", cities, index=cities.index("Delhi"))
    forecast_days = st.slider("Forecast Days", 1, 7, 7)

# Fetch current data
try:
    token = get_token()
except ValueError:
    st.error(
        "WAQI API token not configured. Add `WAQI_API_TOKEN` to your `.env` file. "
        "Get a free token at https://aqicn.org/data-platform/token/"
    )
    st.stop()


@st.cache_data(ttl=600, show_spinner="Fetching current AQI data...")
def get_current_data(city_name, _token):
    return fetch_city_realtime(city_name, _token)


current = get_current_data(city, token)

with col_status:
    if current:
        c1, c2, c3 = st.columns(3)
        with c1:
            current_aqi = current.get("aqi")
            if current_aqi and str(current_aqi).replace("-", "").isdigit():
                st.plotly_chart(
                    aqi_gauge(float(current_aqi), title=f"Current AQI — {city}"),
                    use_container_width=True,
                )
            else:
                st.metric("Current AQI", "N/A")
        with c2:
            station_name = current.get("station", "N/A")
            st.metric("Monitoring Station", station_name)
            dominant = current.get("dominant_pollutant", "N/A")
            st.metric("Dominant Pollutant", dominant.upper() if dominant else "N/A")
        with c3:
            time_str = current.get("time", "")
            if time_str:
                try:
                    t = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
                    st.metric("Last Updated", t.strftime("%b %d, %I:%M %p"))
                except (ValueError, TypeError):
                    st.metric("Last Updated", str(time_str)[:19])
            else:
                st.metric("Last Updated", "N/A")

        st.caption(
            f"AQI shown is from **{current.get('station', 'WAQI station')}** "
            f"on the US-EPA scale (same as Google, AQICN, IQAir). "
            f"Values may differ from other sites that use different stations or city-wide averages."
        )

        # Current pollutant bar chart
        pollutants = current.get("pollutants", {})
        if pollutants:
            st.plotly_chart(
                pollutant_bar(pollutants, title=f"Current Pollutant Readings — {city}"),
                use_container_width=True,
            )
    else:
        st.warning(f"Could not fetch current data for {city}. The forecast will use available data.")

# ============================================================
# Section 2: 7-Day Forecast
# ============================================================
st.divider()
st.subheader("7-Day AQI Forecast")

# Show models used in forecasting
@st.cache_resource
def get_model_info():
    from src.models.predict import AQIPredictor
    p = AQIPredictor()
    p.load_models()
    return p.reg_model_name, p.cls_model_name

try:
    reg_name, cls_name = get_model_info()
    st.markdown(
        f"> **Models powering this forecast:**\n"
        f"> - **AQI Value (Regression):** `{reg_name}`\n"
        f"> - **AQI Category (Classification):** `{cls_name}`\n"
        f"> - **AQI Formula:** CPCB Sub-Index Method\n"
        f"> - **Weather Data:** Open-Meteo API\n"
        f"> - **Pollutant Forecasts:** WAQI API"
    )
except Exception:
    pass


@st.cache_data(ttl=1800, show_spinner="Generating AQI forecast...")
def run_forecast(city_name, days, _token):
    forecaster = AQIForecaster()
    return forecaster.forecast(city_name, days, _token)


with st.spinner("Running forecast model..."):
    try:
        forecast_data = run_forecast(city, forecast_days, token)
    except Exception as e:
        st.error(f"Forecast generation failed: {e}")
        forecast_data = []

if forecast_data:
    # Forecast line chart
    st.plotly_chart(
        forecast_line_chart(forecast_data, title=f"AQI Forecast — {city}"),
        use_container_width=True,
    )

    # Daily breakdown table
    table_data = []
    for f in forecast_data:
        weather = f.get("weather", {})
        weather_summary = ""
        parts = []
        if weather.get("temp_max") is not None:
            parts.append(f"{weather['temp_max']:.0f}/{weather.get('temp_min', 0):.0f}°C")
        if weather.get("humidity") is not None:
            parts.append(f"{weather['humidity']:.0f}% RH")
        if weather.get("wind_speed") is not None:
            parts.append(f"{weather['wind_speed']:.0f} km/h wind")
        if weather.get("precipitation") and weather["precipitation"] > 0:
            parts.append(f"{weather['precipitation']:.1f}mm rain")
        weather_summary = " | ".join(parts)

        category = f.get("category", "N/A")
        table_data.append({
            "Date": f["date"].strftime("%a, %b %d") if hasattr(f["date"], "strftime") else str(f["date"]),
            "AQI": f"{f.get('predicted_aqi', 0):.0f}" if f.get("predicted_aqi") else "N/A",
            "Category": category,
            "Dominant": f.get("dominant_pollutant", "N/A"),
            "Confidence": f"{f.get('confidence', 0):.0%}",
            "Weather": weather_summary or "N/A",
        })

    st.dataframe(
        pd.DataFrame(table_data),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No forecast data available. Ensure WAQI API token is valid and the city has data.")

# ============================================================
# Section 3: Causal Analysis
# ============================================================
st.divider()
st.subheader("What's Causing This AQI?")

# Use current or first forecast day pollutants for causal analysis
analysis_pollutants = {}
if current and current.get("pollutants"):
    analysis_pollutants = current["pollutants"]
elif forecast_data:
    analysis_pollutants = forecast_data[0].get("pollutants", {})

if analysis_pollutants:
    causal_results = analyze_causes(analysis_pollutants, city=city)

    col_causes, col_chart = st.columns([1, 1])

    with col_causes:
        for c in causal_results:
            confidence_pct = c["confidence"] * 100
            color = "#009966" if confidence_pct < 50 else "#ff9933" if confidence_pct < 75 else "#cc0033"
            st.markdown(
                f"**{c['cause']}** — {confidence_pct:.0f}% confidence"
            )
            st.progress(c["confidence"])

        # Seasonal context
        month = datetime.now().month
        if month in (10, 11) and city in ("Delhi", "Lucknow", "Chandigarh", "Amritsar", "Patna"):
            st.info("Seasonal note: Oct-Nov is crop burning season in Punjab/Haryana, "
                    "which significantly impacts AQI in northern cities.")
        elif month in (12, 1):
            st.info("Seasonal note: Winter temperature inversions trap pollutants near ground level, "
                    "typically worsening AQI across northern India.")
        elif month in (6, 7, 8, 9):
            st.info("Seasonal note: Monsoon rains help wash out particulate matter, "
                    "generally improving air quality.")

    with col_chart:
        st.plotly_chart(
            cause_breakdown_chart(causal_results),
            use_container_width=True,
        )

    # Fetch AI-powered real-time context for the city
    from src.analysis.causal import fetch_aqi_news
    @st.cache_data(ttl=3600, show_spinner="Fetching real-time context...")
    def get_city_context(city_name):
        return fetch_aqi_news(city_name)

    news_context = get_city_context(city)
    if news_context:
        with st.expander("Real-Time City Context (AI-Sourced)"):
            for item in news_context:
                st.markdown(item["snippet"])
                st.caption(f"Source: {item['source']}")
else:
    causal_results = []
    st.info("Pollutant data not available for causal analysis.")

# ============================================================
# Section 4: AI-Powered Insights (Claude API)
# ============================================================
st.divider()
st.subheader("AI-Powered Insights")

has_api_key = bool(os.environ.get("GEMINI_API_KEY", ""))

if forecast_data and causal_results:

    @st.cache_data(ttl=3600, show_spinner="Generating AI insights...")
    def get_narrative(_forecast_key, _causal_key, city_name):
        """Cache-friendly wrapper. Key args are hashable representations."""
        return generate_forecast_narrative(forecast_data, causal_results, city_name)

    @st.cache_data(ttl=3600, show_spinner="Generating health advisory...")
    def get_advisory(aqi_val, cat, _causal_key, city_name):
        return generate_health_advisory(aqi_val, cat, causal_results, city_name)

    # Create hashable keys for caching
    forecast_key = str([(f["date"], f.get("predicted_aqi")) for f in forecast_data])
    causal_key = str([(c["cause"], c["confidence"]) for c in causal_results])

    narrative = get_narrative(forecast_key, causal_key, city)
    st.markdown(narrative)

    if not has_api_key:
        st.caption(
            "Add `GEMINI_API_KEY` to your `.env` file for richer AI-generated insights. "
            "Currently showing template-based analysis."
        )

    with st.expander("Detailed Health Advisory"):
        current_aqi = float(current["aqi"]) if current and current.get("aqi") else (
            forecast_data[0].get("predicted_aqi", 100) if forecast_data else 100
        )
        current_category = forecast_data[0].get("category", "Moderate") if forecast_data else "Moderate"
        advisory = get_advisory(current_aqi, current_category, causal_key, city)
        st.markdown(advisory)
else:
    st.info("Forecast and causal data needed to generate insights.")

# ============================================================
# Section 5: Recommendations
# ============================================================
st.divider()
st.subheader("Recommendations")

if analysis_pollutants and causal_results:
    current_aqi_val = float(current["aqi"]) if current and current.get("aqi") else (
        forecast_data[0].get("predicted_aqi", 100) if forecast_data else 100
    )
    current_cat = forecast_data[0].get("category", "Moderate") if forecast_data else "Moderate"

    recs = get_recommendations(current_aqi_val, causal_results, current_cat)

    col_health, col_actions = st.columns(2)

    with col_health:
        st.markdown("#### Health Advisory")
        severity_icons = {
            "Good": "🟢", "Satisfactory": "🟡", "Moderate": "🟠",
            "Poor": "🔴", "Very Poor": "🟣", "Severe": "⚫",
        }
        icon = severity_icons.get(recs["category"], "⚪")
        st.markdown(f"{icon} **Current Level: {recs['category']}**")

        for advice in recs["health"]:
            st.markdown(f"- {advice}")

    with col_actions:
        st.markdown("#### What Can Be Done")

        if recs["actions_personal"]:
            st.markdown("**Personal Actions:**")
            for action in recs["actions_personal"]:
                st.markdown(f"- {action}")

        if recs["actions_community"]:
            st.markdown("**Community Actions:**")
            for action in recs["actions_community"]:
                st.markdown(f"- {action}")

        if recs["actions_policy"]:
            st.markdown("**Policy Level:**")
            for action in recs["actions_policy"]:
                st.markdown(f"- {action}")

    # Vulnerable groups section
    if recs.get("vulnerable_groups"):
        with st.expander("For Vulnerable Groups"):
            group_icons = {
                "children": "👶", "elderly": "👴", "respiratory": "🫁",
                "pregnant": "🤰", "outdoor_workers": "👷",
            }
            for group, advice in recs["vulnerable_groups"].items():
                icon = group_icons.get(group, "⚠️")
                st.markdown(f"{icon} **{group.replace('_', ' ').title()}**: {advice}")
else:
    st.info("Select a city above to see recommendations.")
