"""Shared sidebar component for the Streamlit app."""
import streamlit as st


def render_sidebar():
    """Render the shared sidebar with project info and navigation help."""
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/air-quality.png", width=60)
        st.title("India AQI Predictor")
        st.caption("ML-powered Air Quality Analysis")

        st.divider()

        st.markdown("### About")
        st.markdown(
            "This project predicts India's Air Quality Index using "
            "CPCB-verified data (2015-2020) and multiple ML models."
        )

        st.divider()

        st.markdown("### AQI Scale (CPCB)")
        aqi_scale = {
            "Good": ("0-50", "#009966"),
            "Satisfactory": ("51-100", "#ffde33"),
            "Moderate": ("101-200", "#ff9933"),
            "Poor": ("201-300", "#cc0033"),
            "Very Poor": ("301-400", "#660099"),
            "Severe": ("401-500", "#7e0023"),
        }
        for cat, (rng, color) in aqi_scale.items():
            st.markdown(
                f'<span style="color:{color}; font-weight:bold;">●</span> '
                f'**{cat}**: {rng}',
                unsafe_allow_html=True,
            )

        st.divider()
        st.caption("Built with Streamlit | Data: CPCB via Kaggle")
