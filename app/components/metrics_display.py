"""Styled metric card components."""
import streamlit as st


AQI_COLORS = {
    "Good": "#009966",
    "Satisfactory": "#ffde33",
    "Moderate": "#ff9933",
    "Poor": "#cc0033",
    "Very Poor": "#660099",
    "Severe": "#7e0023",
}


def aqi_category_badge(category: str) -> str:
    """Return HTML for a colored AQI category badge."""
    color = AQI_COLORS.get(category, "#999")
    text_color = "#fff" if category not in ["Satisfactory"] else "#333"
    return (
        f'<span style="background-color:{color}; color:{text_color}; '
        f'padding:4px 12px; border-radius:20px; font-weight:bold;">'
        f'{category}</span>'
    )


def metric_card(label: str, value, subtitle: str = "", color: str = "#667eea"):
    """Render a styled metric card using st.markdown."""
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, {color} 0%, {color}cc 100%);
                    border-radius: 12px; padding: 1.2rem; color: white;
                    text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h4 style="margin:0; font-size:0.85rem; opacity:0.9;">{label}</h4>
            <div style="font-size:1.8rem; font-weight:bold; margin:0.3rem 0;">{value}</div>
            <div style="font-size:0.75rem; opacity:0.8;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def model_agreement_indicator(agree: bool):
    """Show whether CPCB formula and ML model agree."""
    if agree is None:
        st.info("Cannot determine model agreement")
    elif agree:
        st.success("Models Agree - CPCB formula and ML model predict the same category")
    else:
        st.warning("Models Disagree - CPCB formula and ML model predict different categories")
