"""City Analysis page - Historical trends and city comparisons."""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config import DATA_PROCESSED, FIGURES_DIR, AQI_CATEGORIES
from app.components.sidebar import render_sidebar

st.set_page_config(page_title="City Analysis", page_icon="🏙️", layout="wide")
render_sidebar()

st.title("City Analysis")

# Load data
@st.cache_data
def load_clean_data():
    path = DATA_PROCESSED / "city_day_clean.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None

df = load_clean_data()

if df is None:
    st.warning("Cleaned data not found. Run notebook 03 (preprocessing) first.")
    st.stop()

df["Date"] = pd.to_datetime(df["Date"])
cities = sorted(df["City"].unique())

tab_trends, tab_compare, tab_explore = st.tabs([
    "Historical Trends", "City Comparison", "Date Range Explorer"
])

# --- Historical Trends ---
with tab_trends:
    st.subheader("AQI Trends Over Time")
    selected_city = st.selectbox("Select City", cities, key="trend_city")

    city_data = df[df["City"] == selected_city].sort_values("Date")

    # Monthly average
    monthly = city_data.set_index("Date").resample("M")["AQI"].mean().reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=city_data["Date"], y=city_data["AQI"],
        mode="markers", marker=dict(size=3, opacity=0.3, color="steelblue"),
        name="Daily AQI",
    ))
    fig.add_trace(go.Scatter(
        x=monthly["Date"], y=monthly["AQI"],
        mode="lines", line=dict(width=3, color="red"),
        name="Monthly Average",
    ))

    # AQI category bands
    band_colors = ["rgba(0,153,102,0.1)", "rgba(255,222,51,0.1)", "rgba(255,153,51,0.1)",
                   "rgba(204,0,51,0.1)", "rgba(102,0,153,0.1)", "rgba(126,0,35,0.1)"]
    for i, (cat, (lo, hi)) in enumerate(AQI_CATEGORIES.items()):
        fig.add_hrect(y0=lo, y1=hi, fillcolor=band_colors[i],
                      annotation_text=cat if i < 4 else "",
                      annotation_position="top left", line_width=0)

    fig.update_layout(title=f"{selected_city} - AQI Over Time",
                      xaxis_title="Date", yaxis_title="AQI",
                      height=500, yaxis=dict(range=[0, min(city_data["AQI"].max() * 1.1, 500)]))
    st.plotly_chart(fig, use_container_width=True)

    # Seasonal pattern
    city_data["month"] = city_data["Date"].dt.month
    monthly_avg = city_data.groupby("month")["AQI"].mean().reset_index()
    month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                   7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
    monthly_avg["month_name"] = monthly_avg["month"].map(month_names)

    fig2 = px.bar(monthly_avg, x="month_name", y="AQI",
                  title=f"{selected_city} - Seasonal AQI Pattern",
                  color="AQI", color_continuous_scale="RdYlGn_r")
    fig2.update_layout(xaxis_title="Month", yaxis_title="Mean AQI")
    st.plotly_chart(fig2, use_container_width=True)

# --- City Comparison ---
with tab_compare:
    st.subheader("Compare Cities")
    selected_cities = st.multiselect("Select cities to compare", cities,
                                     default=["Delhi", "Mumbai", "Bengaluru", "Chennai", "Kolkata"])

    if selected_cities:
        compare_data = df[df["City"].isin(selected_cities)]

        # Boxplot
        fig = px.box(compare_data, x="City", y="AQI", color="City",
                     title="AQI Distribution by City")
        fig.update_layout(showlegend=False, height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Monthly trends overlay
        monthly_compare = (
            compare_data.set_index("Date")
            .groupby("City")
            .resample("M")["AQI"].mean()
            .reset_index()
        )
        fig2 = px.line(monthly_compare, x="Date", y="AQI", color="City",
                       title="Monthly AQI Trends")
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)

        # Summary stats table
        summary = compare_data.groupby("City")["AQI"].agg(
            ["mean", "median", "std", "min", "max", "count"]
        ).round(2)
        summary.columns = ["Mean", "Median", "Std Dev", "Min", "Max", "Days"]
        st.dataframe(summary.sort_values("Mean", ascending=False), use_container_width=True)

# --- Date Range Explorer ---
with tab_explore:
    st.subheader("Explore Data by Date Range")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", df["Date"].min().date())
    with col2:
        end_date = st.date_input("End Date", df["Date"].max().date())

    explore_city = st.selectbox("City", ["All Cities"] + cities, key="explore_city")

    mask = (df["Date"] >= pd.Timestamp(start_date)) & (df["Date"] <= pd.Timestamp(end_date))
    if explore_city != "All Cities":
        mask = mask & (df["City"] == explore_city)
    filtered = df[mask]

    st.markdown(f"**{len(filtered):,} records** in selected range")

    if not filtered.empty:
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Mean AQI", f"{filtered['AQI'].mean():.1f}")
        col_m2.metric("Median AQI", f"{filtered['AQI'].median():.1f}")
        col_m3.metric("Max AQI", f"{filtered['AQI'].max():.1f}")
        col_m4.metric("Days > 200", f"{(filtered['AQI'] > 200).sum()}")

        fig = px.histogram(filtered, x="AQI", nbins=50,
                           title="AQI Distribution in Selected Range",
                           color_discrete_sequence=["steelblue"])
        st.plotly_chart(fig, use_container_width=True)

        if explore_city == "All Cities":
            city_means = filtered.groupby("City")["AQI"].mean().sort_values(ascending=True)
            fig2 = px.bar(city_means, orientation="h",
                          title="Mean AQI by City (Selected Range)",
                          color=city_means.values, color_continuous_scale="RdYlGn_r")
            fig2.update_layout(showlegend=False, yaxis_title="", xaxis_title="Mean AQI",
                               height=max(300, len(city_means) * 25))
            st.plotly_chart(fig2, use_container_width=True)
