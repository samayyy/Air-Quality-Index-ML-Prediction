"""Reusable Plotly chart builders for the Streamlit app."""
import plotly.graph_objects as go
import plotly.express as px
import numpy as np


AQI_COLORS = {
    "Good": "#009966",
    "Satisfactory": "#ffde33",
    "Moderate": "#ff9933",
    "Poor": "#cc0033",
    "Very Poor": "#660099",
    "Severe": "#7e0023",
}

AQI_RANGES = [
    (0, 50, "Good"),
    (51, 100, "Satisfactory"),
    (101, 200, "Moderate"),
    (201, 300, "Poor"),
    (301, 400, "Very Poor"),
    (401, 500, "Severe"),
]


def aqi_gauge(value: float, title: str = "AQI") -> go.Figure:
    """Create a gauge chart for AQI value."""
    if value is None:
        value = 0

    # Determine category
    category = "Unknown"
    color = "#999"
    for low, high, cat in AQI_RANGES:
        if low <= value <= high:
            category = cat
            color = AQI_COLORS[cat]
            break
    if value > 500:
        category = "Severe"
        color = AQI_COLORS["Severe"]

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        title={"text": f"{title}<br><span style='font-size:0.8em;color:{color}'>{category}</span>"},
        gauge={
            "axis": {"range": [0, 500], "tickwidth": 1},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 50], "color": "#e8f5e9"},
                {"range": [51, 100], "color": "#fff9c4"},
                {"range": [101, 200], "color": "#ffe0b2"},
                {"range": [201, 300], "color": "#ffcdd2"},
                {"range": [301, 400], "color": "#e1bee7"},
                {"range": [401, 500], "color": "#f8bbd0"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": value,
            },
        },
    ))
    fig.update_layout(height=300, margin=dict(t=80, b=20, l=30, r=30))
    return fig


def model_comparison_bar(df, metric="R2", title="Model Comparison"):
    """Create a horizontal bar chart comparing models on a metric."""
    df_sorted = df.sort_values(metric, ascending=True)

    fig = px.bar(
        df_sorted, x=metric, y=df_sorted.index,
        orientation="h", title=title,
        color=metric, color_continuous_scale="Viridis",
    )
    fig.update_layout(
        height=max(300, len(df_sorted) * 35),
        showlegend=False,
        yaxis_title="",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def pollutant_bar(pollutants: dict, title: str = "Pollutant Levels") -> go.Figure:
    """Create a bar chart of pollutant concentrations."""
    names = list(pollutants.keys())
    values = list(pollutants.values())

    fig = go.Figure(go.Bar(
        x=names, y=values,
        marker_color=px.colors.qualitative.Set2[:len(names)],
        text=[f"{v:.1f}" if v else "N/A" for v in values],
        textposition="auto",
    ))
    fig.update_layout(
        title=title, height=350,
        xaxis_title="Pollutant", yaxis_title="Concentration",
        margin=dict(t=40, b=10),
    )
    return fig


def forecast_line_chart(forecast_data: list[dict],
                        title: str = "7-Day AQI Forecast") -> go.Figure:
    """
    Create a line chart of AQI forecast with color-coded category bands
    and confidence intervals.
    """
    dates = [f["date"] for f in forecast_data]
    aqi_values = [f.get("predicted_aqi") or 0 for f in forecast_data]
    confidences = [f.get("confidence", 0.5) for f in forecast_data]

    # Confidence interval (wider for lower confidence)
    upper = [v + v * (1 - c) * 0.3 for v, c in zip(aqi_values, confidences)]
    lower = [max(0, v - v * (1 - c) * 0.3) for v, c in zip(aqi_values, confidences)]

    fig = go.Figure()

    # AQI category background bands
    band_colors = [
        (0, 50, "rgba(0,153,102,0.1)"),
        (50, 100, "rgba(255,222,51,0.1)"),
        (100, 200, "rgba(255,153,51,0.1)"),
        (200, 300, "rgba(204,0,51,0.1)"),
        (300, 400, "rgba(102,0,153,0.1)"),
        (400, 500, "rgba(126,0,35,0.1)"),
    ]

    max_aqi = max(max(upper), 100)
    for y0, y1, color in band_colors:
        if y0 > max_aqi * 1.3:
            break
        fig.add_hrect(
            y0=y0, y1=min(y1, max_aqi * 1.3),
            fillcolor=color, line_width=0,
            annotation_text=AQI_RANGES[[r[0] for r in AQI_RANGES].index(y0 if y0 > 0 else 0)][2] if y0 in [0, 51, 101, 201, 301, 401] else "",
            annotation_position="right",
        )

    # Confidence interval shading
    fig.add_trace(go.Scatter(
        x=dates + dates[::-1],
        y=upper + lower[::-1],
        fill="toself",
        fillcolor="rgba(99,110,250,0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        showlegend=True,
        name="Confidence Interval",
    ))

    # Main AQI line
    # Color each point by category
    point_colors = []
    for v in aqi_values:
        color = "#999"
        for low, high, cat in AQI_RANGES:
            if low <= v <= high:
                color = AQI_COLORS[cat]
                break
        if v > 500:
            color = AQI_COLORS["Severe"]
        point_colors.append(color)

    fig.add_trace(go.Scatter(
        x=dates, y=aqi_values,
        mode="lines+markers",
        name="Predicted AQI",
        line=dict(color="#636EFA", width=3),
        marker=dict(size=10, color=point_colors, line=dict(width=2, color="white")),
        text=[f"AQI: {v:.0f}<br>Confidence: {c:.0%}" for v, c in zip(aqi_values, confidences)],
        hoverinfo="text+x",
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="AQI",
        height=450,
        yaxis=dict(range=[0, max(max_aqi * 1.2, 100)]),
        margin=dict(t=50, b=30, l=50, r=80),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def cause_breakdown_chart(causes: list[dict],
                          title: str = "Pollution Source Analysis") -> go.Figure:
    """Create a donut chart showing identified pollution causes and confidence."""
    if not causes:
        fig = go.Figure()
        fig.add_annotation(text="No causes identified", showarrow=False)
        return fig

    labels = [c["cause"] for c in causes]
    values = [c["confidence"] for c in causes]
    colors = px.colors.qualitative.Set2[:len(causes)]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.45,
        marker=dict(colors=colors),
        textinfo="label+percent",
        textposition="outside",
        hovertemplate="%{label}<br>Confidence: %{value:.0%}<extra></extra>",
    ))

    fig.update_layout(
        title=title,
        height=400,
        margin=dict(t=50, b=10, l=10, r=10),
        showlegend=False,
    )
    return fig
