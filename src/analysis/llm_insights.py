"""
LLM-powered insights using Google Gemini API (google-genai SDK).
Generates natural language narratives for AQI forecasts and health advisories.
Falls back to template-based text when API key is unavailable.
"""
import os
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import GEMINI_MODEL

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _get_client():
    """Get Google GenAI client if API key is available."""
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return None
    try:
        from google import genai
        return genai.Client(api_key=api_key)
    except (ImportError, Exception):
        return None


def generate_forecast_narrative(forecast_data: list[dict],
                                causal_analysis: list[dict],
                                city: str) -> str:
    """
    Generate a natural language narrative explaining the AQI forecast trend.

    Args:
        forecast_data: List of daily forecast dicts from AQIForecaster
        causal_analysis: List of cause dicts from analyze_causes
        city: City name

    Returns:
        2-3 paragraph narrative string. Falls back to template if no API key.
    """
    client = _get_client()
    if client is None:
        return _template_forecast_narrative(forecast_data, causal_analysis, city)

    # Get city profile for context
    from src.analysis.causal import CITY_PROFILES
    city_profile = CITY_PROFILES.get(city, {})
    city_context = city_profile.get("context", f"a major Indian city")
    primary_sources = city_profile.get("primary_sources", [])

    # Build structured prompt
    forecast_summary = []
    for f in forecast_data:
        weather = f.get("weather", {})
        weather_str = ""
        if weather:
            parts = []
            if weather.get("temp_max") is not None:
                parts.append(f"{weather['temp_max']:.0f}°C")
            if weather.get("humidity") is not None:
                parts.append(f"{weather['humidity']:.0f}% humidity")
            if weather.get("wind_speed") is not None:
                parts.append(f"wind {weather['wind_speed']:.0f} km/h")
            if weather.get("precipitation") and weather["precipitation"] > 0:
                parts.append(f"{weather['precipitation']:.1f}mm rain")
            weather_str = ", ".join(parts)

        forecast_summary.append(
            f"  Day {forecast_data.index(f)+1} ({f['date']}): "
            f"AQI={f.get('predicted_aqi', 'N/A')}, "
            f"Category={f.get('category', 'N/A')}, "
            f"Dominant={f.get('dominant_pollutant', 'N/A')}, "
            f"Weather: {weather_str or 'N/A'}"
        )

    causes_str = "\n".join(
        f"  - {c['cause']} (confidence: {c['confidence']:.0%})"
        for c in causal_analysis[:5]
    )

    primary_str = ", ".join(primary_sources) if primary_sources else "various urban sources"

    prompt = f"""You are an air quality expert analyzing AQI data for {city}, India.

City Context: {city} is {city_context}. Its primary pollution sources are: {primary_str}.

7-Day AQI Forecast:
{chr(10).join(forecast_summary)}

Identified Pollution Causes:
{causes_str}

IMPORTANT: Base your analysis on the ACTUAL pollution sources for {city}.
Do NOT mention crop burning or stubble burning unless it appears in the identified causes above.
Focus on the real drivers: {primary_str}.

Write a concise 2-3 paragraph analysis that:
1. Summarizes the AQI trend over the forecast period (improving, worsening, stable)
2. Explains WHY the AQI is at this level, citing the specific causes identified and weather factors
3. Notes what to expect in coming days based on weather changes

Also briefly mention any current real-world factors affecting {city}'s air quality
(e.g., ongoing metro construction, industrial activity, traffic patterns, weather patterns).

Keep the tone informative but accessible. Use specific numbers from the data.
Do not use markdown headers or bullet points — write in flowing paragraphs."""

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        return response.text
    except Exception as e:
        print(f"Gemini API error: {e}")
        return _template_forecast_narrative(forecast_data, causal_analysis, city)


def generate_health_advisory(aqi: float, category: str,
                             causes: list[dict], city: str) -> str:
    """
    Generate a personalized health advisory using Gemini API.

    Falls back to template-based advisory if API is unavailable.
    """
    client = _get_client()
    if client is None:
        return _template_health_advisory(aqi, category, causes, city)

    causes_str = ", ".join(c["cause"] for c in causes[:3])

    # Get city context
    from src.analysis.causal import CITY_PROFILES
    city_profile = CITY_PROFILES.get(city, {})
    city_context = city_profile.get("context", "a major Indian city")

    prompt = f"""You are a public health advisor for {city}, India.
{city} is {city_context}.

Current AQI: {aqi} ({category})
Main causes: {causes_str}

Write a brief (3-4 sentences) conversational health advisory for residents.
Include: who should be most careful, what to do/avoid today, and one practical tip.
Be specific to the causes mentioned and the city's characteristics.
Do NOT mention crop burning or stubble burning unless it is listed in the main causes above.
Do not use bullet points."""

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        return response.text
    except Exception as e:
        print(f"Gemini API error: {e}")
        return _template_health_advisory(aqi, category, causes, city)


# === Template Fallbacks ===

def _template_forecast_narrative(forecast_data: list[dict],
                                 causal_analysis: list[dict],
                                 city: str) -> str:
    """Generate a template-based forecast narrative without LLM."""
    if not forecast_data:
        return f"Unable to generate forecast narrative for {city}. Insufficient data available."

    aqi_values = [f.get("predicted_aqi") for f in forecast_data if f.get("predicted_aqi")]
    if not aqi_values:
        return f"Forecast data for {city} is incomplete. Please try again later."

    avg_aqi = sum(aqi_values) / len(aqi_values)
    first_aqi = aqi_values[0]
    last_aqi = aqi_values[-1]

    # Trend
    if last_aqi > first_aqi * 1.15:
        trend = "worsening"
        trend_detail = f"rising from {first_aqi:.0f} to {last_aqi:.0f}"
    elif last_aqi < first_aqi * 0.85:
        trend = "improving"
        trend_detail = f"declining from {first_aqi:.0f} to {last_aqi:.0f}"
    else:
        trend = "relatively stable"
        trend_detail = f"hovering around {avg_aqi:.0f}"

    # Causes
    top_causes = [c["cause"] for c in causal_analysis[:3]]
    causes_text = ", ".join(top_causes) if top_causes else "general urban pollution"

    # Weather context
    weather_notes = []
    for f in forecast_data[:3]:
        w = f.get("weather", {})
        if w.get("precipitation") and w["precipitation"] > 2:
            weather_notes.append("Expected rainfall may help wash out particulates.")
            break
        if w.get("wind_speed") and w["wind_speed"] > 15:
            weather_notes.append("Strong winds are expected to disperse pollutants.")
            break
    if not weather_notes:
        for f in forecast_data[:3]:
            w = f.get("weather", {})
            if w.get("humidity") and w["humidity"] > 80:
                weather_notes.append(
                    "High humidity may trap pollutants near the surface."
                )
                break

    weather_text = " ".join(weather_notes) if weather_notes else ""

    para1 = (
        f"The air quality forecast for {city} over the next {len(forecast_data)} days "
        f"shows a {trend} trend, with AQI values {trend_detail}. The average predicted "
        f"AQI of {avg_aqi:.0f} falls in the range that warrants attention for sensitive groups."
    )

    para2 = (
        f"The primary factors contributing to current air quality levels include {causes_text}. "
        f"{weather_text}"
    )

    categories = [f.get("category") for f in forecast_data if f.get("category")]
    worst_cat = max(set(categories), key=categories.count) if categories else "Moderate"
    para3 = (
        f"Overall, the predominant AQI category for the forecast period is '{worst_cat}'. "
        f"Residents should monitor daily updates and plan outdoor activities during "
        f"lower-pollution hours."
    )

    return f"{para1}\n\n{para2}\n\n{para3}"


def _template_health_advisory(aqi: float, category: str,
                              causes: list[dict], city: str) -> str:
    """Generate a template-based health advisory without LLM."""
    top_cause = causes[0]["cause"] if causes else "general air pollution"

    if category in ("Good", "Satisfactory"):
        return (
            f"Air quality in {city} is currently {category.lower()} with an AQI of "
            f"{aqi:.0f}. This is a good day for outdoor activities. No special "
            f"precautions are needed for the general population."
        )
    elif category == "Moderate":
        return (
            f"Air quality in {city} is moderate (AQI: {aqi:.0f}), primarily due to "
            f"{top_cause.lower()}. People with respiratory conditions like asthma should "
            f"carry their medication. Consider reducing prolonged outdoor exertion if "
            f"you experience any discomfort."
        )
    elif category == "Poor":
        return (
            f"Air quality in {city} has deteriorated to Poor (AQI: {aqi:.0f}), largely "
            f"driven by {top_cause.lower()}. Everyone should limit extended outdoor "
            f"activities, especially between 6-10 AM. Wearing an N95 mask outdoors is "
            f"recommended. Keep windows closed and run air purifiers if available."
        )
    else:  # Very Poor / Severe
        return (
            f"Air quality in {city} is at {category} levels (AQI: {aqi:.0f}) — this is "
            f"a health emergency primarily caused by {top_cause.lower()}. Avoid all "
            f"outdoor exposure. Stay indoors with air purifiers running. Children, "
            f"elderly, and those with respiratory or heart conditions are at serious "
            f"risk. Seek medical attention if you experience breathing difficulty."
        )
