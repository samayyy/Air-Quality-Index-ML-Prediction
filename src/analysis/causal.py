"""
Rule-based causal analysis for AQI readings.
Identifies likely pollution sources from pollutant signatures,
seasonal patterns, city-specific knowledge, and real-time news context.
"""
import os
import requests
from datetime import datetime, date


# === Cities where crop burning is a relevant cause ===
CROP_BURNING_CITIES = {
    "Delhi", "Lucknow", "Chandigarh", "Amritsar", "Patna",
    "Gurugram", "Jaipur",
}

# === City-Specific Knowledge ===
CITY_PROFILES = {
    "Delhi": {
        "primary_sources": ["Vehicular Emissions", "Construction Activity", "Industrial Emissions"],
        "seasonal_sources": ["Crop Burning (Punjab/Haryana Stubble)"],
        "context": "NCR region with heavy traffic, ongoing construction, and seasonal crop burning impact",
    },
    "Mumbai": {
        "primary_sources": ["Vehicular Emissions", "Construction Activity", "Industrial Emissions"],
        "seasonal_sources": [],
        "context": "Coastal megacity with heavy traffic, port/industrial activity, and massive construction",
    },
    "Kolkata": {
        "primary_sources": ["Industrial Emissions", "Vehicular Emissions", "Construction Activity"],
        "seasonal_sources": [],
        "context": "Industrial belt city with dense traffic and coal-based power generation nearby",
    },
    "Bengaluru": {
        "primary_sources": ["Vehicular Emissions", "Construction Activity"],
        "seasonal_sources": [],
        "context": "Rapidly growing IT hub with heavy construction and increasing vehicle density",
    },
    "Chennai": {
        "primary_sources": ["Vehicular Emissions", "Industrial Emissions"],
        "seasonal_sources": [],
        "context": "Industrial corridor with petrochemical and automobile manufacturing",
    },
    "Hyderabad": {
        "primary_sources": ["Vehicular Emissions", "Industrial Emissions", "Construction Activity"],
        "seasonal_sources": [],
        "context": "Fast-growing city with pharmaceutical and IT industry expansion",
    },
    "Ahmedabad": {
        "primary_sources": ["Industrial Emissions", "Vehicular Emissions", "Dust / Construction"],
        "seasonal_sources": [],
        "context": "Major industrial center with textile, chemical, and pharmaceutical industries",
    },
    "Lucknow": {
        "primary_sources": ["Vehicular Emissions", "Construction Activity"],
        "seasonal_sources": ["Crop Burning (Punjab/Haryana Stubble)"],
        "context": "Growing city affected by regional crop burning in Oct-Nov",
    },
    "Patna": {
        "primary_sources": ["Vehicular Emissions", "Biomass Burning (Domestic)"],
        "seasonal_sources": ["Crop Burning (Punjab/Haryana Stubble)"],
        "context": "Gangetic plain city with domestic biomass use and crop burning impact",
    },
    "Jaipur": {
        "primary_sources": ["Dust / Construction", "Vehicular Emissions"],
        "seasonal_sources": ["Pre-Monsoon Dust Storms"],
        "context": "Arid region city prone to natural dust and construction dust",
    },
    "Chandigarh": {
        "primary_sources": ["Vehicular Emissions"],
        "seasonal_sources": ["Crop Burning (Punjab/Haryana Stubble)"],
        "context": "Planned city directly affected by Punjab/Haryana agricultural burning",
    },
    "Amritsar": {
        "primary_sources": ["Vehicular Emissions"],
        "seasonal_sources": ["Crop Burning (Punjab/Haryana Stubble)"],
        "context": "Punjab city at the epicenter of stubble burning zone",
    },
    "Bhopal": {
        "primary_sources": ["Vehicular Emissions", "Industrial Emissions"],
        "seasonal_sources": [],
        "context": "City with chemical and manufacturing industry presence",
    },
    "Gurugram": {
        "primary_sources": ["Vehicular Emissions", "Construction Activity", "Industrial Emissions"],
        "seasonal_sources": ["Crop Burning (Punjab/Haryana Stubble)"],
        "context": "NCR satellite city with heavy construction and traffic congestion",
    },
    "Visakhapatnam": {
        "primary_sources": ["Industrial Emissions", "Vehicular Emissions"],
        "seasonal_sources": [],
        "context": "Port city with steel plant, refineries, and industrial activity",
    },
}


def _identify_from_pollutants(pollutants: dict, city: str = None) -> list[dict]:
    """
    Identify pollution causes from pollutant concentration patterns.
    City-aware: avoids attributing crop burning to non-agricultural cities.
    """
    causes = []
    pm25 = pollutants.get("PM2.5", 0) or 0
    pm10 = pollutants.get("PM10", 0) or 0
    no2 = pollutants.get("NO2", 0) or 0
    so2 = pollutants.get("SO2", 0) or 0
    co = pollutants.get("CO", 0) or 0
    o3 = pollutants.get("O3", 0) or 0

    # --- Vehicular Emissions ---
    # NO2 and CO are primary markers of traffic; PM2.5 is secondary
    if no2 > 40 or co > 1.5:
        conf = 0.60
        if no2 > 60:
            conf += 0.10
        if co > 2.5:
            conf += 0.10
        if pm25 > 40:
            conf += 0.05
        causes.append({
            "cause": "Vehicular Emissions",
            "confidence": min(0.95, conf),
            "type": "pollutant_signature",
        })

    # --- Industrial Emissions ---
    # High SO2 is the key industrial marker
    if so2 > 40:
        conf = 0.60
        if so2 > 80:
            conf += 0.15
        if pm10 > 100:
            conf += 0.05
        causes.append({
            "cause": "Industrial Emissions",
            "confidence": min(0.95, conf),
            "type": "pollutant_signature",
        })

    # --- Construction / Dust ---
    # High PM10 relative to PM2.5 (coarse particles from dust/construction)
    if pm10 > 80:
        ratio = pm10 / max(pm25, 1)
        if ratio > 2.0:
            conf = 0.55
            if ratio > 3.0:
                conf += 0.10
            if pm10 > 150:
                conf += 0.10
            causes.append({
                "cause": "Construction Activity / Road Dust",
                "confidence": min(0.90, conf),
                "type": "pollutant_signature",
            })

    # --- Crop Burning / Biomass (ONLY for relevant cities) ---
    # High PM2.5+PM10 with relatively low SO2/NO2 (biomass doesn't produce much SO2)
    is_crop_burning_city = city in CROP_BURNING_CITIES if city else False
    if pm25 > 80 and pm10 > 120 and so2 < 40 and no2 < 60:
        if is_crop_burning_city:
            causes.append({
                "cause": "Crop Burning / Biomass Combustion",
                "confidence": 0.70,
                "type": "pollutant_signature",
            })
        else:
            # For non-agricultural cities, this pattern = general urban PM pollution
            causes.append({
                "cause": "Urban Particulate Pollution (Multiple Sources)",
                "confidence": 0.65,
                "type": "pollutant_signature",
            })

    # --- Photochemical Smog ---
    if o3 > 100 and pm25 < 60:
        causes.append({
            "cause": "Photochemical Smog (Ground-Level Ozone)",
            "confidence": 0.70,
            "type": "pollutant_signature",
        })

    # --- Atmospheric Stagnation ---
    if pm25 > 100 and pm10 > 150 and no2 > 50 and so2 > 30:
        causes.append({
            "cause": "Atmospheric Stagnation (Trapped Pollutants)",
            "confidence": 0.60,
            "type": "pollutant_signature",
        })

    return causes


def _apply_seasonal_context(causes: list[dict], city: str, month: int) -> list[dict]:
    """Apply seasonal overlays — boost or add seasonal causes."""

    # Oct-Nov: Crop burning season (only for northern cities)
    if month in (10, 11) and city in CROP_BURNING_CITIES:
        _boost_or_add(causes, "Crop Burning", "Seasonal Crop Burning (Oct-Nov stubble burning)", 0.15)

    # Dec-Jan: Winter inversions everywhere
    if month in (12, 1):
        _boost_or_add(causes, "Inversion", "Winter Temperature Inversions (trapping pollutants)", 0.10)

    # Apr-Jun: Dust storms in arid cities
    if month in (4, 5, 6) and city in ("Jaipur", "Delhi", "Lucknow", "Chandigarh", "Ahmedabad"):
        _boost_or_add(causes, "Dust", "Pre-Monsoon Dust Storms", 0.10)

    # Jun-Sep: Monsoon washout
    if month in (6, 7, 8, 9):
        _boost_or_add(causes, "Monsoon", "Monsoon Washout (Improved Air Quality)", 0.05)

    # Diwali season (typically Oct-Nov)
    if month in (10, 11) and city in ("Delhi", "Kolkata", "Jaipur", "Mumbai", "Lucknow"):
        _boost_or_add(causes, "Firecracker", "Diwali Firecracker Emissions", 0.08)

    return causes


def _boost_or_add(causes: list[dict], keyword: str, full_cause: str, boost: float):
    """Boost confidence of existing matching cause, or add a new one."""
    for c in causes:
        if keyword.lower() in c["cause"].lower():
            c["confidence"] = min(0.95, c["confidence"] + boost)
            return
    causes.append({
        "cause": full_cause,
        "confidence": 0.45 + boost,
        "type": "seasonal",
    })


def _apply_city_knowledge(causes: list[dict], city: str) -> list[dict]:
    """Boost causes that match known city pollution profile."""
    if city not in CITY_PROFILES:
        return causes

    profile = CITY_PROFILES[city]
    known = profile["primary_sources"] + profile.get("seasonal_sources", [])

    for c in causes:
        for known_source in known:
            # Partial match (e.g., "Vehicular" matches "Vehicular Emissions")
            if known_source.split()[0] in c["cause"] or c["cause"].split()[0] in known_source:
                c["confidence"] = min(0.95, c["confidence"] + 0.05)
                break

    return causes


def fetch_aqi_news(city: str, max_results: int = 3) -> list[dict]:
    """
    Fetch recent AQI-related news for a city using web search.
    Returns list of {title, snippet, source} dicts.

    Uses Google Custom Search API if GOOGLE_SEARCH_API_KEY is available,
    otherwise returns empty list (graceful degradation).
    """
    # Try using Gemini to generate context (reuse existing key)
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return []

    try:
        from google import genai
        client = genai.Client(api_key=api_key)

        prompt = (
            f"What are the current main causes of air pollution in {city}, India "
            f"as of today ({datetime.now().strftime('%B %Y')})? "
            f"List the top 3-4 specific causes with brief explanations. "
            f"Focus on: recent news events, ongoing construction projects, "
            f"industrial activity, traffic conditions, seasonal factors, "
            f"and any government advisories. "
            f"Format each as a short bullet point. Be factual and specific to {city}."
        )

        from config import GEMINI_MODEL
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )

        if response and response.text:
            return [{"title": f"AI-sourced context for {city}", "snippet": response.text, "source": "Gemini AI"}]
    except Exception:
        pass

    return []


def analyze_causes(pollutants: dict, city: str = None,
                   dt: date | datetime = None) -> list[dict]:
    """
    Identify likely causes of current AQI based on pollutant signatures,
    seasonal patterns, city knowledge, and news context.

    Args:
        pollutants: Dict of pollutant name -> concentration value
        city: City name (optional, for city-specific analysis)
        dt: Date for seasonal context (defaults to today)

    Returns:
        List of dicts: [{"cause": str, "confidence": float, "type": str}, ...]
        Sorted by confidence descending.
    """
    if dt is None:
        dt = datetime.now()
    month = dt.month

    # 1. Identify from pollutant patterns (city-aware)
    causes = _identify_from_pollutants(pollutants, city)

    # 2. Apply seasonal context
    if city:
        causes = _apply_seasonal_context(causes, city, month)

    # 3. Apply city-specific knowledge boost
    if city:
        causes = _apply_city_knowledge(causes, city)

    # 4. If no causes identified, provide generic assessment
    if not causes:
        pm25 = pollutants.get("PM2.5", 0) or 0
        if pm25 > 60:
            causes.append({
                "cause": "General Urban Air Pollution",
                "confidence": 0.50,
                "type": "generic",
            })
        else:
            causes.append({
                "cause": "Normal Background Levels",
                "confidence": 0.70,
                "type": "generic",
            })

    # Sort by confidence
    causes.sort(key=lambda x: x["confidence"], reverse=True)
    return causes


def get_recommendations(aqi: float, causes: list[dict],
                        category: str = None) -> dict:
    """
    Generate health advisories and actionable recommendations based on
    AQI level, identified causes, and category.

    Returns:
        dict with keys: health, actions_personal, actions_community,
        actions_policy, vulnerable_groups, category
    """
    if category is None:
        if aqi <= 50:
            category = "Good"
        elif aqi <= 100:
            category = "Satisfactory"
        elif aqi <= 200:
            category = "Moderate"
        elif aqi <= 300:
            category = "Poor"
        elif aqi <= 400:
            category = "Very Poor"
        else:
            category = "Severe"

    # --- Health advisories by category ---
    health_map = {
        "Good": [
            "Air quality is satisfactory — enjoy outdoor activities.",
            "No special precautions needed for any group.",
        ],
        "Satisfactory": [
            "Air quality is acceptable for most people.",
            "Unusually sensitive individuals may experience minor discomfort.",
        ],
        "Moderate": [
            "People with respiratory or heart conditions may experience discomfort.",
            "Limit prolonged outdoor exertion if you feel symptoms.",
            "Consider wearing an N95 mask during outdoor exercise.",
        ],
        "Poor": [
            "Most people may experience breathing discomfort.",
            "Wear an N95/KN95 mask when outdoors.",
            "Avoid outdoor exercise, especially during morning (6-10 AM) and evening hours.",
            "Keep windows closed and use air purifiers indoors.",
        ],
        "Very Poor": [
            "Serious health effects possible for everyone.",
            "Mandatory N95/KN95 mask usage outdoors.",
            "Avoid all outdoor physical activity.",
            "Use HEPA air purifiers indoors; keep all windows sealed.",
            "Consider working from home if possible.",
        ],
        "Severe": [
            "Health emergency — avoid any outdoor exposure.",
            "Stay indoors with air purifiers running continuously.",
            "N95 mask mandatory even for brief outdoor exposure.",
            "Seek medical attention if experiencing breathing difficulty.",
            "Schools and outdoor workplaces should consider closure.",
        ],
    }

    # --- Actionable recommendations ---
    actions_personal = []
    actions_community = []
    actions_policy = []

    cause_names = [c["cause"] for c in causes]

    # Personal actions
    if category in ("Moderate", "Poor", "Very Poor", "Severe"):
        actions_personal.extend([
            "Use public transport or carpool to reduce vehicular emissions.",
            "Avoid burning waste, incense, or candles indoors.",
            "Stay hydrated — drink warm fluids to soothe airways.",
        ])
    if category in ("Poor", "Very Poor", "Severe"):
        actions_personal.extend([
            "Avoid outdoor exercise between 6-10 AM when pollutants concentrate.",
            "Use indoor HEPA air purifiers in bedrooms and living areas.",
            "Check AQI before planning any outdoor activities.",
        ])

    # Cause-specific personal actions
    if any("Vehicular" in c for c in cause_names):
        actions_personal.append("Avoid walking or cycling near busy roads during peak traffic.")
    if any("Dust" in c or "Construction" in c for c in cause_names):
        actions_personal.append("Keep windows closed during dusty conditions; wet-mop floors regularly.")
    if any("Crop Burning" in c or "Biomass" in c for c in cause_names):
        actions_personal.append(
            "Monitor stubble burning alerts and plan outdoor activities accordingly."
        )
    if any("Industrial" in c for c in cause_names):
        actions_personal.append("Avoid areas near industrial zones, especially downwind.")

    # Community actions
    if category in ("Poor", "Very Poor", "Severe"):
        actions_community.extend([
            "Organize community awareness campaigns about air quality.",
            "Support local tree-planting and urban greening initiatives.",
            "Report illegal waste burning to local authorities.",
        ])

    # Policy-level
    if category in ("Very Poor", "Severe"):
        actions_policy.extend([
            "Implement odd-even vehicle rationing scheme.",
            "Increase monitoring of industrial emission compliance.",
            "Enforce construction dust suppression regulations.",
            "Consider temporary restrictions on heavy diesel vehicles.",
        ])
    if any("Crop Burning" in c or "Biomass" in c for c in cause_names):
        actions_policy.append(
            "Provide subsidized alternatives to crop stubble burning for farmers."
        )
    if any("Construction" in c for c in cause_names):
        actions_policy.append(
            "Enforce mandatory dust suppression (water sprinklers, covers) at construction sites."
        )

    # --- Vulnerable groups ---
    vulnerable = {
        "children": "Children breathe faster and absorb more pollutants. Keep them indoors during poor AQI days. Ensure school playgrounds have AQI monitoring.",
        "elderly": "Seniors with pre-existing heart or lung conditions are at highest risk. Ensure medication is accessible. Avoid morning walks when AQI > 200.",
        "respiratory": "People with asthma, COPD, or bronchitis should keep rescue inhalers handy. Consult doctor about adjusting medication during high-AQI periods.",
        "pregnant": "Pregnant women should minimize outdoor exposure during Poor+ AQI. Air pollution is linked to low birth weight and preterm delivery.",
        "outdoor_workers": "Workers with outdoor occupations should use N95 masks, take frequent breaks in clean-air spaces, and stay hydrated.",
    }

    return {
        "health": health_map.get(category, health_map["Moderate"]),
        "actions_personal": actions_personal,
        "actions_community": actions_community,
        "actions_policy": actions_policy,
        "vulnerable_groups": vulnerable if category in ("Poor", "Very Poor", "Severe") else {},
        "category": category,
    }
