"""
Rule-based causal analysis for AQI readings.
Identifies likely pollution sources from pollutant signatures,
seasonal patterns, and city-specific knowledge.
"""
from datetime import datetime, date


# === Pollutant Signature Rules ===
# Each rule: (condition_fn, cause, base_confidence)
def _is_biomass(p):
    """High PM2.5 + PM10, low SO2/NO2 → crop burning / biomass."""
    pm25 = p.get("PM2.5", 0) or 0
    pm10 = p.get("PM10", 0) or 0
    so2 = p.get("SO2", 0) or 0
    no2 = p.get("NO2", 0) or 0
    return pm25 > 80 and pm10 > 120 and so2 < 40 and no2 < 50


def _is_vehicular(p):
    """High NO2 + CO, moderate PM2.5 → vehicular emissions."""
    no2 = p.get("NO2", 0) or 0
    co = p.get("CO", 0) or 0
    pm25 = p.get("PM2.5", 0) or 0
    return no2 > 60 and co > 2.0 and pm25 > 40


def _is_industrial(p):
    """High SO2 + PM10 → industrial emissions."""
    so2 = p.get("SO2", 0) or 0
    pm10 = p.get("PM10", 0) or 0
    return so2 > 60 and pm10 > 100


def _is_photochemical(p):
    """High O3, low PM → photochemical smog."""
    o3 = p.get("O3", 0) or 0
    pm25 = p.get("PM2.5", 0) or 0
    return o3 > 100 and pm25 < 60


def _is_stagnation(p):
    """All pollutants elevated → atmospheric stagnation / multiple sources."""
    pm25 = p.get("PM2.5", 0) or 0
    pm10 = p.get("PM10", 0) or 0
    no2 = p.get("NO2", 0) or 0
    so2 = p.get("SO2", 0) or 0
    return pm25 > 100 and pm10 > 150 and no2 > 50 and so2 > 40


def _is_dust(p):
    """Very high PM10 relative to PM2.5 → dust storm / construction."""
    pm25 = p.get("PM2.5", 0) or 0
    pm10 = p.get("PM10", 0) or 0
    if pm10 < 100:
        return False
    ratio = pm10 / max(pm25, 1)
    return ratio > 3.0


SIGNATURE_RULES = [
    (_is_biomass, "Crop Burning / Biomass Combustion", 0.80),
    (_is_vehicular, "Vehicular Emissions", 0.85),
    (_is_industrial, "Industrial Emissions", 0.75),
    (_is_photochemical, "Photochemical Smog", 0.70),
    (_is_dust, "Dust Storm / Construction Dust", 0.65),
    (_is_stagnation, "Atmospheric Stagnation (Multiple Sources)", 0.60),
]

# === Seasonal Overlays ===
SEASONAL_PATTERNS = {
    # (month_start, month_end): {cities: [...], cause, confidence_boost}
    (10, 11): {
        "cities": ["Delhi", "Lucknow", "Chandigarh", "Amritsar", "Patna"],
        "cause": "Seasonal Crop Burning (Punjab/Haryana stubble)",
        "confidence_boost": 0.15,
    },
    (12, 1): {
        "cities": None,  # All cities
        "cause": "Winter Temperature Inversions",
        "confidence_boost": 0.10,
    },
    (4, 6): {
        "cities": ["Jaipur", "Delhi", "Lucknow", "Chandigarh"],
        "cause": "Pre-Monsoon Dust Storms",
        "confidence_boost": 0.10,
    },
    (6, 9): {
        "cities": None,
        "cause": "Monsoon Washout (Lower AQI Expected)",
        "confidence_boost": 0.05,
    },
    (10, 11): {
        "cities": ["Delhi", "Kolkata", "Jaipur"],
        "cause": "Diwali Firecracker Emissions",
        "confidence_boost": 0.10,
    },
}

# === City-Specific Knowledge ===
CITY_PROFILES = {
    "Delhi": ["Vehicular Emissions", "Crop Burning / Biomass Combustion",
              "Construction Dust", "Industrial Emissions"],
    "Mumbai": ["Vehicular Emissions", "Industrial Emissions", "Construction Dust"],
    "Kolkata": ["Industrial Emissions", "Vehicular Emissions"],
    "Bengaluru": ["Vehicular Emissions", "Construction Dust"],
    "Chennai": ["Vehicular Emissions", "Industrial Emissions"],
    "Hyderabad": ["Vehicular Emissions", "Industrial Emissions"],
    "Ahmedabad": ["Industrial Emissions", "Vehicular Emissions", "Dust Storm / Construction Dust"],
    "Lucknow": ["Vehicular Emissions", "Crop Burning / Biomass Combustion"],
    "Patna": ["Crop Burning / Biomass Combustion", "Vehicular Emissions"],
    "Jaipur": ["Dust Storm / Construction Dust", "Vehicular Emissions"],
    "Chandigarh": ["Crop Burning / Biomass Combustion", "Vehicular Emissions"],
    "Amritsar": ["Crop Burning / Biomass Combustion", "Vehicular Emissions"],
    "Bhopal": ["Vehicular Emissions", "Industrial Emissions"],
    "Gurugram": ["Vehicular Emissions", "Construction Dust", "Industrial Emissions"],
    "Visakhapatnam": ["Industrial Emissions", "Vehicular Emissions"],
}


def analyze_causes(pollutants: dict, city: str = None,
                   dt: date | datetime = None) -> list[dict]:
    """
    Identify likely causes of current AQI based on pollutant signatures,
    seasonal patterns, and city knowledge.

    Args:
        pollutants: Dict of pollutant name → concentration value
        city: City name (optional, for city-specific analysis)
        dt: Date for seasonal context (defaults to today)

    Returns:
        List of dicts: [{"cause": str, "confidence": float, "type": str}, ...]
        Sorted by confidence descending.
    """
    if dt is None:
        dt = datetime.now()
    if isinstance(dt, datetime):
        month = dt.month
    else:
        month = dt.month

    causes = []

    # 1. Check pollutant signature rules
    for check_fn, cause, base_conf in SIGNATURE_RULES:
        if check_fn(pollutants):
            causes.append({
                "cause": cause,
                "confidence": base_conf,
                "type": "pollutant_signature",
            })

    # 2. Apply seasonal overlays
    for (m_start, m_end), pattern in SEASONAL_PATTERNS.items():
        in_season = False
        if m_start <= m_end:
            in_season = m_start <= month <= m_end
        else:  # Wraps around year (e.g., Dec-Jan)
            in_season = month >= m_start or month <= m_end

        if not in_season:
            continue

        city_match = pattern["cities"] is None or (city and city in pattern["cities"])
        if city_match:
            # Boost existing matching causes or add new
            matched = False
            for c in causes:
                if pattern["cause"].split("(")[0].strip() in c["cause"]:
                    c["confidence"] = min(0.95, c["confidence"] + pattern["confidence_boost"])
                    matched = True
                    break
            if not matched:
                causes.append({
                    "cause": pattern["cause"],
                    "confidence": 0.5 + pattern["confidence_boost"],
                    "type": "seasonal",
                })

    # 3. City-specific knowledge boost
    if city and city in CITY_PROFILES:
        known_sources = CITY_PROFILES[city]
        for c in causes:
            if c["cause"] in known_sources:
                c["confidence"] = min(0.95, c["confidence"] + 0.05)

    # 4. If no causes identified, provide a generic assessment
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
        dict with keys: health, actions, vulnerable_groups, policy
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
    if any("Dust" in c for c in cause_names):
        actions_personal.append("Keep windows closed during windy/dusty conditions.")
    if any("Crop Burning" in c or "Biomass" in c for c in cause_names):
        actions_personal.append(
            "Monitor stubble burning alerts and plan outdoor activities accordingly."
        )

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
