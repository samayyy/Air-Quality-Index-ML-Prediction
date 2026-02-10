"""
Central configuration for the India AQI Prediction project.
All paths, constants, hyperparameters, and CPCB AQI breakpoints are defined here.
"""
from pathlib import Path

# === Paths ===
PROJECT_ROOT = Path(__file__).parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_EXTERNAL = PROJECT_ROOT / "data" / "external"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# === Dataset ===
KAGGLE_DATASET = "rohanrao/air-quality-data-in-india"
RAW_CSV = DATA_RAW / "city_day.csv"  # Primary: 2015-2020 (26 cities, CPCB verified)

# === Pollutant Columns ===
POLLUTANT_COLS = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3",
                  "Benzene", "Toluene", "Xylene"]
AQI_POLLUTANTS = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "NH3"]

# === CPCB AQI Breakpoints ===
# Format: pollutant -> list of (C_low, C_high, I_low, I_high)
AQI_BREAKPOINTS = {
    "PM2.5": [
        (0, 30, 0, 50), (31, 60, 51, 100), (61, 90, 101, 200),
        (91, 120, 201, 300), (121, 250, 301, 400), (250, 380, 401, 500),
    ],
    "PM10": [
        (0, 50, 0, 50), (51, 100, 51, 100), (101, 250, 101, 200),
        (251, 350, 201, 300), (351, 430, 301, 400), (430, 510, 401, 500),
    ],
    "NO2": [
        (0, 40, 0, 50), (41, 80, 51, 100), (81, 180, 101, 200),
        (181, 280, 201, 300), (281, 400, 301, 400), (400, 520, 401, 500),
    ],
    "SO2": [
        (0, 40, 0, 50), (41, 80, 51, 100), (81, 380, 101, 200),
        (381, 800, 201, 300), (801, 1600, 301, 400), (1600, 2100, 401, 500),
    ],
    "CO": [
        (0, 1.0, 0, 50), (1.1, 2.0, 51, 100), (2.1, 10, 101, 200),
        (10, 17, 201, 300), (17, 34, 301, 400), (34, 46, 401, 500),
    ],
    "O3": [
        (0, 50, 0, 50), (51, 100, 51, 100), (101, 168, 101, 200),
        (169, 208, 201, 300), (209, 748, 301, 400), (748, 940, 401, 500),
    ],
    "NH3": [
        (0, 200, 0, 50), (201, 400, 51, 100), (401, 800, 101, 200),
        (801, 1200, 201, 300), (1201, 1800, 301, 400), (1800, 2400, 401, 500),
    ],
}

AQI_CATEGORIES = {
    "Good": (0, 50),
    "Satisfactory": (51, 100),
    "Moderate": (101, 200),
    "Poor": (201, 300),
    "Very Poor": (301, 400),
    "Severe": (401, 500),
}

# === Feature Engineering ===
TEMPORAL_FEATURES = ["year", "month", "day_of_week", "day_of_year", "quarter", "is_weekend"]
INDIA_SEASONS = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Summer", 4: "Summer", 5: "Summer",
    6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
    10: "Post-Monsoon", 11: "Post-Monsoon",
}
LAG_DAYS = [1, 2, 3, 7, 14, 30]
ROLLING_WINDOWS = [3, 7, 14, 30]

# === Model Training ===
RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15
N_OPTUNA_TRIALS = 50
EARLY_STOPPING_ROUNDS = 50
