# AQI Prediction App — Setup Guide

## Quick Start (One Command)

```bash
chmod +x setup.sh && ./setup.sh
```

This interactive script handles everything: environment, data, models, API keys, and launches the app.

---

## Manual Setup

### Prerequisites
- **Conda** (Miniconda or Anaconda) — [Install](https://docs.conda.io/en/latest/miniconda.html)
- **Git**

### Step 1: Create Environment
```bash
conda env create -f environment.yml
```

### Step 2: Download Dataset
Download from [Kaggle](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india) and place all CSV files in `data/raw/`:
```
data/raw/
├── city_day.csv       (required)
├── city_hour.csv
├── station_day.csv
├── station_hour.csv
└── stations.csv
```

### Step 3: Process Data & Train Models
Run these notebooks in order (inside `notebooks/` directory):
```bash
conda run -n aqi-prediction jupyter lab
```
1. `01_data_download.ipynb` — Data inspection
2. `02_eda.ipynb` — Exploratory analysis
3. `03_preprocessing.ipynb` — Cleaning & feature engineering
4. `04_regression_models.ipynb` — Train regression models
5. `05_classification_models.ipynb` — Train classification models
6. `06_deep_learning.ipynb` — Deep learning models (optional)
7. `07_optuna_tuning.ipynb` — Hyperparameter tuning (produces final models)
8. `08_shap_analysis.ipynb` — Model explainability
9. `09_final_comparison.ipynb` — Model comparison

**Minimum required**: Notebooks 03, 04, 05, 07 (for the app to work).

### Step 4: API Keys
```bash
cp .env.example .env
```
Edit `.env` and add:

| Key | Required? | Where to get it |
|-----|-----------|----------------|
| `WAQI_API_TOKEN` | Yes (for live data) | [aqicn.org/data-platform/token](https://aqicn.org/data-platform/token/) |
| `GEMINI_API_KEY` | Optional (AI insights) | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |

### Step 5: Launch
```bash
conda run -n aqi-prediction streamlit run app/app.py --server.headless true
```
Open http://localhost:8501

---

## App Pages

| # | Page | Description |
|---|------|-------------|
| 1 | Project Overview | Architecture, methodology |
| 2 | EDA Visualizations | Interactive data exploration |
| 3 | Model Comparison | ML model performance metrics |
| 4 | Live Prediction | Real-time AQI prediction for any city |
| 5 | City Analysis | Historical city-wise trends |
| 6 | Explainability | SHAP-based feature importance |
| 7 | Forecast & Insights | 7-day forecast, causal analysis, AI insights |

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError` | Run: `conda run -n aqi-prediction pip install <module>` |
| Live prediction shows "No data" | Check WAQI_API_TOKEN in `.env` |
| AI insights show template text | Check GEMINI_API_KEY in `.env` |
| Models not found | Run notebooks 03 → 04 → 05 → 07 |
| Port 8501 in use | Use `--server.port 8502` |

## Tech Stack
- **ML**: LightGBM, XGBoost, CatBoost (tuned with Optuna)
- **APIs**: WAQI (air quality), Open-Meteo (weather), Google Gemini (AI insights)
- **Frontend**: Streamlit + Plotly
- **Data**: CPCB India AQI dataset (2015-2020, 26 cities)
