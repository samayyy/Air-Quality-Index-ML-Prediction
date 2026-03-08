#!/bin/bash
# ============================================================
#  AQI Prediction App — One-Click Setup Script
#  Run: chmod +x setup.sh && ./setup.sh
# ============================================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo ""
echo "======================================================"
echo "  India AQI Prediction — Setup Script"
echo "======================================================"
echo ""

# ------- 1. Check prerequisites -------
echo -e "${YELLOW}[1/6] Checking prerequisites...${NC}"

# Check conda
if ! command -v conda &> /dev/null; then
    echo -e "${RED}ERROR: conda is not installed.${NC}"
    echo "Install Miniconda from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi
echo "  - conda: OK"

# Check git
if ! command -v git &> /dev/null; then
    echo -e "${RED}ERROR: git is not installed.${NC}"
    exit 1
fi
echo "  - git: OK"

# ------- 2. Create conda environment -------
echo ""
echo -e "${YELLOW}[2/6] Creating conda environment (aqi-prediction)...${NC}"
echo "  This may take 5-10 minutes on first run."

if conda env list | grep -q "aqi-prediction"; then
    echo "  Environment 'aqi-prediction' already exists. Updating..."
    conda env update -f environment.yml --prune -q
else
    conda env create -f environment.yml -q
fi
echo -e "${GREEN}  Environment ready.${NC}"

# ------- 3. Download Kaggle dataset -------
echo ""
echo -e "${YELLOW}[3/6] Setting up data...${NC}"

DATA_RAW="data/raw"
mkdir -p "$DATA_RAW"

if [ -f "$DATA_RAW/city_day.csv" ]; then
    echo -e "${GREEN}  Dataset already exists. Skipping download.${NC}"
else
    echo "  Downloading dataset from Kaggle..."
    echo ""
    echo "  You have two options:"
    echo "  (a) Auto-download via kagglehub (requires Kaggle account)"
    echo "  (b) Manual download"
    echo ""

    # Try kagglehub first
    if conda run -n aqi-prediction python -c "
import kagglehub
path = kagglehub.dataset_download('rohanrao/air-quality-data-in-india')
print(path)
" 2>/dev/null; then
        # Copy files from kagglehub cache to data/raw
        KAGGLE_PATH=$(conda run -n aqi-prediction python -c "
import kagglehub
path = kagglehub.dataset_download('rohanrao/air-quality-data-in-india')
print(path)
" 2>/dev/null)
        if [ -d "$KAGGLE_PATH" ]; then
            cp "$KAGGLE_PATH"/*.csv "$DATA_RAW/"
            echo -e "${GREEN}  Dataset downloaded successfully.${NC}"
        fi
    else
        echo -e "${YELLOW}  Auto-download failed. Please download manually:${NC}"
        echo ""
        echo "  1. Go to: https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india"
        echo "  2. Click 'Download' (requires free Kaggle account)"
        echo "  3. Extract the ZIP and copy all CSV files to: $(pwd)/data/raw/"
        echo ""
        echo "  Required files: city_day.csv, city_hour.csv, station_day.csv, station_hour.csv, stations.csv"
        echo ""
        read -p "  Press ENTER after placing the files, or Ctrl+C to exit... "

        if [ ! -f "$DATA_RAW/city_day.csv" ]; then
            echo -e "${RED}  ERROR: city_day.csv not found in data/raw/. Cannot continue.${NC}"
            exit 1
        fi
    fi
fi

# ------- 4. Run preprocessing & train models -------
echo ""
echo -e "${YELLOW}[4/6] Processing data & training models...${NC}"

PROCESSED="data/processed"
MODELS="models/tuned"

if [ -f "$PROCESSED/city_day_clean.parquet" ] && [ -f "$MODELS/best_regression.joblib" ]; then
    echo -e "${GREEN}  Processed data and trained models already exist. Skipping.${NC}"
    echo "  (Delete data/processed/ and models/ to retrain from scratch)"
else
    echo "  Running notebooks to process data and train models..."
    echo "  This will take 15-30 minutes depending on your machine."
    echo ""

    mkdir -p "$PROCESSED" models/regression models/classification models/deep_learning models/tuned

    # Run notebooks in order
    NOTEBOOKS=(
        "03_preprocessing.ipynb"
        "04_regression_models.ipynb"
        "05_classification_models.ipynb"
        "07_optuna_tuning.ipynb"
    )

    for nb in "${NOTEBOOKS[@]}"; do
        if [ -f "notebooks/$nb" ]; then
            echo "  Running $nb..."
            conda run -n aqi-prediction jupyter nbconvert --to notebook --execute \
                --ExecutePreprocessor.timeout=600 \
                "notebooks/$nb" --output /dev/null 2>&1 || {
                echo -e "${YELLOW}  Warning: $nb had issues, continuing...${NC}"
            }
        fi
    done

    echo -e "${GREEN}  Model training complete.${NC}"
fi

# ------- 5. Setup API keys -------
echo ""
echo -e "${YELLOW}[5/6] Setting up API keys...${NC}"

if [ -f ".env" ]; then
    echo -e "${GREEN}  .env file already exists.${NC}"
else
    cp .env.example .env
    echo ""
    echo "  Created .env file. You need to add your API keys:"
    echo ""

    # WAQI token
    echo "  1. WAQI API Token (Required for live predictions & forecasts)"
    echo "     Get free token at: https://aqicn.org/data-platform/token/"
    read -p "     Enter your WAQI token (or press ENTER to skip): " WAQI_TOKEN
    if [ -n "$WAQI_TOKEN" ]; then
        if [[ "$OSTYPE" == "darwin"* ]]; then
            sed -i '' "s/your_token_here/$WAQI_TOKEN/" .env
        else
            sed -i "s/your_token_here/$WAQI_TOKEN/" .env
        fi
        echo -e "${GREEN}     WAQI token saved.${NC}"
    else
        echo -e "${YELLOW}     Skipped. Live prediction & forecast pages won't work without this.${NC}"
    fi

    echo ""

    # Gemini key
    echo "  2. Google Gemini API Key (Optional — for AI-powered insights)"
    echo "     Get free key at: https://aistudio.google.com/apikey"
    read -p "     Enter your Gemini key (or press ENTER to skip): " GEMINI_KEY
    if [ -n "$GEMINI_KEY" ]; then
        if [[ "$OSTYPE" == "darwin"* ]]; then
            sed -i '' "s/your_key_here/$GEMINI_KEY/" .env
        else
            sed -i "s/your_key_here/$GEMINI_KEY/" .env
        fi
        echo -e "${GREEN}     Gemini key saved.${NC}"
    else
        echo -e "${YELLOW}     Skipped. AI insights will use template-based fallback.${NC}"
    fi
fi

# ------- 6. Launch the app -------
echo ""
echo -e "${YELLOW}[6/6] Launching Streamlit app...${NC}"
echo ""
echo "======================================================"
echo -e "${GREEN}  Setup complete! Starting the app...${NC}"
echo "======================================================"
echo ""
echo "  The app will open at: http://localhost:8501"
echo "  Press Ctrl+C to stop the server."
echo ""

conda run -n aqi-prediction streamlit run app/app.py --server.headless true
