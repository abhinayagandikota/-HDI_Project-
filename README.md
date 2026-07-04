# Human Development Index (HDI) Prediction System

An end-to-end Machine Learning web application that predicts a country's HDI category (Very High, High, Medium, Low) based on key development indicators.

## Dataset

The HDI is a composite index published by the UNDP. This project uses a synthetic dataset (~400 countries) generated from realistic distributions based on:

- **Life Expectancy** at birth (years)
- **Mean Years of Schooling**
- **Expected Years of Schooling**
- **GNI per Capita** (USD, PPP)

**Reliable real-world sources:**
- [UNDP Human Development Data](https://hdr.undp.org/data-center)
- [World Bank Open Data](https://data.worldbank.org)
- [Kaggle - HDI Dataset](https://www.kaggle.com/datasets/elmartini/human-development-index-hdi)

## Project Structure

```
HDI_Project/
├── data/
│   ├── hdi_data.csv               # Training dataset
│   └── generate_dataset.py        # Synthetic data generator
├── notebooks/                     # Jupyter notebooks (optional)
├── src/
│   ├── __init__.py
│   ├── preprocessing.py           # Data cleaning, scaling, encoding
│   ├── model.py                   # Model definitions
│   ├── train.py                   # Training + evaluation + viz pipeline
│   └── artifacts/                 # Saved models (pickle)
│       ├── hdi_model.pkl
│       ├── scaler.pkl
│       └── encoder.pkl
├── static/
│   ├── style.css                  # UI styling
│   └── viz/                       # Generated visualizations
├── templates/
│   └── index.html                 # Flask frontend
├── app.py                         # Flask web application
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Generate Dataset & Train Model

```bash
python data/generate_dataset.py
python src/train.py
```

This will:
- Generate a synthetic HDI dataset
- Train Logistic Regression, Random Forest, and Decision Tree models
- Pick the best model and save it along with the scaler and label encoder
- Save visualizations to `static/viz/`

### 2. Run the Web App

```bash
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

### 3. Make Predictions

Enter the four indicators and click **Predict HDI Category**.

### API Endpoint

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"life_expectancy": 82.3, "mean_schooling_years": 13.1, "expected_schooling_years": 18.7, "gni_per_capita": 65000}'
```

## Machine Learning Pipeline

1. **Data Collection** – UNDP / World Bank data (or synthetic generation)
2. **Data Cleaning** – Handle missing values, drop duplicates
3. **EDA** – Correlation heatmaps, distribution plots, pair plots
4. **Feature Scaling** – StandardScaler normalization
5. **Label Encoding** – Encode target categories
6. **Train-Test Split** – 80/20 stratified split
7. **Model Training** – Logistic Regression, Random Forest, Decision Tree
8. **Evaluation** – Accuracy, confusion matrix, classification report
9. **Model Selection** – Pick the best performing model
10. **Deployment** – Flask web application

## ER Diagram (Conceptual)

```
┌──────────────┐       ┌──────────────────┐       ┌───────────────────┐
│   Country    │ 1──N  │   Indicators     │ N──1  │ PredictionResult  │
├──────────────┤       ├──────────────────┤       ├───────────────────┤
│ country_id   │       │ indicator_id     │       │ prediction_id     │
│ country_name │       │ country_id (FK)  │       │ country_id (FK)   │
│ region       │       │ life_expectancy  │       │ life_expectancy   │
└──────────────┘       │ mean_schooling   │       │ mean_schooling    │
                       │ expected_school  │       │ expected_school   │
                       │ gni_per_capita   │       │ gni_per_capita    │
                       │ year             │       │ predicted_hdi     │
                       └──────────────────┘       │ confidence_score  │
                                                  │ timestamp         │
                                                  └───────────────────┘
```

- **Country**: Stores country reference information
- **Indicators**: Stores annual development indicator values per country
- **PredictionResult**: Stores model predictions with input values and timestamps

## How This Project Helps

- **Policy Making**: Governments can simulate "what-if" scenarios (e.g., raising life expectancy by 2 years) to see its impact on HDI category, helping prioritize budget allocation.
- **Development Analysis**: Track how specific indicators drive HDI changes over time and identify which dimension (health, education, income) most needs improvement.
- **Country Comparison**: Compare development profiles across countries and identify peer nations with similar indicator profiles for benchmarking.
