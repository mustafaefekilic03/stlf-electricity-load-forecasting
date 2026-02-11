# Short-Term Electricity Load Forecasting (STLF)

This repository contains implementations of short-term hourly electricity load forecasting models using tree-based ensemble methods.

Two different modeling approaches are implemented and compared:

- `baslangic.py` → XGBoost-based forecasting model  
- `baslangicc.py` → Oblique Random Forest (ORF) model  

---

## Problem Description

Accurate short-term electricity load forecasting is critical for operational planning, energy balancing, maintenance scheduling, and demand-side management in power distribution systems.

This project focuses on building machine learning models to forecast hourly electricity load using advanced feature engineering and time-based validation.

---

## Data

The original dataset is **not included** in this repository due to confidentiality constraints.

The expected dataset structure includes:

- `timestamp` (datetime)
- `load` (target variable)
- Optional meteorological variables (e.g., temperature, humidity, wind speed)
- Calendar-based variables (hour, day of week, holiday indicators, etc.)

---

## Modeling Approaches

### 1️⃣ XGBoost Model (`baslangic.py`)

- Gradient boosting tree-based regression
- Time-based train/validation split
- Feature engineering with lag and rolling statistics
- Evaluation using standard regression metrics

### 2️⃣ Oblique Random Forest (ORF) (`baslangicc.py`)

- Tree ensemble model using oblique splits
- Designed to capture complex nonlinear relationships
- Compared against XGBoost for forecasting performance

---

## Feature Engineering (Examples)

The following feature groups are typically used:

- Calendar features (hour, day of week, month)
- Lag features (e.g., 1-hour, 24-hour, 168-hour lags)
- Rolling statistics (moving averages, rolling standard deviation)
- Seasonal encodings (sin/cos transformations)
- Meteorological features (if available)

---

## Evaluation Metrics

Model performance is evaluated using:

- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **MAPE** (Mean Absolute Percentage Error)

Validation is performed using time-based splits to prevent data leakage.

---

## How to Use

1. Prepare a dataset matching the expected column structure.
2. Update the dataset path inside the script if necessary.
3. Run:

Notes

This repository focuses on modeling structure and methodology.

The dataset is excluded due to confidentiality.

Results may vary depending on feature configuration and data quality.

Author

Mustafa Efe Kılıç
Industrial Engineering | Data Science | Forecasting & Optimization

```bash
python baslangic.py
python baslangicc.py


