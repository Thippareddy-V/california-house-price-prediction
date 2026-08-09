# 🏡 California House Price Prediction

[![CI](https://github.com/Thippareddy-V/california-house-price-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/Thippareddy-V/california-house-price-prediction/actions)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An end-to-end machine learning project that predicts median house prices for California census
block groups — from raw data to a trained, tuned, cross-validated model served through a REST
API and a live web interface.

**Live demo:** *(add your Streamlit Cloud link here once deployed)*

![App Screenshot](docs/screenshot.png)

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Live Demo](#live-demo)
- [Approach](#approach)
- [Results](#results)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [API Reference](#api-reference)
- [Getting Started](#getting-started)
- [Testing](#testing)
- [CI/CD](#cicd)
- [Tech Stack](#tech-stack)
- [What I'd Improve Next](#what-id-improve-next)
- [License](#license)

---

## Problem Statement

Given census-level features for a California neighborhood (income, house age, room counts,
population, location), predict the median house value for that area. This is a supervised
regression problem on the classic [California Housing dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset),
derived from the 1990 U.S. Census.

## Approach

1. **Baseline first** — Linear Regression, to establish a floor and understand the data's
   limitations before reaching for complexity.
2. **Diagnosed *why* the baseline underperforms** — `Latitude`/`Longitude` and price don't have
   a linear relationship (location effects are highly non-linear), which caps Linear Regression
   at R² ≈ 0.67 regardless of tuning.
3. **Compared model families** — Ridge, Decision Tree, Random Forest, XGBoost — scored with a
   single shared evaluation function so every result is directly comparable.
4. **Tuned the winner** — `RandomizedSearchCV` over XGBoost's hyperparameter space.
5. **Validated with 5-fold cross-validation** — confirmed the tuned R² is stable across data
   splits, not a lucky train/test partition (important lesson from this project: an unshuffled
   `KFold` on geographically-ordered data gave a misleading 0.67 — fixed by shuffling folds).
6. **Deployed** — trained model served via a Flask REST API, consumed by a Streamlit UI.

## Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 0.491 | 0.676 | 0.666 |
| Ridge Regression | 0.491 | 0.676 | 0.666 |
| Decision Tree | 0.466 | 0.678 | 0.664 |
| Random Forest | 0.336 | 0.518 | 0.804 |
| XGBoost (default params) | 0.300 | 0.459 | 0.846 |
| **XGBoost (tuned)** | **0.286** | **0.439** | **0.853** |

**5-fold cross-validation (shuffled):** R² = 0.854 ± 0.008 — tight spread confirms the result
generalizes, it isn't split-dependent.

MAE is in units of $100,000 → the deployed model's predictions are off by **~$28,600 on
average**.

## Dataset

- **20,640 rows**, one per California census block group (1990 census).
- **8 features:** `MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`,
  `Latitude`, `Longitude`.
- **Target (`Price`)** is median house value, in units of $100,000.
- **Known data quirk:** the target is **capped at $500,001** — any block group above that is
  clipped to this ceiling in the raw data. No model can predict these exactly, since it's an
  artificial cutoff rather than a real price signal. `MedInc` is similarly capped/floored.
- No missing values in the processed feature set.

## Architecture

┌─────────────────┐ HTTP POST /predict ┌──────────────────┐
│ Streamlit UI │ ───────────────────────────▶ │ Flask API │
│ (app/app.py) │ ◀─────────────────────────── │ (api/main.py) │
└─────────────────┘ JSON response └──────────────────┘
│
▼
┌──────────────────────┐
│ src/house_price/ │
│ predict.py │
│ → loads model.joblib │
└──────────────────────┘


The UI never touches the model directly — every prediction goes through the API, so the model
logic has one source of truth and could serve other clients (mobile app, another frontend)
without duplication.

## Project Structure

├── src/house_price/ # Core logic — data loading, training, prediction
│ ├── config.py # Paths, constants, feature ranges, tuned hyperparameters
│ ├── data.py # load_data(), get_X_y() — with a fallback data source
│ ├── train.py # Trains XGBoost, saves model + metrics
│ ├── predict.py # Loads model once (cached), exposes predict()
│ └── evaluate.py # Shared MAE/RMSE/R² scoring function
├── api/main.py # Flask REST API — /health, /predict, input validation
├── app/app.py # Streamlit web interface
├── models/v1/ # Trained model, feature list, metrics.json (versioned)
├── notebooks/ # Exploration notebook — EDA, model comparison, tuning
├── tests/ # pytest suite — data, prediction, and API tests
├── .github/workflows/ci.yml # Runs tests automatically on every push
├── Dockerfile # Containerizes the API
└── requirements.txt


## API Reference

### `GET /health`
Health check — confirms the server is running.

**Response**
```json
{ "status": "ok" }
```

### `POST /predict`
Predicts median house price for a given set of features.

**Request body**
```json
{
  "MedInc": 8.5,
  "HouseAge": 30,
  "AveRooms": 6.5,
  "AveBedrms": 1.0,
  "Population": 1000,
  "AveOccup": 3.0,
  "Latitude": 37.8,
  "Longitude": -122.2
}
```

**Response — 200 OK**
```json
{ "predicted_price_usd": 370678.11 }
```

**Response — 400 Bad Request** (missing/invalid field)
```json
{ "errors": ["missing field: Population", "Latitude must be between 32.0 and 42.0"] }
```

Input is validated against known feature ranges from the training data before it ever reaches
the model.

## Getting Started

### Prerequisites
- Python 3.11+
- Git

### Installation

```bash
git clone https://github.com/Thippareddy-V/california-house-price-prediction.git
cd california-house-price-prediction

python -m venv venv
venv\Scripts\Activate.ps1      # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

### Train the model
A trained model is already included in `models/v1/`, but to retrain:
```bash
python -m src.house_price.train
```

### Run the API
```bash
python -m api.main
```
Runs on `http://localhost:5000`.

### Run the web interface
In a second terminal (with the API still running):
```bash
streamlit run app/app.py
```

## Testing

```bash
pytest tests/ -v
```

Covers:
- **`test_data.py`** — dataset loads correctly, columns match expected feature set
- **`test_predict.py`** — prediction function returns a valid float
- **`test_api.py`** — `/health` responds, `/predict` returns 200 for valid input and 400 for
  invalid input

## CI/CD

Every push to `main` triggers `.github/workflows/ci.yml`, which spins up a clean environment,
installs dependencies from scratch, and runs the full test suite — catching regressions before
they reach `main`.

## Tech Stack

**ML:** scikit-learn · XGBoost · pandas · NumPy
**Backend:** Flask
**Frontend:** Streamlit
**Testing:** pytest
**CI:** GitHub Actions
**Containerization:** Docker *(Dockerfile included; container build pending — learning Docker)*

## What I'd Improve Next

- [ ] Build and test the Docker image; deploy the API as a container
- [ ] Deploy the Streamlit app publicly (Streamlit Cloud / Render) for a live demo link
- [ ] Model versioning beyond `v1` — track experiments as hyperparameters evolve
- [ ] Add a `/metrics` endpoint to the API for monitoring
- [ ] Expand feature engineering (the `RoomsPerHousehold` etc. features explored in the
      notebook aren't in the deployed model yet — would need retraining `train.py` to include
      them consistently across training and inference)

## License

MIT — see [LICENSE](LICENSE).

---

Built by [Thippareddy-V](https://github.com/Thippareddy-V) as part of ML/placement preparation.