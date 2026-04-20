# AI-Assisted Real Estate Price Prediction and Dashboard

## Project Goal

This project builds a simple AI-assisted system for evaluating real estate listings. It compares a home's listed price with a machine learning predicted fair price, then highlights whether the listing appears overpriced, fairly priced, or undervalued.

The project also includes interactive visualizations, property-level insights, and a pricing assistant that helps users understand market patterns and the factors influencing property value.

---

## Table of Contents
1. Introduction
2. Project Structure
3. House Price Prediction
4. Time on Market Prediction
5. Real Estate Dashboard
6. AI-Powered Pricing Assistant
7. Data Description
8. How to Run
9. How to Use

---

## Introduction

Real estate pricing depends on many factors such as location, size, quality, and overall market conditions. This project combines machine learning and interactive analytics to:

- Predict house prices using multiple regression models
- Estimate how long a house may stay on the market
- Provide an interactive dashboard for exploring listings and pricing insights

The main comparison is:

```text
Listed Price vs Predicted Price
```

Properties are labeled as:

- Overpriced
- Fairly priced
- Undervalued

---

## Project Structure

- `app.py`
  Main Dash dashboard application

- `setup_environment.py`
  Creates the environment and installs project dependencies

- `house_price_prediction.ipynb`
  Notebook for house price prediction using XGBoost, LightGBM, and CatBoost

- `Time on market prediction.ipynb`
  Notebook for time on market prediction

- `train.csv`, `test.csv`
  Dataset files used by the project

- `submission.csv`
  Predicted house prices used by the dashboard

- `time_on_market_submission.csv`
  Output file for time on market predictions

- `data_description.txt`
  Description of dataset columns and meanings

---

## House Price Prediction

### Overview

This notebook builds regression models to estimate house prices from property features.

### Models Used

- XGBoost
- LightGBM
- CatBoost

### Typical Workflow

- Handle missing values
- Encode categorical variables
- Train models
- Evaluate model performance
- Save final predictions

### Output

```text
submission.csv
```

This file is the main prediction input used by `app.py`.

---

## Time on Market Prediction

### Overview

This notebook estimates how long a property may remain on the market.

### Note

The original dataset does not include a direct `TimeOnMarket` column, so the notebook creates a proxy target:

```text
TimeOnMarket = (YrSold - YearBuilt) * 12 + MoSold
```

### Output

```text
time_on_market_submission.csv
```

At the moment, the dashboard primarily uses `submission.csv`. The time on market output is part of the project workflow, but it is not the main file currently consumed by `app.py`.

---

## Real Estate Dashboard

### Overview

`app.py` builds an interactive dashboard using Dash and Plotly.

### Features

- Interactive property map
- Listing cards with pricing details
- Filters for neighborhood, bedrooms, and property type
- Property insight panel
- Price comparison charts
- Feature comparison charts
- Real-time pricing labels

### Dashboard Behavior

- If `submission.csv` exists, the app uses your generated house price predictions
- If `submission.csv` is missing, the app falls back to demo estimates so the dashboard can still load

This means the dashboard can still run without crashing, but you should generate `submission.csv` first if you want to see your real model output.

---

## AI-Powered Pricing Assistant

### Overview

The dashboard includes a simple rule-based pricing assistant.

### Capabilities

- Explain listed price vs predicted price
- Compare a property to neighborhood averages
- Highlight deal or overpriced signals
- Explain why a property received its pricing label
- Suggest value opportunities from the filtered results

---

## Data Description

Important features include:

- `GrLivArea` -> Living area
- `OverallQual` -> Overall quality
- `YearBuilt` -> Construction year
- `Neighborhood` -> Property location
- `SalePrice` -> Actual sale price target

For the full data dictionary, see:

```text
data_description.txt
```

---

## How to Run

Run the project in this order.

### 1. Open the project folder

Open the full project folder in VS Code or another editor.

### 2. Run setup

This script creates the virtual environment, installs dependencies, registers the Jupyter kernel, and writes setup details to `SETUP_COMPLETE.txt`.

```powershell
python setup_environment.py
```

### 3. Run the house price prediction notebook

Run:

- `house_price_prediction.ipynb`

This step generates:

```text
submission.csv
```

### 4. Run the time on market notebook

Run:

- `Time on market prediction.ipynb`

This step generates:

```text
time_on_market_submission.csv
```

### 5. Run the dashboard

```powershell
python app.py
```

Then open:

```text
http://127.0.0.1:8050/
```

### Notes

- `setup_environment.py` installs the project dependencies, including `dash`, `plotly`, `xgboost`, `lightgbm`, and `catboost`
- If Python is not installed on a machine, setup will not work until Python is installed first
- If `submission.csv` has not been generated yet, the dashboard will still open, but it will use fallback demo estimates instead of your trained model predictions

---

## How to Use

### Filters

Use the left sidebar to filter by:

- Neighborhood
- Number of bedrooms
- Property type

### Selecting a Property

You can select a property by clicking:

- A point on the map
- A point on the scatter plot
- A listing card

### Dashboard Views

The dashboard includes:

- A map of listings
- Listing cards with pricing labels
- A scatter plot comparing living area and listed price
- Feature comparison charts
- Price context charts
- A property insight panel

### Pricing Assistant

Ask the assistant questions such as:

- "Is this a good deal?"
- "What is the price gap?"
- "How does this compare to the neighborhood average?"
- "Why is this property overpriced?"
- "What are the best deals right now?"

### Pricing Labels

- `Overpriced`: listed price is meaningfully above the predicted price
- `Fairly priced`: listed price is close to the predicted price
- `Undervalued`: listed price is below the predicted price
