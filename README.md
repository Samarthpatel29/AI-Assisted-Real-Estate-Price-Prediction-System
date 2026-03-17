# AI-Assisted Real Estate Price Prediction and Dashboard

## Project Goal

The goal of this project is to build a simple AI-assisted system that helps users evaluate real estate listings by comparing a property’s listed price with a machine learning–predicted fair price. By highlighting pricing gaps, the system allows users to better understand whether a property is overpriced, fairly priced, or undervalued. In addition to this comparison, the project provides interactive visualizations and property-specific insights that help users explore market patterns and understand which features influence pricing. Overall, the objective is to improve transparency in real estate decisions and give users a clearer, data-driven perspective when evaluating properties.



---

## Table of Contents
1. Introduction  
2. Project Structure  
3. House Price Prediction  
4. Time on Market Prediction  
5. Real Estate Dashboard  
6. Data Description  
7. How to Run  

---

## Introduction

Real estate pricing depends on many factors such as location, size, quality, and market conditions. This project builds a simple AI-assisted system that:

- Predicts house prices using machine learning models  
- Estimates how long a house may stay on the market  
- Provides an interactive dashboard for exploring listings and insights  

The main goal is to help users make better pricing decisions by comparing:

```
Listed Price vs Predicted Price
```

and labeling properties as:
- Overpriced  
- Fairly priced  
- Undervalued  

---

## Project Structure

- `house_price_prediction.ipynb`  
  Notebook for predicting house prices  

- `Time on market prediction.ipynb`  
  Notebook for estimating time on market  

- `app.py`  
  Interactive Dash dashboard  

- `train.csv`, `test.csv`  
  Dataset files  

- `submission.csv`  
  Predicted house prices  

- `time_on_market_submission.csv`  
  Predicted time on market  

- `data_description.txt`  
  Dataset feature descriptions  

---

## House Price Prediction

### Overview
This notebook builds regression models to estimate house prices based on property features.

### Steps
1. Data preprocessing  
   - Handle missing values  
   - Encode categorical variables  
   - Scale numerical features  

2. Modeling  
   - XGBoost  
   - LightGBM  
   - CatBoost  
   - Ensemble averaging  

3. Evaluation  
   - RMSE  
   - R² Score  

### Output
Predictions are saved to:

```
submission.csv
```

---

## Time on Market Prediction

### Overview
This notebook estimates how long a property will stay on the market.

### Important Note
The dataset does not include a real "time on market" column, so a proxy variable is created:

```
TimeOnMarket = (YrSold - YearBuilt) * 12 + MoSold
```

### Steps
- Feature selection  
- Data preprocessing  
- Random Forest model  

### Output
Saved as:

```
time_on_market_submission.csv
```

---

## Real Estate Dashboard

### Overview
The `app.py` file builds an interactive dashboard using Dash and Plotly.

### Features
- Interactive property map  
- Clickable listing cards  
- Filters (neighborhood, bedrooms, property type)  
- Property-specific insight panel  
- Pricing classification (Overpriced / Fair / Undervalued)  
- Dynamic charts based on selected property  

### Visualizations
- Map view with pricing labels  
- Scatter plot (Living Area vs Price)  
- Feature comparison chart  
- Price comparison (listed vs predicted vs averages)  

---

## Data Description

The dataset contains housing features such as:

- **GrLivArea** → Living area (sq ft)  
- **OverallQual** → Overall quality  
- **YearBuilt** → Construction year  
- **Neighborhood** → Location  
- **GarageCars** → Garage size  
- **SalePrice** → Target variable  

For full details, see:

```
data_description.txt
```

---

## How to Run

### 1. Clone the repository

```bash
git clone <your-repo-link>
cd <your-repo-folder>
```

---

### 2. Fix file paths (IMPORTANT)

If your code contains paths like:

```python
C:/Users/yourname/...
```

Replace them with:

```python
train.csv
test.csv
submission.csv
time_on_market_submission.csv
```

---

### 3. Create virtual environment (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

---

### 4. Install dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install pandas numpy scikit-learn dash plotly notebook xgboost lightgbm catboost
```

---

### 5. Run notebooks

```powershell
.\.venv\Scripts\python.exe -m notebook
```

Run:
- `house_price_prediction.ipynb`
- `Time on market prediction.ipynb`

---

### 6. Run dashboard

```powershell
.\.venv\Scripts\python.exe app.py
```

Open in browser:

```
http://127.0.0.1:8050
```

---

## How to Use

- Apply filters on the left panel  
- Adjust zoom using slider  
- Click a property from:
  - map  
  - scatter plot  
  - listing cards  
- View detailed insights and charts  

---


