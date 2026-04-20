# AI-Assisted Real Estate Price Prediction and Dashboard

## Project Goal

The goal of this project is to build a simple AI-assisted system that helps users evaluate real estate listings by comparing a property's listed price with a machine learning–predicted fair price. By highlighting pricing gaps, the system allows users to better understand whether a property is overpriced, fairly priced, or undervalued. In addition to this comparison, the project provides interactive visualizations and property-specific insights that help users explore market patterns and understand which features influence pricing. Overall, the objective is to improve transparency in real estate decisions and give users a clearer, data-driven perspective when evaluating properties.

---

## Table of Contents
1. Introduction  
2. Project Structure  
3. House Price Prediction  
4. Time on Market Prediction  
5. Real Estate Dashboard  
6. AI-Powered Pricing Assistant (Chatbot)  
7. Data Description  
8. How to Run  
9. How to Use  

---

## Introduction

Real estate pricing depends on many factors such as location, size, quality, and market conditions. This project builds a simple AI-assisted system that:

- Predicts house prices using machine learning models  
- Estimates how long a house may stay on the market  
- Provides an interactive dashboard for exploring listings and insights  

The main goal is to help users make better pricing decisions by comparing:

Listed Price vs Predicted Price

and labeling properties as:
- Overpriced  
- Fairly priced  
- Undervalued  

---

## Project Structure

- app.py → Interactive dashboard  
- house_price_prediction.ipynb → Price prediction model  
- Time on market prediction.ipynb → Time on market model  
- train.csv, test.csv → Dataset files  
- submission.csv → Predicted prices  
- time_on_market_submission.csv → Time predictions  
- data_description.txt → Feature descriptions  

---

## House Price Prediction

Overview: Builds regression models to estimate house prices.

Steps:
- Data preprocessing (missing values, encoding, scaling)  
- Modeling (XGBoost, LightGBM, CatBoost)  
- Evaluation (RMSE, R²)  

Output:
submission.csv

---

## Time on Market Prediction

Overview: Estimates how long a property stays on the market.

Note:
TimeOnMarket = (YrSold - YearBuilt) * 12 + MoSold

Steps:
- Feature selection  
- Data preprocessing  
- Random Forest model  

Output:
time_on_market_submission.csv

---

## Real Estate Dashboard

Overview: app.py builds an interactive dashboard using Dash.

Features:
- Interactive map  
- Listing cards  
- Filters (neighborhood, bedrooms, type)  
- Property insight panel  
- Pricing labels (Overpriced / Fair / Undervalued)  
- Charts and comparisons  

---

## AI-Powered Pricing Assistant (Chatbot)

Overview: Simple rule-based assistant.

Capabilities:
- Price analysis  
- Market comparison  
- Deal detection  
- Feature explanation  

---

## Data Description

Key features:
- GrLivArea → Living area  
- OverallQual → Quality  
- YearBuilt → Build year  
- Neighborhood → Location  
- SalePrice → Target  

See:
data_description.txt

---

## How to Run

Follow these steps in order:

### 1. Open project in VS Code
- Download all the code files, make a folder and then open in vscode
---

### 2. Create virtual environment

```powershell
python -m venv .venv
```
### 3. Activate environment
```powershell
.\.venv\Scripts\Activate.ps1
```
### 4. Install dependencies
```powershell
pip install pandas numpy scikit-learn dash plotly notebook xgboost lightgbm catboost
```
### 5. Change the datasets file path with your local path where you have them saved, and then 
Run:
house_price_prediction.ipynb &
Time on market prediction.ipynb

### 6. Run Dashboard
```powershell
python app.py
```
### 7. Open browser
```powershell
http://127.0.0.1:8050/
```
