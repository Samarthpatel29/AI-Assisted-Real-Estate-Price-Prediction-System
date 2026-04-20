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

- `app.py`  
  Interactive Dash dashboard with AI-powered pricing assistant  

- `house_price_prediction.ipynb`  
  Notebook for predicting house prices  

- `Time on market prediction.ipynb`  
  Notebook for estimating time on market  

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
submission.csv

---

## Time on Market Prediction

### Overview
This notebook estimates how long a property will stay on the market.

### Important Note
The dataset does not include a real "time on market" column, so a proxy variable is created:

TimeOnMarket = (YrSold - YearBuilt) * 12 + MoSold

### Steps
- Feature selection  
- Data preprocessing  
- Random Forest model  

### Output
Saved as:
time_on_market_submission.csv

---

## Real Estate Dashboard

### Overview
The app.py file builds an interactive dashboard using Dash and Plotly.

### Features
- Interactive property map  
- Clickable listing cards  
- Dynamic filters (neighborhood, bedrooms, property type)  
- Property insight panel  
- Pricing classification (Overpriced / Fair / Undervalued)  
- Real-time statistics  
- Feature comparison charts  
- Price comparison visualizations  

### Visualizations
- Map View  
- Listing Cards  
- Scatter Plot (Living Area vs Price)  
- Feature Comparison Chart  
- Price Comparison Chart  

---

## AI-Powered Pricing Assistant (Chatbot)

### Overview
The dashboard includes a simple rule-based assistant for answering questions about properties.

### Capabilities
- Price analysis  
- Market comparison  
- Deal detection  
- Feature explanation  

---

## Data Description

The dataset contains housing features such as:

- GrLivArea → Living area  
- OverallQual → Overall quality  
- YearBuilt → Construction year  
- Neighborhood → Location  
- GarageCars → Garage size  
- SalePrice → Target variable  

For full details, see:
data_description.txt

---

## How to Run

Follow these steps in order to run the full project.

1. Open the project  
Open the project folder in VS Code  

2. Create virtual environment (first time only)  
python -m venv .venv  

3. Activate environment  
.\.venv\Scripts\Activate.ps1  

4. Install dependencies  
pip install pandas numpy scikit-learn dash plotly notebook xgboost lightgbm catboost  

5. Replace file path for datasets with your own local path for wherever its mentioned
   
7. Run Jupyter Notebook  
python -m notebook  

8. Run notebooks (IMPORTANT)  
- house_price_prediction.ipynb → Run All → creates submission.csv  
- Time on market prediction.ipynb → Run All → creates time_on_market_submission.csv  

These files are required for the dashboard.

8. Run dashboard  
python app.py  

9. Open in browser  
http://127.0.0.1:8050  

---

## How to Use

### Filtering & Selection
- Select a neighborhood  
- Choose number of bedrooms  
- Click a property type button  

### Selecting a Property
Click on:
- Map point  
- Scatter plot  
- Listing card  

### Understanding Pricing Labels
- Overpriced → more than 8% above predicted price  
- Fairly priced → within 8% of predicted price  
- Undervalued → more than 8% below predicted price  

---
