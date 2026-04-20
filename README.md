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
7. Security & Testing  
8. Data Description  
9. How to Run  

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

- `app.py`  
  Interactive Dash dashboard with AI-powered pricing assistant  

- `house_price_prediction.ipynb`  
  Notebook for predicting house prices using XGBoost, LightGBM, and CatBoost  

- `Time on market prediction.ipynb`  
  Notebook for estimating time on market using Random Forest  

- `train.csv`, `test.csv`  
  Dataset files containing housing features and sale prices  

- `submission.csv`  
  ML model predictions for house prices  

- `time_on_market_submission.csv`  
  Predictions for time on market  

- `data_description.txt`  
  Dataset feature definitions and descriptions  
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
- Interactive property map with zoom controls  
- Clickable listing cards with pricing details  
- Dynamic filters (neighborhood, bedrooms, property type)  
- Property-specific insight panel with AI-generated explanations  
- Pricing classification (Overpriced / Fair / Undervalued)  
- Real-time statistics and metrics  
- Feature comparison charts  
- Price comparison visualizations  

### Visualizations
- **Map View** - Geographic distribution with pricing labels and hover details  
- **Listing Cards** - Sortable property cards showing address, price, and classification  
- **Scatter Plot** - Living Area vs Price with market context  
- **Feature Impact Chart** - Property features vs dataset averages  
- **Price Comparison Chart** - Listed price vs predicted price vs neighborhood/market averages  

### Interactive Features
- Click a property on the map, scatter plot, or listing card to see details  
- Apply filters to narrow down by neighborhood, bedrooms, or property type  
- Adjust map zoom level with the slider  
- Real-time updates to all charts based on selection  

---

## AI-Powered Pricing Assistant (Chatbot)

### Overview
The dashboard includes an intelligent pricing assistant that answers questions about the selected property.

### Capabilities
The chatbot can answer questions about:
- **Price Analysis** - Listed price, AI estimate, price gaps, and value assessment  
- **Market Comparisons** - Neighborhood averages, price positioning relative to market  
- **Deal Alerts** - Identifies great deals (undervalued) and overpriced warnings  
- **Property Features** - Spaciousness, quality ratings, age, and how they impact value  
- **Recommendations** - Suggests best deals based on current filters  
- **Pricing Labels** - Explains why a property is marked as Overpriced, Fair, or Undervalued  
- **Filter Status** - Reports current filters and available properties  

### Example Questions
- "Is this a good deal?"  
- "What's the price gap?"  
- "How does this compare to neighborhood average?"  
- "Why is this property labeled as overpriced?"  
- "What's the best deal in my filtered results?"  
- "Tell me about the features of this home."  

### Technical Details
- Uses keyword matching to identify question types  
- Generates context-aware responses based on selected property and filters  
- Safely handles malicious input and malformed user text  
- All responses are plain-text to prevent injection attacks  



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

### Filtering & Selection
1. **Apply Filters** on the left sidebar to refine your search:
   - Select a neighborhood from the dropdown
   - Choose number of bedrooms
   - Click a property type button to filter homes

2. **Select a Property** by clicking on:
   - A point on the interactive map
   - A dot on the scatter plot
   - A listing card in the right panel

### Exploring Data
1. **View Statistics** at the top showing:
   - Number of listings matching your filters
   - Average listed price
   - AI price estimate
   - Count of overpriced homes

2. **Examine Visualizations**:
   - **Map View** - Geographic distribution with pricing labels
   - **Listing Cards** - Properties sorted by price gap
   - **Scatter Plot** - Living area vs price correlation
   - **Feature Comparison** - Selected property vs dataset average
   - **Price Context** - Neighborhood and market price comparison

### Using the Pricing Assistant
1. **Select a Property** - Click any property to populate the assistant context
2. **Ask Questions** in the chat box on the left:
   - "Is this a good deal?"
   - "What's the neighborhood average?"
   - "Why is this overpriced?"
   - "What are the best deals?"
   - "How does this compare to average?"
3. **Read Insights** - The assistant provides context-aware answers about pricing, features, and market position

### Understanding Pricing Labels
- **Overpriced** (Red) - Listed price is >8% above AI estimate
- **Fairly Priced** (Green) - Listed price is within 8% of AI estimate
- **Undervalued** (Orange) - Listed price is >8% below AI estimate


---
