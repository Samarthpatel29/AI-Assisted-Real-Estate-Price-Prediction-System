# AI-Assisted Real Estate Price Prediction and Decision Support System

This project is a simple real estate decision-support tool built for a DS440 capstone project. The main idea is to take housing data, predict a fair home price using machine learning, and then show that prediction in an interactive dashboard next to the listed price.

Instead of only showing raw listing data, the project tries to answer a more useful question:

**Does this property look overpriced, fairly priced, or undervalued based on its features?**

The dashboard lets users explore homes, click on individual properties, and view property-specific insights such as:
- listed price
- predicted price
- price gap
- pricing label
- feature comparison
- neighborhood price context

A second notebook is also included for **time-on-market prediction**. In this version, time on market is treated as a proxy-based experimental feature using the available dataset.

---

## What the Project Does

This project combines:
- **machine learning notebooks** for prediction
- **data preprocessing** for housing features
- **an interactive Dash web app** for visualization and decision support

The main system predicts a property’s fair value using housing features like:
- living area
- overall quality
- age
- bathrooms
- garage capacity
- neighborhood-related differences

The app then compares:
- **Listed Price**
- **Predicted Price**

Using that comparison, the app labels properties as:
- **Overpriced**
- **Fairly priced**
- **Undervalued**

Users can then click a property in the dashboard and view more detailed insights about why it received that label.

---

## Main Features

- sale price prediction notebook
- time-on-market prediction notebook
- interactive dashboard built with Dash and Plotly
- map view of listings
- clickable listing cards
- clickable scatter plot
- selected-property insight panel
- selected-property charts for:
  - property vs market
  - property feature comparison
  - property price context

---

## Project Files

- `app.py`  
  Main Dash application for the dashboard

- `train.csv`  
  Training dataset

- `test.csv`  
  Test dataset

- `house_price_prediction.ipynb`  
  Notebook for home price prediction

- `Time on market prediction.ipynb`  
  Notebook for time-on-market prediction

- `submission.csv`  
  Predicted sale prices generated from the house price notebook

- `time_on_market_submission.csv`  
  Predicted time-on-market values generated from the second notebook

---

## Important Before Running

If you copied this project from another machine or repo, make sure you **replace any old hard-coded file paths**.

For example, if you see paths like these in your notebooks or `app.py`:

```python
train_path = "C:/Users/yourname/Downloads/.../train.csv"
test_path = "C:/Users/yourname/Downloads/.../test.csv"
