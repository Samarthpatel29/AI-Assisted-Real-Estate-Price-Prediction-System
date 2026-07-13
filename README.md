# Seattle Housing Intelligence

**Live demo:** https://ai-real-estate-dashboard-puce.vercel.app

An AI-powered housing analytics platform built on **21,000+ real home sales from King County / Seattle** (May 2014 – May 2015). It goes beyond simple price prediction: every listing gets a **Deal Score**, investor economics, an estimated time on market, comparable sales, and a live **renovation what-if simulator** — all served by a real regression model that trains at startup.

---

## What makes this different from a typical price-prediction project

| Feature | What it does |
|---|---|
| **Deal Score (0–100)** | Blends the price gap vs. the AI estimate, construction grade & condition, and view/waterfront appeal into a single ranked score. The dashboard surfaces the top deals for any filter. |
| **Live hedonic model** | A ridge regression on log(price) with ZIP-code fixed effects, fit **in production at startup** with pure NumPy (R² ≈ 0.88). No pickled artifacts, no fallback demo data. |
| **Renovation simulator** | Select a home, move sliders ("upgrade grade +2", "add 500 sq ft"), and the model re-prices the home instantly using its learned coefficients. |
| **Investor analytics** | Estimated market rent, gross rental yield, and a 30-year mortgage payment (20% down @ 6.5%) for every property. |
| **Estimated days on market** | Overpriced homes sit longer — a transparent model turns the price gap and quality into an expected DOM figure. |
| **Comparable sales** | Nearest-neighbor comps in the same ZIP by location, size, grade, and bedrooms. |
| **Real map & real trend** | Every point sits at its true latitude/longitude on a Seattle map, and the trend chart shows actual median sale prices by month, city vs. county. |
| **Pricing assistant** | A chat assistant that answers questions about deal scores, rent & yield, mortgage payments, days on market, comps, and recommendations for the selected home. |

---

## Dataset

`kc_house_data.csv` — 21,597 house sales in King County, WA (includes Seattle, Bellevue, Redmond, Kirkland, Medina, …), May 2014 – May 2015.

Key fields: `price`, `sqft_living`, `sqft_lot`, `bedrooms`, `bathrooms`, `floors`, `waterfront`, `view` (0–4), `condition` (1–5), `grade` (1–13 construction quality), `yr_built`, `yr_renovated`, `zipcode`, `lat`, `long`.

Data cleaning handled in `app.py`: NaN `waterfront`/`view`/`yr_renovated` filled with 0, `?` strings in `sqft_basement` coerced, resales deduplicated to the most recent sale.

> Street addresses shown in the app are synthetic (the public dataset does not include addresses); coordinates, prices, and features are real.

## The model

- **Target:** log(price)
- **Features:** log living/lot area, beds, baths, floors, waterfront, view, condition, grade, age, renovated flag, lat/long, and one-hot ZIP-code fixed effects (~70 ZIPs)
- **Estimator:** ridge regression solved in closed form with NumPy (`(XᵀX + λI)β = Xᵀy`), Duan smearing for unbiased back-transform to dollars
- **Fit:** ~0.5 s at startup, R² ≈ 0.88 on log price

Because the coefficient vector lives in memory, the renovation simulator can re-price any modified home instantly.

## Project structure

- `app.py` — the whole application: data prep, model, analytics, and the Dash dashboard
- `index.py` — Vercel serverless entrypoint (exposes the Flask/WSGI server)
- `vercel.json`, `requirements.txt`, `.vercelignore` — deployment config
- `kc_house_data.csv` — dataset
- `house_price_prediction.ipynb`, `Time on market prediction.ipynb` — original Ames-dataset notebooks (kept for reference; `train.csv`/`test.csv` belong to them)

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:8050.

## Deploy

The app runs on Vercel's free tier as a Python serverless function:

```bash
vercel deploy --prod
```

## How to use

1. **Filter** by city, bedrooms, or home tier (Starter → Luxury, or Waterfront) in the sidebar.
2. **Select a home** by clicking a map point, a scatter-plot point, or a listing card (cards are ranked by Deal Score).
3. **Read the Deal Analysis panel**: Deal Score, price gap, days on market, rent & yield, mortgage, why the model priced it that way, and comparable sales.
4. **Play with the Renovation Simulator** to see how upgrades change the AI estimate.
5. **Ask the Pricing Assistant** things like:
   - "Is this a good rental investment?"
   - "What's the deal score?"
   - "How long will it take to sell?"
   - "Show me comps"
   - "What are the best deals right now?"
