import os
import ast
import numpy as np
import pandas as pd
import dash
from dash import dcc, html, Input, Output, State
from dash.dependencies import ALL
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------
# Load and prepare data (King County / Seattle home sales)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
raw = pd.read_csv(os.path.join(BASE_DIR, "kc_house_data.csv"))

# Clean known quirks of this dataset: NaNs in waterfront/view/yr_renovated,
# '?' strings in sqft_basement
raw["waterfront"] = raw["waterfront"].fillna(0).astype(int)
raw["view"] = raw["view"].fillna(0).astype(int)
raw["yr_renovated"] = raw["yr_renovated"].fillna(0).astype(int)
raw["sqft_basement"] = pd.to_numeric(raw["sqft_basement"], errors="coerce").fillna(0)

# Keep the most recent sale per home id
raw["date"] = pd.to_datetime(raw["date"])
raw = raw.sort_values("date").drop_duplicates("id", keep="last").reset_index(drop=True)
data = raw.copy()
data["Id"] = data.index.astype(int)

ZIP_CITY = {
    98001: "Auburn", 98002: "Auburn", 98003: "Federal Way", 98004: "Bellevue",
    98005: "Bellevue", 98006: "Bellevue", 98007: "Bellevue", 98008: "Bellevue",
    98010: "Black Diamond", 98011: "Bothell", 98014: "Carnation", 98019: "Duvall",
    98022: "Enumclaw", 98023: "Federal Way", 98024: "Fall City", 98027: "Issaquah",
    98028: "Kenmore", 98029: "Issaquah", 98030: "Kent", 98031: "Kent",
    98032: "Kent", 98033: "Kirkland", 98034: "Kirkland", 98038: "Maple Valley",
    98039: "Medina", 98040: "Mercer Island", 98042: "Kent", 98045: "North Bend",
    98052: "Redmond", 98053: "Redmond", 98055: "Renton", 98056: "Renton",
    98058: "Renton", 98059: "Renton", 98065: "Snoqualmie", 98070: "Vashon Island",
    98072: "Woodinville", 98074: "Sammamish", 98075: "Sammamish", 98077: "Woodinville",
    98092: "Auburn", 98146: "Burien", 98148: "SeaTac", 98155: "Shoreline",
    98166: "Burien", 98168: "Tukwila", 98177: "Shoreline", 98188: "SeaTac",
    98198: "Des Moines",
}


def zip_to_city(z):
    z = int(z)
    if z in ZIP_CITY:
        return ZIP_CITY[z]
    if 98100 <= z <= 98199:
        return "Seattle"
    return f"ZIP {z}"


data["City"] = data["zipcode"].apply(zip_to_city)
data["Age"] = 2015 - data["yr_built"]
data["Renovated"] = (data["yr_renovated"] > 0).astype(int)
data["Bedrooms"] = data["bedrooms"].clip(upper=8).astype(int)

# -----------------------------
# Hedonic pricing model: ridge regression on log(price)
# with zipcode fixed effects — fit at startup with pure numpy
# -----------------------------
def build_features(df, zip_categories):
    numeric = np.column_stack([
        np.log(df["sqft_living"].clip(lower=200)),
        np.log(df["sqft_lot"].clip(lower=400)),
        df["bedrooms"].clip(upper=8),
        df["bathrooms"],
        df["floors"],
        df["waterfront"],
        df["view"],
        df["condition"],
        df["grade"],
        2015 - df["yr_built"],
        (df["yr_renovated"] > 0).astype(int),
        df["lat"],
        df["long"],
    ])
    zips = pd.Categorical(df["zipcode"], categories=zip_categories)
    zip_dummies = pd.get_dummies(zips, drop_first=True).to_numpy(dtype=float)
    ones = np.ones((len(df), 1))
    return np.hstack([ones, numeric, zip_dummies])


ZIP_CATEGORIES = sorted(data["zipcode"].unique())
X = build_features(data, ZIP_CATEGORIES)
y = np.log(data["price"].to_numpy())

lam = 1.0
XtX = X.T @ X + lam * np.eye(X.shape[1])
beta = np.linalg.solve(XtX, X.T @ y)

log_pred = X @ beta
residuals = y - log_pred
smearing = float(np.mean(np.exp(residuals)))  # Duan smearing for unbiased back-transform
data["PredictedPrice"] = np.exp(log_pred) * smearing

r2 = 1 - np.sum(residuals ** 2) / np.sum((y - y.mean()) ** 2)
MODEL_NOTE = f"Ridge hedonic model • R² = {r2:.2f} (log price)"


def predict_price(row_like):
    """Re-predict price for a single (possibly modified) property dict/Series."""
    df = pd.DataFrame([row_like])
    xv = build_features(df, ZIP_CATEGORIES)
    return float(np.exp(xv @ beta)[0] * smearing)


# -----------------------------
# Derived analytics
# -----------------------------
data["ListedPrice"] = data["price"]
data["PriceGap"] = data["ListedPrice"] - data["PredictedPrice"]
data["GapPct"] = data["PriceGap"] / data["PredictedPrice"]


def pricing_label(x):
    if x > 0.08:
        return "Overpriced"
    elif x < -0.08:
        return "Undervalued"
    return "Fairly priced"


data["PricingLabel"] = data["GapPct"].apply(pricing_label)

# Deal Score (0-100): value gap + quality + appeal
value_rank = (-data["GapPct"]).rank(pct=True)
quality_rank = (data["grade"].rank(pct=True) + data["condition"].rank(pct=True)) / 2
appeal = (data["view"] / 4) * 0.6 + data["waterfront"] * 0.4
data["DealScore"] = (value_rank * 55 + quality_rank * 25 + appeal * 20).clip(1, 99).round(0).astype(int)

# Estimated days on market: overpricing slows sales, quality speeds them up
data["EstDOM"] = (
    28 * np.exp(2.2 * data["GapPct"]) - (data["grade"] - 7) * 2
).clip(5, 120).round(0).astype(int)

# Investor metrics (rough 2014-15 Seattle rent heuristic)
data["RentEst"] = (0.9 * data["sqft_living"] + 340 * data["bathrooms"] + 220).clip(900, 9000)
data["GrossYield"] = (data["RentEst"] * 12 / data["ListedPrice"] * 100).round(1)

RATE, YEARS, DOWN = 0.065, 30, 0.20
_r = RATE / 12
_n = YEARS * 12
data["Mortgage"] = (data["ListedPrice"] * (1 - DOWN)) * (_r * (1 + _r) ** _n) / ((1 + _r) ** _n - 1)

data["PricePerSqft"] = data["ListedPrice"] / data["sqft_living"]

street_names = [
    "Alder St", "Rainier Ave", "Cascade Dr", "Pine St", "Madrona Ln",
    "Elliott Ave", "Union St", "Magnolia Blvd", "Lakeview Rd", "Greenwood Ave"
]
data["Address"] = data.apply(
    lambda r: f"{100 + int(r['Id']) % 9800} {street_names[int(r['Id']) % len(street_names)]}",
    axis=1
)

GRADE_TIERS = {
    "ALL": "All Homes",
    "starter": "Starter",
    "standard": "Standard",
    "premium": "Premium",
    "luxury": "Luxury",
    "waterfront": "Waterfront",
}


def tier_mask(df, tier):
    if tier == "starter":
        return df["grade"] <= 6
    if tier == "standard":
        return df["grade"] == 7
    if tier == "premium":
        return df["grade"].between(8, 9)
    if tier == "luxury":
        return df["grade"] >= 10
    if tier == "waterfront":
        return df["waterfront"] == 1
    return pd.Series(True, index=df.index)


CITIES = ["ALL"] + sorted(data["City"].unique().tolist())

# Monthly market trend (real sale dates: May 2014 - May 2015)
data["SaleMonth"] = data["date"].dt.to_period("M").astype(str)
county_trend = data.groupby("SaleMonth")["ListedPrice"].median()


def label_color(label):
    return {
        "Overpriced": "#f97316",
        "Undervalued": "#f59e0b",
        "Fairly priced": "#22c55e"
    }.get(label, "#64748b")


def label_bg(label):
    return {
        "Overpriced": "rgba(249,115,22,0.12)",
        "Undervalued": "rgba(245,158,11,0.12)",
        "Fairly priced": "rgba(34,197,94,0.12)"
    }.get(label, "rgba(100,116,139,0.12)")


def score_color(score):
    if score >= 70:
        return "#16a34a"
    if score >= 45:
        return "#f59e0b"
    return "#ef4444"


# -----------------------------
# Helpers
# -----------------------------
def money(x):
    return f"${x:,.0f}"


def glass_card(radius="24px", padding="20px"):
    return {
        "background": "rgba(255,255,255,0.78)",
        "backdropFilter": "blur(14px)",
        "WebkitBackdropFilter": "blur(14px)",
        "padding": padding,
        "borderRadius": radius,
        "boxShadow": "0 16px 40px rgba(15, 23, 42, 0.08)",
        "border": "1px solid rgba(255,255,255,0.6)",
        "boxSizing": "border-box"
    }


def make_stat_card(title, value, subtitle="", accent="#3b82f6"):
    return html.Div(
        [
            html.Div(
                style={
                    "width": "46px",
                    "height": "46px",
                    "borderRadius": "16px",
                    "background": f"linear-gradient(135deg, {accent}, rgba(255,255,255,0.95))",
                    "marginBottom": "14px",
                    "boxShadow": f"0 10px 24px {accent}30"
                }
            ),
            html.Div(title, style={"fontSize": "13px", "color": "#64748b", "marginBottom": "6px", "fontWeight": "700"}),
            html.Div(value, style={"fontSize": "22px", "fontWeight": "900", "color": "#0f172a"}),
            html.Div(subtitle, style={"fontSize": "12px", "color": "#94a3b8", "marginTop": "4px"}),
        ],
        style={
            **glass_card(radius="22px", padding="20px"),
            "flex": "1",
            "minWidth": "200px",
            "transition": "transform 0.2s ease, box-shadow 0.2s ease"
        },
        className="hover-card"
    )


def make_mini_metric(title, value, subtitle=""):
    return html.Div(
        [
            html.Div(title, style={"fontSize": "12px", "color": "#94a3b8", "fontWeight": "700"}),
            html.Div(value, style={"fontSize": "20px", "fontWeight": "900", "color": "#0f172a", "marginTop": "4px"}),
            html.Div(subtitle, style={"fontSize": "12px", "color": "#64748b", "marginTop": "2px"}),
        ],
        style={
            "padding": "16px",
            "borderRadius": "18px",
            "background": "rgba(248,250,252,0.9)",
            "border": "1px solid rgba(226,232,240,0.9)"
        }
    )


def deal_badge(score, size="normal"):
    return html.Div(
        [
            html.Div(f"{score}", style={"fontSize": "20px" if size == "normal" else "30px", "fontWeight": "900", "lineHeight": "1"}),
            html.Div("DEAL SCORE", style={"fontSize": "8px" if size == "normal" else "10px", "fontWeight": "800", "letterSpacing": "0.06em", "opacity": "0.85"}),
        ],
        style={
            "background": f"linear-gradient(135deg, {score_color(score)}, {score_color(score)}cc)",
            "color": "white",
            "borderRadius": "16px",
            "padding": "8px 10px" if size == "normal" else "12px 16px",
            "textAlign": "center",
            "boxShadow": f"0 8px 20px {score_color(score)}40",
            "flexShrink": "0",
            "height": "fit-content"
        }
    )


def make_listing_card(row, selected_id=None):
    badge = row["PricingLabel"]
    badge_color = label_color(badge)
    is_selected = selected_id == row["Id"]

    return html.Button(
        [
            html.Div(
                [
                    html.Div("🏡" if row["waterfront"] == 0 else "🌊", style={"fontSize": "34px"}),
                    deal_badge(row["DealScore"]),
                ],
                style={
                    "width": "96px",
                    "borderRadius": "20px",
                    "background": "linear-gradient(135deg, #dbeafe 0%, #bfdbfe 50%, #ffffff 100%)",
                    "display": "flex",
                    "flexDirection": "column",
                    "alignItems": "center",
                    "justifyContent": "center",
                    "gap": "8px",
                    "padding": "10px 6px",
                    "flexShrink": "0",
                    "boxShadow": "inset 0 1px 0 rgba(255,255,255,0.75)"
                },
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        row["Address"],
                                        style={"fontSize": "20px", "fontWeight": "900", "color": "#0f172a", "lineHeight": "1.1"}
                                    ),
                                    html.Div(
                                        f"{row['City']} • {int(row['Bedrooms'])} bed • {row['bathrooms']} bath • {int(row['sqft_living']):,} sq ft",
                                        style={"fontSize": "13px", "color": "#64748b", "marginTop": "7px", "fontWeight": "600"}
                                    ),
                                ],
                                style={"flex": "1"}
                            ),
                            html.Span(
                                "Selected" if is_selected else badge,
                                style={
                                    "background": label_bg(badge) if not is_selected else "rgba(59,130,246,0.14)",
                                    "color": badge_color if not is_selected else "#1e40af",
                                    "padding": "7px 12px",
                                    "borderRadius": "999px",
                                    "fontSize": "12px",
                                    "fontWeight": "800",
                                    "border": f"1px solid {badge_color}22" if not is_selected else "1px solid rgba(59,130,246,0.18)",
                                    "height": "fit-content"
                                },
                            )
                        ],
                        style={"display": "flex", "justifyContent": "space-between", "gap": "12px", "alignItems": "flex-start"},
                    ),
                    html.Div(
                        [
                            make_mini_metric("Listed", money(row["ListedPrice"])),
                            make_mini_metric("AI Estimate", money(row["PredictedPrice"])),
                            make_mini_metric("Est. Days on Mkt", f"{int(row['EstDOM'])} days"),
                        ],
                        style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(110px, 1fr))", "gap": "10px", "marginTop": "14px"}
                    )
                ],
                style={"flex": "1", "textAlign": "left"},
            ),
        ],
        id={"type": "listing-card", "index": int(row["Id"])},
        n_clicks=0,
        className="listing-hover",
        style={
            "display": "flex",
            "gap": "16px",
            "padding": "18px",
            "width": "100%",
            "background": "linear-gradient(180deg, rgba(255,255,255,0.95) 0%, rgba(248,250,252,0.9) 100%)",
            "borderRadius": "24px",
            "border": "2px solid #3b82f6" if is_selected else "1px solid rgba(226,232,240,0.85)",
            "marginBottom": "16px",
            "cursor": "pointer",
            "textAlign": "left",
            "appearance": "none",
            "WebkitAppearance": "none",
            "outline": "none",
            "transform": "translateY(-2px)" if is_selected else "none",
            "boxShadow": "0 18px 38px rgba(59,130,246,0.16)" if is_selected else "0 12px 30px rgba(15, 23, 42, 0.08)",
            "transition": "all 0.2s ease"
        }
    )


def tier_button(value, selected_value):
    selected = value == selected_value
    return html.Button(
        GRADE_TIERS[value],
        id={"type": "tier-chip", "index": value},
        n_clicks=0,
        style={
            "padding": "10px 12px",
            "borderRadius": "14px",
            "border": "1px solid rgba(59,130,246,0.22)" if selected else "1px solid #dbeafe",
            "background": "linear-gradient(135deg, #1e40af, #60a5fa)" if selected else "rgba(255,255,255,0.9)",
            "color": "white" if selected else "#1e293b",
            "fontWeight": "800",
            "fontSize": "13px",
            "cursor": "pointer",
            "boxShadow": "0 10px 20px rgba(59,130,246,0.22)" if selected else "0 4px 12px rgba(15,23,42,0.04)",
            "transition": "all 0.2s ease",
            "textAlign": "center",
            "width": "100%"
        },
        className="chip-hover"
    )


def build_explanation(row):
    reasons = []
    zc = data[data["zipcode"] == row["zipcode"]]
    ppsf_pct = (zc["PricePerSqft"] < row["PricePerSqft"]).mean() * 100
    reasons.append(
        f"At {money(row['PricePerSqft'])}/sq ft, this home is pricier than {ppsf_pct:.0f}% of homes in its ZIP code."
    )
    if row["grade"] >= 9:
        reasons.append(f"Construction grade {int(row['grade'])}/13 is high-end, which lifts the model's estimate.")
    elif row["grade"] <= 6:
        reasons.append(f"Construction grade {int(row['grade'])}/13 is below average, which lowers the estimate.")
    else:
        reasons.append(f"Construction grade {int(row['grade'])}/13 is about average for the county.")
    if row["waterfront"] == 1:
        reasons.append("Waterfront location adds a large premium in the model.")
    elif row["view"] >= 3:
        reasons.append("A strong view rating adds a meaningful premium.")
    elif row["Age"] <= 10:
        reasons.append("Newer construction supports the estimated value.")
    elif row["Age"] >= 70:
        reasons.append("The home's age pulls the estimate down relative to newer stock.")
    else:
        reasons.append(f"Built in {int(row['yr_built'])}, the home is typical for its neighborhood.")
    if row["GapPct"] > 0.08:
        reasons.append(f"Listed {row['GapPct']*100:.0f}% above the AI estimate — the market may push back.")
    elif row["GapPct"] < -0.08:
        reasons.append(f"Listed {abs(row['GapPct'])*100:.0f}% below the AI estimate — a potential value opportunity.")
    else:
        reasons.append("The listed price sits close to the AI estimate.")
    return reasons[:4]


def get_filtered_data(city, bedrooms, tier):
    filtered = data
    if city != "ALL":
        filtered = filtered[filtered["City"] == city]
    if bedrooms != -1:
        filtered = filtered[filtered["Bedrooms"] == bedrooms]
    if tier != "ALL":
        filtered = filtered[tier_mask(filtered, tier)]
    if filtered.empty:
        filtered = data.head(10)
    return filtered


def find_comps(row, k=4):
    """Nearest comparable sales: same ZIP, similar size/grade/location."""
    pool = data[(data["zipcode"] == row["zipcode"]) & (data["Id"] != row["Id"])]
    if len(pool) < k:
        pool = data[(data["City"] == row["City"]) & (data["Id"] != row["Id"])]
    if pool.empty:
        return pool
    d = (
        ((pool["lat"] - row["lat"]) * 69) ** 2
        + ((pool["long"] - row["long"]) * 51) ** 2
        + ((pool["sqft_living"] - row["sqft_living"]) / 800) ** 2
        + ((pool["grade"] - row["grade"]) / 1.5) ** 2
        + ((pool["Bedrooms"] - row["Bedrooms"]) / 2) ** 2
    )
    return pool.loc[d.nsmallest(k).index]


def make_chat_message(role, text):
    is_user = role == "user"
    return html.Div(
        text,
        style={
            "maxWidth": "88%",
            "width": "fit-content",
            "alignSelf": "flex-end" if is_user else "flex-start",
            "background": "linear-gradient(135deg, #1e40af, #60a5fa)" if is_user else "rgba(248,250,252,0.95)",
            "color": "white" if is_user else "#334155",
            "padding": "10px 12px",
            "borderRadius": "14px",
            "fontSize": "13px",
            "lineHeight": "1.5",
            "marginBottom": "10px",
            "border": "none" if is_user else "1px solid rgba(226,232,240,0.9)",
            "wordBreak": "break-word",
            "overflowWrap": "anywhere"
        }
    )


def generate_ai_insights(selected_id):
    if selected_id is None or selected_id not in data["Id"].values:
        return None
    row = data[data["Id"] == selected_id].iloc[0]
    insights = []
    gap_pct = row["GapPct"] * 100
    if row["DealScore"] >= 75:
        insights.append(f"🚀 Deal Score {row['DealScore']}/100 — one of the stronger opportunities in {row['City']}.")
    elif row["DealScore"] <= 30:
        insights.append(f"⚠️ Deal Score {row['DealScore']}/100 — weak value at the current list price.")
    if gap_pct < -12:
        insights.append(f"💰 Listed {abs(gap_pct):.0f}% below the AI estimate.")
    elif gap_pct > 12:
        insights.append(f"📈 Listed {gap_pct:.0f}% above the AI estimate — expect ~{int(row['EstDOM'])} days on market.")
    if row["GrossYield"] >= 6:
        insights.append(f"🏦 Est. gross rental yield {row['GrossYield']}% — strong for the area.")
    if row["waterfront"] == 1:
        insights.append("🌊 Waterfront property — scarce inventory, large model premium.")
    return insights[:3] if insights else None


def build_chat_reply(user_text, selected_id, city, bedrooms, tier):
    question = (user_text or "").strip().lower()
    filtered = get_filtered_data(city, bedrooms, tier)

    if selected_id is not None and selected_id in data["Id"].values:
        row = data[data["Id"] == selected_id].iloc[0]
    else:
        row = filtered.iloc[0]

    city_avg = data[data["City"] == row["City"]]["ListedPrice"].median()
    gap_pct = row["GapPct"] * 100

    if any(w in question for w in ["score", "deal score"]):
        return (
            f"{row['Address']} scores {row['DealScore']}/100. The score blends the price gap vs the AI estimate, "
            f"construction grade and condition, and view/waterfront appeal."
        )
    if any(w in question for w in ["rent", "yield", "roi", "invest", "cash flow"]):
        return (
            f"Estimated rent for {row['Address']} is about {money(row['RentEst'])}/month, a gross yield of "
            f"{row['GrossYield']}% on the {money(row['ListedPrice'])} list price. "
            f"With 20% down at 6.5%, the mortgage runs about {money(row['Mortgage'])}/month."
        )
    if any(w in question for w in ["mortgage", "payment", "monthly", "afford"]):
        return (
            f"With 20% down and a 30-year loan at 6.5%, {row['Address']} costs about {money(row['Mortgage'])}/month "
            f"before taxes and insurance."
        )
    if any(w in question for w in ["days", "how long", "market time", "sell"]):
        return (
            f"The model estimates ~{int(row['EstDOM'])} days on market for {row['Address']}. "
            f"Homes priced above the AI estimate tend to sit longer."
        )
    if any(w in question for w in ["comp", "similar", "nearby"]):
        comps = find_comps(row, 3)
        parts = [f"{c['Address']} ({money(c['ListedPrice'])}, {int(c['sqft_living']):,} sq ft)" for _, c in comps.iterrows()]
        return "Closest comparable sales: " + "; ".join(parts) + "."
    if any(w in question for w in ["why", "reason", "label", "labeled"]):
        return " ".join(build_explanation(row)[:2])
    if any(w in question for w in ["price", "estimate", "listed", "gap", "deal", "bargain", "value"]):
        response = (
            f"{row['Address']} is listed at {money(row['ListedPrice'])}. "
            f"The AI estimate is {money(row['PredictedPrice'])}, a gap of {gap_pct:.1f}%. "
            f"It is labeled {row['PricingLabel'].lower()} with a Deal Score of {row['DealScore']}/100."
        )
        return response
    if any(w in question for w in ["average", "market", "neighborhood", "city", "compare"]):
        return (
            f"The median list price in {row['City']} is {money(city_avg)}. This home offers "
            f"{int(row['sqft_living']):,} sq ft at {money(row['PricePerSqft'])}/sq ft, grade {int(row['grade'])}/13."
        )
    if any(w in question for w in ["filter", "filtered", "results", "homes"]):
        return (
            f"Your current filters show {len(filtered):,} homes. City: {city if city != 'ALL' else 'All'}, "
            f"Bedrooms: {bedrooms if bedrooms != -1 else 'Any'}, Tier: {GRADE_TIERS.get(tier, tier)}."
        )
    if any(w in question for w in ["recommend", "suggest", "best", "advice"]):
        best = filtered.nlargest(3, "DealScore")
        if len(best) > 0:
            top = best.iloc[0]
            return (
                f"💡 Top pick: {top['Address']} in {top['City']} at {money(top['ListedPrice'])} — "
                f"Deal Score {top['DealScore']}/100, listed {top['GapPct']*100:.0f}% vs the AI estimate."
            )
        return "Not enough data for recommendations yet."
    return (
        f"Ask me about price, deal score, rent & yield, mortgage payments, days on market, comps, or recommendations. "
        f"Right now I'm focused on {row['Address']} in {row['City']}."
    )


# -----------------------------
# App setup
# -----------------------------
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "AI Assisted Real Estate"
server = app.server  # exposed for deployment (Vercel/gunicorn)
app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            html, body {
                overflow-x: hidden;
            }
            .dash-graph svg, .js-plotly-plot svg {
                overflow: hidden !important;
            }
            body {
                margin: 0;
                background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 45%, #f0f9ff 100%);
                background-size: 180% 180%;
                animation: gradientShift 18s ease infinite;
            }
            @keyframes gradientShift {
                0% { background-position: 0% 50%; }
                50% { background-position: 100% 50%; }
                100% { background-position: 0% 50%; }
            }
            @keyframes fadeSlideUp {
                from { opacity: 0; transform: translateY(18px); }
                to { opacity: 1; transform: translateY(0); }
            }
            @keyframes floatPulse {
                0% { transform: translateY(0px); }
                50% { transform: translateY(-6px); }
                100% { transform: translateY(0px); }
            }
            @keyframes shimmer {
                0% { background-position: -200% 0; }
                100% { background-position: 200% 0; }
            }
            .page-shell { animation: fadeSlideUp 0.8s ease; }
            .hero-title { animation: fadeSlideUp 0.9s ease; }
            .floating-badge { animation: floatPulse 3.8s ease-in-out infinite; }
            .hover-card, .listing-hover, .chip-hover, .chat-card, .glass-panel { animation: fadeSlideUp 0.7s ease; }
            .hover-card:hover {
                transform: translateY(-8px) scale(1.01);
                box-shadow: 0 22px 48px rgba(59,130,246,0.16) !important;
            }
            .listing-hover:hover {
                transform: translateY(-6px) scale(1.01) !important;
                box-shadow: 0 24px 46px rgba(15, 23, 42, 0.13) !important;
            }
            .chip-hover:hover {
                transform: translateY(-3px);
                box-shadow: 0 16px 32px rgba(59,130,246,0.18) !important;
            }
            .glass-panel:hover, .chat-card:hover { box-shadow: 0 22px 52px rgba(15, 23, 42, 0.10) !important; }
            .sparkle-dot {
                background: linear-gradient(90deg, rgba(255,255,255,0.4), rgba(255,255,255,0.95), rgba(255,255,255,0.4));
                background-size: 200% 100%;
                animation: shimmer 2.8s linear infinite;
            }
            .dash-graph { transition: transform 0.25s ease, box-shadow 0.25s ease; }
            .dash-graph:hover { transform: translateY(-4px); }
            * { scroll-behavior: smooth; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""

app.layout = html.Div(
    [
        dcc.Store(id="tier-store", data="ALL"),
        dcc.Store(id="selected-property-store"),
        dcc.Store(
            id="chat-history-store",
            data=[
                {
                    "role": "assistant",
                    "text": "Hi, I'm the housing intelligence assistant. Ask about deal scores, rent & yield, mortgage payments, days on market, or comps."
                }
            ]
        ),

        html.Div(
            [
                html.Div(
                    [
                        html.Div("Filters", style={"fontSize": "22px", "fontWeight": "900", "color": "#0f172a", "marginBottom": "16px"}),

                        html.Div("City / Area", style={"fontSize": "13px", "fontWeight": "800", "marginBottom": "6px", "color": "#475569"}),
                        dcc.Dropdown(
                            id="city-filter",
                            options=[{"label": "All of King County", "value": "ALL"}] +
                                    [{"label": c, "value": c} for c in CITIES if c != "ALL"],
                            value="ALL",
                            clearable=False,
                            style={"marginBottom": "16px"}
                        ),

                        html.Div("Bedrooms", style={"fontSize": "13px", "fontWeight": "800", "marginBottom": "6px", "color": "#475569"}),
                        dcc.Dropdown(
                            id="bedroom-filter",
                            options=[{"label": "Any", "value": -1}] +
                                    [{"label": f"{b} Bedrooms", "value": int(b)} for b in sorted(data["Bedrooms"].unique()) if b > 0],
                            value=-1,
                            clearable=False,
                            style={"marginBottom": "16px"}
                        ),

                        html.Div("Home Tier", style={"fontSize": "13px", "fontWeight": "800", "marginBottom": "4px", "color": "#475569"}),
                        html.Div(
                            "Based on King County construction grade.",
                            style={"fontSize": "11px", "color": "#64748b", "marginBottom": "8px"}
                        ),
                        html.Div(
                            id="tier-buttons",
                            style={"display": "grid", "gridTemplateColumns": "repeat(3, minmax(80px, 1fr))", "gap": "6px", "marginBottom": "8px"}
                        ),

                        html.Div(
                            [
                                html.Div("Live model", style={"fontSize": "12px", "color": "#94a3b8", "fontWeight": "800", "textTransform": "uppercase", "letterSpacing": "0.08em"}),
                                html.Div("AI Assisted Real Estate", style={"fontSize": "18px", "fontWeight": "900", "color": "#0f172a", "marginTop": "6px"}),
                                html.Div(
                                    f"{len(data):,} real King County sales. {MODEL_NOTE}. Click any home on the map, scatter plot, or cards for deal analysis, comps, and a renovation simulator.",
                                    style={"fontSize": "13px", "color": "#64748b", "lineHeight": "1.4", "marginTop": "6px"}
                                ),
                            ],
                            className="glass-panel",
                            style={
                                "marginTop": "14px",
                                **glass_card(radius="22px", padding="14px"),
                                "background": "linear-gradient(135deg, rgba(239,246,255,0.92) 0%, rgba(248,250,252,0.92) 100%)"
                            }
                        ),

                        html.Div(
                            [
                                html.Div("Pricing Assistant", style={"fontSize": "18px", "fontWeight": "900", "color": "#0f172a"}),
                                html.Div("Deal scores, yields, mortgages, comps", style={"fontSize": "12px", "color": "#64748b", "marginTop": "4px"}),
                                html.Div(
                                    id="chat-messages",
                                    children=[make_chat_message("assistant", "Hi, I'm the housing intelligence assistant. Ask about deal scores, rent & yield, mortgage payments, days on market, or comps.")],
                                    style={
                                        "marginTop": "10px",
                                        "height": "180px",
                                        "overflowY": "auto",
                                        "overflowX": "hidden",
                                        "display": "flex",
                                        "flexDirection": "column",
                                        "boxSizing": "border-box"
                                    }
                                ),
                                dcc.Input(
                                    id="chat-input",
                                    type="text",
                                    placeholder="Is this a good rental investment?",
                                    style={
                                        "width": "100%",
                                        "marginTop": "10px",
                                        "padding": "10px 12px",
                                        "borderRadius": "14px",
                                        "border": "1px solid rgba(203,213,225,0.95)",
                                        "outline": "none",
                                        "fontSize": "12px",
                                        "boxSizing": "border-box"
                                    }
                                ),
                                html.Button(
                                    "Send",
                                    id="chat-send-btn",
                                    n_clicks=0,
                                    style={
                                        "marginTop": "8px",
                                        "width": "100%",
                                        "padding": "10px 12px",
                                        "borderRadius": "14px",
                                        "border": "none",
                                        "background": "linear-gradient(135deg, #1e40af, #60a5fa)",
                                        "color": "white",
                                        "fontWeight": "800",
                                        "cursor": "pointer",
                                        "boxShadow": "0 10px 22px rgba(59,130,246,0.24)"
                                    }
                                )
                            ],
                            className="chat-card",
                            style={
                                "marginTop": "12px",
                                "minHeight": "300px",
                                "width": "100%",
                                "boxSizing": "border-box",
                                **glass_card(radius="22px", padding="14px"),
                                "background": "linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(239,246,255,0.9) 100%)"
                            }
                        ),
                    ],
                    className="glass-panel",
                    style={
                        "width": "280px",
                        "background": "linear-gradient(180deg, rgba(255,255,255,0.72) 0%, rgba(248,250,252,0.78) 100%)",
                        "padding": "12px 10px",
                        "borderRight": "1px solid rgba(226,232,240,0.85)",
                        "position": "sticky",
                        "top": "0",
                        "boxSizing": "border-box",
                        "backdropFilter": "blur(14px)",
                        "WebkitBackdropFilter": "blur(14px)"
                    },
                ),

                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Span("AI Assisted Real Estate", className="hero-title", style={"fontSize": "38px", "fontWeight": "900", "color": "#0f172a"}),
                                                html.Span(" ✦", style={"fontSize": "24px", "color": "#7c9cff", "fontWeight": "900"})
                                            ]
                                        ),
                                        html.Div(
                                            "AI deal scoring, investor analytics, and a renovation simulator over 21,000+ real King County sales",
                                            style={"fontSize": "16px", "color": "#64748b", "marginTop": "8px"}
                                        )
                                    ]
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            [html.Span(className="sparkle-dot", style={"display": "inline-block", "width": "8px", "height": "8px", "borderRadius": "999px", "marginRight": "8px", "verticalAlign": "middle"}), MODEL_NOTE],
                                            className="floating-badge",
                                            style={
                                                "background": "linear-gradient(135deg, #eff6ff, #dbeafe)",
                                                "color": "#1d4ed8",
                                                "padding": "10px 16px",
                                                "borderRadius": "999px",
                                                "fontSize": "13px",
                                                "fontWeight": "800",
                                                "border": "1px solid #bfdbfe"
                                            }
                                        )
                                    ]
                                )
                            ],
                            style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "24px", "flexWrap": "wrap", "gap": "12px"}
                        ),

                        html.Div(id="stats-row", style={"display": "flex", "gap": "14px", "flexWrap": "wrap", "marginBottom": "18px"}),

                        html.Div(
                            [
                                html.Div(
                                    [
                                        dcc.Graph(
                                            id="map-graph",
                                            config={"displayModeBar": False, "scrollZoom": True, "responsive": True},
                                            className="animated-graph",
                                            style={"height": "590px"}
                                        )
                                    ],
                                    className="glass-panel",
                                    style={"flex": "1.45", "minWidth": "360px", "minHeight": "0", **glass_card(radius="26px", padding="12px")}
                                ),
                                html.Div(
                                    id="listing-cards",
                                    className="glass-panel",
                                    style={
                                        "flex": "1",
                                        "minWidth": "320px",
                                        "maxHeight": "620px",
                                        "overflowY": "auto",
                                        "paddingRight": "8px"
                                    }
                                ),
                            ],
                            style={"display": "flex", "gap": "18px", "alignItems": "stretch", "marginBottom": "18px", "flexWrap": "wrap"}
                        ),

                        html.Div(
                            id="property-insight-panel",
                            className="glass-panel",
                            style={**glass_card(radius="24px", padding="18px"), "marginBottom": "18px"}
                        ),

                        html.Div(
                            [
                                html.Div(dcc.Graph(id="scatter-graph", config={"displayModeBar": False, "responsive": True}, className="animated-graph", style={"width": "100%", "minWidth": "0"}), className="glass-panel", style=glass_card(radius="24px", padding="12px")),
                                html.Div(dcc.Graph(id="feature-impact-graph", config={"displayModeBar": False, "responsive": True}, className="animated-graph", style={"width": "100%", "minWidth": "0"}), className="glass-panel", style=glass_card(radius="24px", padding="12px")),
                                html.Div(dcc.Graph(id="trend-graph", config={"displayModeBar": False, "responsive": True}, className="animated-graph", style={"width": "100%", "minWidth": "0"}), className="glass-panel", style=glass_card(radius="24px", padding="12px")),
                            ],
                            style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(300px, 1fr))", "gap": "16px"}
                        ),
                    ],
                    style={"flex": "1", "minWidth": "0", "padding": "20px", "background": "transparent", "boxSizing": "border-box"}
                )
            ],
            className="page-shell", style={"display": "flex", "fontFamily": "Inter, Arial, sans-serif", "minHeight": "100vh"}
        )
    ]
)

# -----------------------------
# Tier chip callbacks
# -----------------------------
@app.callback(
    Output("tier-store", "data"),
    Input({"type": "tier-chip", "index": ALL}, "n_clicks"),
    prevent_initial_call=True
)
def update_tier(_):
    ctx = dash.callback_context
    if not ctx.triggered:
        return "ALL"
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
    return ast.literal_eval(triggered_id)["index"]


@app.callback(
    Output("tier-buttons", "children"),
    Input("tier-store", "data")
)
def render_tier_buttons(selected_tier):
    return [tier_button(value, selected_tier) for value in GRADE_TIERS]


# -----------------------------
# Selection callbacks
# -----------------------------
@app.callback(
    Output("selected-property-store", "data", allow_duplicate=True),
    Input({"type": "listing-card", "index": ALL}, "n_clicks"),
    prevent_initial_call=True
)
def select_property_from_card(_):
    ctx = dash.callback_context
    if not ctx.triggered or not any(v for v in ctx.triggered[0].values() if v):
        return dash.no_update
    if ctx.triggered[0]["value"] in (None, 0):
        return dash.no_update
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
    return ast.literal_eval(triggered_id)["index"]


@app.callback(
    Output("selected-property-store", "data"),
    Input("map-graph", "clickData"),
    Input("scatter-graph", "clickData"),
    prevent_initial_call=True
)
def select_property(map_click, scatter_click):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update
    trigger = ctx.triggered[0]["prop_id"].split(".")[0]
    if trigger == "map-graph" and map_click:
        return map_click["points"][0]["customdata"][0]
    if trigger == "scatter-graph" and scatter_click:
        return scatter_click["points"][0]["customdata"][0]
    return dash.no_update


# -----------------------------
# Chatbot callbacks
# -----------------------------
@app.callback(
    Output("chat-history-store", "data"),
    Output("chat-messages", "children"),
    Output("chat-input", "value"),
    Input("chat-send-btn", "n_clicks"),
    Input("chat-input", "n_submit"),
    State("chat-input", "value"),
    State("chat-history-store", "data"),
    State("selected-property-store", "data"),
    State("city-filter", "value"),
    State("bedroom-filter", "value"),
    State("tier-store", "data"),
    prevent_initial_call=True
)
def update_chatbot(_, __, user_text, history, selected_id, city, bedrooms, tier):
    history = history or []
    if not user_text or not user_text.strip():
        rendered = [make_chat_message(msg["role"], msg["text"]) for msg in history]
        return history, rendered, ""

    user_text = user_text.strip()
    reply = build_chat_reply(user_text, selected_id, city, bedrooms, tier)
    history = history + [
        {"role": "user", "text": user_text},
        {"role": "assistant", "text": reply}
    ]
    rendered = [make_chat_message(msg["role"], msg["text"]) for msg in history]
    return history, rendered, ""


# -----------------------------
# Main dashboard callback
# -----------------------------
MAX_MAP_POINTS = 2000
MAX_CARDS = 25


@app.callback(
    Output("stats-row", "children"),
    Output("map-graph", "figure"),
    Input("city-filter", "value"),
    Input("bedroom-filter", "value"),
    Input("tier-store", "data"),
)
def update_overview(city, bedrooms, tier):
    filtered = get_filtered_data(city, bedrooms, tier)

    undervalued_count = int((filtered["PricingLabel"] == "Undervalued").sum())
    hot_deals = int((filtered["DealScore"] >= 75).sum())

    stats = [
        make_stat_card("Listings", f"{len(filtered):,}", "Real King County sales", "#8bb5ff"),
        make_stat_card("Median Price", money(filtered["ListedPrice"].median()), f"{money(filtered['PricePerSqft'].median())}/sq ft", "#9c9df4"),
        make_stat_card("AI Median Estimate", money(filtered["PredictedPrice"].median()), MODEL_NOTE, "#7adab3"),
        make_stat_card("Hot Deals", f"{hot_deals:,}", f"Deal Score 75+ • {undervalued_count:,} undervalued", "#f5b06f"),
    ]

    map_df = filtered
    if len(map_df) > MAX_MAP_POINTS:
        map_df = map_df.sample(MAX_MAP_POINTS, random_state=7)

    map_fig = px.scatter_mapbox(
        map_df,
        lat="lat",
        lon="long",
        color="PricingLabel",
        hover_name="Address",
        hover_data={
            "City": True,
            "Bedrooms": True,
            "DealScore": True,
            "ListedPrice": ":,.0f",
            "PredictedPrice": ":,.0f",
            "lat": False,
            "long": False,
        },
        custom_data=["Id"],
        zoom=8.6,
        height=565,
        color_discrete_map={
            "Fairly priced": "#22c55e",
            "Overpriced": "#f97316",
            "Undervalued": "#f59e0b",
        },
    )
    map_fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_center={"lat": 47.45, "lon": -122.15},
        margin=dict(l=0, r=0, t=8, b=0),
        paper_bgcolor="white",
        plot_bgcolor="white",
        uirevision="map-stable",
        legend_title_text="",
        legend=dict(orientation="h", yanchor="bottom", y=0.98, xanchor="left", x=0.02, bgcolor="rgba(255,255,255,0.7)"),
        clickmode="event+select"
    )
    map_fig.update_traces(marker={"opacity": 0.82, "size": 9}, selector=dict(mode="markers"))

    return stats, map_fig


@app.callback(
    Output("listing-cards", "children"),
    Output("scatter-graph", "figure"),
    Output("feature-impact-graph", "figure"),
    Output("trend-graph", "figure"),
    Input("city-filter", "value"),
    Input("bedroom-filter", "value"),
    Input("tier-store", "data"),
    Input("selected-property-store", "data"),
)
def update_selection_views(city, bedrooms, tier, selected_id):
    filtered = get_filtered_data(city, bedrooms, tier)

    top_cards = filtered.nlargest(MAX_CARDS, "DealScore")
    if selected_id is not None and selected_id in filtered["Id"].values and selected_id not in top_cards["Id"].values:
        top_cards = pd.concat([filtered[filtered["Id"] == selected_id], top_cards])
    if selected_id is not None and selected_id in top_cards["Id"].values:
        sel_df = top_cards[top_cards["Id"] == selected_id]
        top_cards = pd.concat([sel_df, top_cards[top_cards["Id"] != selected_id]])
    cards = [
        html.Div(
            f"Top {min(MAX_CARDS, len(filtered))} homes by Deal Score ({len(filtered):,} match your filters)",
            style={"fontSize": "13px", "fontWeight": "800", "color": "#64748b", "margin": "4px 4px 12px"}
        )
    ] + [make_listing_card(row, selected_id=selected_id) for _, row in top_cards.iterrows()]

    if selected_id is not None and selected_id in data["Id"].values:
        selected_row = data[data["Id"] == selected_id].iloc[0]
    else:
        selected_row = filtered.iloc[0]

    scatter_df = filtered if len(filtered) <= 1500 else filtered.sample(1500, random_state=7)
    scatter_fig = px.scatter(
        scatter_df,
        x="sqft_living",
        y="ListedPrice",
        color="PricingLabel",
        hover_name="Address",
        custom_data=["Id"],
        title=f"Market Position: {selected_row['Address']}",
        labels={"sqft_living": "Living Area (sq ft)", "ListedPrice": "Listed Price"},
        color_discrete_map={
            "Fairly priced": "#22c55e",
            "Overpriced": "#f97316",
            "Undervalued": "#f59e0b",
        },
        opacity=0.45
    )
    scatter_fig.add_scatter(
        x=[selected_row["sqft_living"]],
        y=[selected_row["ListedPrice"]],
        mode="markers+text",
        text=["Selected"],
        textposition="top center",
        marker=dict(size=22, color="#111827", symbol="star"),
        name="Selected Property"
    )
    scatter_fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=10, r=10, t=50, b=10),
        title_font_size=18,
        legend_title_text="",
        clickmode="event+select"
    )

    compare_df = pd.DataFrame({
        "Feature": ["Living Area (00s sqft)", "Grade (of 13)", "Condition (of 5)", "Bathrooms", "Age (decades)"],
        "Selected Property": [
            selected_row["sqft_living"] / 100,
            selected_row["grade"],
            selected_row["condition"],
            selected_row["bathrooms"],
            selected_row["Age"] / 10,
        ],
        "County Average": [
            data["sqft_living"].mean() / 100,
            data["grade"].mean(),
            data["condition"].mean(),
            data["bathrooms"].mean(),
            data["Age"].mean() / 10,
        ]
    })
    compare_long = compare_df.melt(id_vars="Feature", value_vars=["Selected Property", "County Average"], var_name="Type", value_name="Value")
    feature_fig = px.bar(
        compare_long,
        x="Feature",
        y="Value",
        color="Type",
        barmode="group",
        title=f"Feature Comparison: {selected_row['Address']}",
    )
    feature_fig.update_layout(paper_bgcolor="white", plot_bgcolor="white", margin=dict(l=10, r=10, t=50, b=10), title_font_size=18)

    sel_city = selected_row["City"]
    city_trend = data[data["City"] == sel_city].groupby("SaleMonth")["ListedPrice"].median()
    trend_fig = go.Figure()
    trend_fig.add_trace(go.Scatter(
        x=list(county_trend.index), y=list(county_trend.values),
        mode="lines+markers", name="King County median",
        line=dict(color="#94a3b8", width=2, dash="dot")
    ))
    trend_fig.add_trace(go.Scatter(
        x=list(city_trend.index), y=list(city_trend.values),
        mode="lines+markers", name=f"{sel_city} median",
        line=dict(color="#3b82f6", width=3)
    ))
    trend_fig.update_layout(
        title=f"Real Sale-Price Trend: {sel_city} vs County",
        paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=10, r=10, t=50, b=10), title_font_size=18,
        legend=dict(orientation="h", y=1.02, x=0),
        yaxis_tickformat="$,.0f"
    )

    return cards, scatter_fig, feature_fig, trend_fig


# -----------------------------
# Property insight panel (deal analysis, investor metrics, comps, what-if)
# -----------------------------
@app.callback(
    Output("property-insight-panel", "children"),
    Input("selected-property-store", "data"),
    Input("city-filter", "value"),
    Input("bedroom-filter", "value"),
    Input("tier-store", "data"),
)
def render_property_insights(selected_id, city, bedrooms, tier):
    filtered = get_filtered_data(city, bedrooms, tier)

    if selected_id is None or selected_id not in data["Id"].values:
        top = filtered.nlargest(1, "DealScore").iloc[0]
        return html.Div(
            [
                html.Div("Deal Analysis", style={"fontSize": "22px", "fontWeight": "900", "color": "#0f172a"}),
                html.Div("Select a home to see its Deal Score, investor metrics, comparable sales, and the renovation simulator.", style={"color": "#64748b", "fontSize": "15px", "marginTop": "6px"}),
                html.Div(
                    [
                        make_mini_metric("Best Deal Right Now", top["Address"], f"{top['City']} • Deal Score {top['DealScore']}/100"),
                        make_mini_metric("Listed Price", money(top["ListedPrice"])),
                        make_mini_metric("AI Price Estimate", money(top["PredictedPrice"])),
                        make_mini_metric("Est. Gross Yield", f"{top['GrossYield']}%"),
                    ],
                    style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(160px, 1fr))", "gap": "14px", "marginTop": "18px"}
                )
            ]
        )

    row = data[data["Id"] == selected_id].iloc[0]
    gap_pct = row["GapPct"] * 100
    reasons = build_explanation(row)
    comps = find_comps(row, 4)

    comp_cards = [
        html.Div(
            [
                html.Div(c["Address"], style={"fontWeight": "800", "fontSize": "14px", "color": "#0f172a"}),
                html.Div(f"{c['City']} • {int(c['Bedrooms'])} bd • {int(c['sqft_living']):,} sqft • grade {int(c['grade'])}",
                         style={"fontSize": "12px", "color": "#64748b", "marginTop": "3px"}),
                html.Div(money(c["ListedPrice"]), style={"fontSize": "17px", "fontWeight": "900", "color": "#1e40af", "marginTop": "6px"}),
                html.Div(f"{money(c['PricePerSqft'])}/sqft", style={"fontSize": "11px", "color": "#94a3b8"}),
            ],
            style={
                "padding": "14px",
                "borderRadius": "16px",
                "background": "rgba(248,250,252,0.9)",
                "border": "1px solid rgba(226,232,240,0.9)"
            }
        )
        for _, c in comps.iterrows()
    ]

    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(row["Address"], style={"margin": "0", "color": "#0f172a", "fontSize": "28px", "fontWeight": "900"}),
                            html.Div(
                                f"{row['City']} (ZIP {int(row['zipcode'])}) • {int(row['Bedrooms'])} bed • {row['bathrooms']} bath • {int(row['sqft_living']):,} sq ft • built {int(row['yr_built'])}",
                                style={"color": "#64748b", "marginTop": "4px", "fontSize": "15px"}
                            )
                        ]
                    ),
                    html.Div(
                        [
                            deal_badge(row["DealScore"], size="big"),
                            html.Div(
                                row["PricingLabel"],
                                style={
                                    "background": label_bg(row["PricingLabel"]),
                                    "color": label_color(row["PricingLabel"]),
                                    "padding": "9px 14px",
                                    "borderRadius": "999px",
                                    "fontWeight": "800",
                                    "height": "fit-content",
                                    "border": f"1px solid {label_color(row['PricingLabel'])}22"
                                }
                            )
                        ],
                        style={"display": "flex", "gap": "12px", "alignItems": "center"}
                    )
                ],
                style={"display": "flex", "justifyContent": "space-between", "alignItems": "flex-start", "marginBottom": "18px"}
            ),

            html.Div(
                [
                    make_mini_metric("Listed Price", money(row["ListedPrice"]), f"{money(row['PricePerSqft'])}/sq ft"),
                    make_mini_metric("AI Price Estimate", money(row["PredictedPrice"]), f"Gap: {gap_pct:+.1f}%"),
                    make_mini_metric("Est. Days on Market", f"{int(row['EstDOM'])} days", "Model estimate"),
                    make_mini_metric("Est. Rent / Yield", f"{money(row['RentEst'])}/mo", f"{row['GrossYield']}% gross yield"),
                    make_mini_metric("Mortgage (20% down)", f"{money(row['Mortgage'])}/mo", "30-yr @ 6.5%"),
                ],
                style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(160px, 1fr))", "gap": "14px", "marginBottom": "22px"}
            ),

            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Why the model prices it this way", style={"fontSize": "18px", "fontWeight": "900", "color": "#0f172a", "marginBottom": "10px"}),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Span(f"0{i+1}", style={"fontSize": "13px", "fontWeight": "900", "color": "#1e40af", "marginRight": "10px"}),
                                            html.Span(reason, style={"color": "#334155", "lineHeight": "1.6", "fontSize": "14px"})
                                        ],
                                        style={"marginBottom": "12px"}
                                    ) for i, reason in enumerate(reasons)
                                ]
                            )
                        ],
                        style={"flex": "1.1", "minWidth": "280px"}
                    ),
                    html.Div(
                        [
                            html.Div("Renovation Simulator", style={"fontSize": "18px", "fontWeight": "900", "color": "#0f172a"}),
                            html.Div("What would upgrades do to the AI estimate?", style={"fontSize": "12px", "color": "#64748b", "marginBottom": "12px"}),
                            html.Div("Upgrade construction grade by", style={"fontSize": "12px", "fontWeight": "800", "color": "#475569"}),
                            dcc.Slider(id="whatif-grade", min=0, max=3, step=1, value=0,
                                       marks={i: f"+{i}" for i in range(4)}),
                            html.Div("Add living space (sq ft)", style={"fontSize": "12px", "fontWeight": "800", "color": "#475569", "marginTop": "10px"}),
                            dcc.Slider(id="whatif-sqft", min=0, max=1000, step=250, value=0,
                                       marks={i: f"+{i}" for i in range(0, 1001, 250)}),
                            html.Div(id="whatif-output", style={"marginTop": "14px"})
                        ],
                        style={
                            "flex": "1",
                            "minWidth": "280px",
                            "padding": "16px",
                            "borderRadius": "18px",
                            "background": "linear-gradient(135deg, rgba(239,246,255,0.9), rgba(248,250,252,0.95))",
                            "border": "1px solid rgba(191,219,254,0.8)"
                        }
                    )
                ],
                style={"display": "flex", "gap": "20px", "marginBottom": "22px", "flexWrap": "wrap"}
            ),

            html.Div(
                [
                    html.Div("Comparable Sales Nearby", style={"fontSize": "18px", "fontWeight": "900", "color": "#0f172a", "marginBottom": "12px"}),
                    html.Div(comp_cards, style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(180px, 1fr))", "gap": "12px"})
                ]
            )
        ]
    )


# -----------------------------
# What-if renovation simulator
# -----------------------------
@app.callback(
    Output("whatif-output", "children"),
    Input("whatif-grade", "value"),
    Input("whatif-sqft", "value"),
    State("selected-property-store", "data"),
)
def run_whatif(grade_up, sqft_add, selected_id):
    if selected_id is None or selected_id not in data["Id"].values:
        return html.Div("Select a home first.", style={"fontSize": "12px", "color": "#94a3b8"})

    row = data[data["Id"] == selected_id].iloc[0]
    modified = row.copy()
    modified["grade"] = min(13, row["grade"] + (grade_up or 0))
    modified["sqft_living"] = row["sqft_living"] + (sqft_add or 0)
    new_price = predict_price(modified)
    delta = new_price - row["PredictedPrice"]

    return html.Div(
        [
            html.Div("New AI estimate", style={"fontSize": "11px", "color": "#64748b", "fontWeight": "800"}),
            html.Div(money(new_price), style={"fontSize": "24px", "fontWeight": "900", "color": "#0f172a"}),
            html.Div(
                f"{'+' if delta >= 0 else ''}{money(delta)} vs current estimate",
                style={"fontSize": "13px", "fontWeight": "800", "color": "#16a34a" if delta >= 0 else "#ef4444", "marginTop": "2px"}
            ),
        ]
    )


# -----------------------------
# Real-time AI insights on selection
# -----------------------------
@app.callback(
    Output("chat-history-store", "data", allow_duplicate=True),
    Output("chat-messages", "children", allow_duplicate=True),
    Input("selected-property-store", "data"),
    State("chat-history-store", "data"),
    prevent_initial_call=True
)
def auto_generate_insights(selected_id, history):
    if selected_id is None or selected_id not in data["Id"].values:
        return dash.no_update, dash.no_update

    history = history or []
    insights = generate_ai_insights(selected_id)
    if not insights:
        return dash.no_update, dash.no_update

    insight_text = " ".join(insights)
    history = history + [{"role": "assistant", "text": f"🤖 Real-time: {insight_text}"}]
    rendered = [make_chat_message(msg["role"], msg["text"]) for msg in history]
    return history, rendered


if __name__ == "__main__":
    app.run(debug=True)
