import os
import ast
import numpy as np
import pandas as pd
import dash
from dash import dcc, html, Input, Output, State
from dash.dependencies import ALL
import plotly.express as px

# -----------------------------
# Load and prepare data
# -----------------------------
data = pd.read_csv("train.csv")

pred_path = "submission.csv"
if os.path.exists(pred_path):
    pred = pd.read_csv(pred_path).rename(columns={"SalePrice": "PredictedPrice"})
    data = data.merge(pred[["Id", "PredictedPrice"]], on="Id", how="left")
else:
    data["PredictedPrice"] = np.nan

if data["PredictedPrice"].isna().all():
    quality_adj = (data["OverallQual"].fillna(5) - 5) * 0.025
    area_adj = (
        (data["GrLivArea"].fillna(data["GrLivArea"].median()) - data["GrLivArea"].median())
        / data["GrLivArea"].median()
    ) * 0.06
    age = data["YrSold"].fillna(2010) - data["YearBuilt"].fillna(data["YearBuilt"].median())
    age_adj = np.where(age > 50, -0.04, np.where(age < 10, 0.03, 0.0))
    demo_factor = 1 + quality_adj + area_adj + age_adj
    data["PredictedPrice"] = (data["SalePrice"] * demo_factor).clip(lower=50000)
else:
    data["PredictedPrice"] = data["PredictedPrice"].fillna(data["SalePrice"] * 0.98)

data["ListedPrice"] = data["SalePrice"]
data["PriceGap"] = data["ListedPrice"] - data["PredictedPrice"]
data["GapPct"] = data["PriceGap"] / data["PredictedPrice"]
data["PropertyType"] = data["BldgType"].fillna("House")
data["Bedrooms"] = data["BedroomAbvGr"].fillna(0).astype(int)
data["Age"] = data["YrSold"].fillna(2010) - data["YearBuilt"].fillna(data["YearBuilt"].median())


def pricing_label(x):
    if x > 0.08:
        return "Overpriced"
    elif x < -0.08:
        return "Undervalued"
    return "Fairly priced"


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


data["PricingLabel"] = data["GapPct"].apply(pricing_label)

PROPERTY_LABELS = {
    "ALL": "All Homes",
    "1Fam": "Single Family",
    "2fmCon": "Two-Family",
    "Duplex": "Duplex",
    "Twnhs": "Townhouse",
    "TwnhsE": "End-Unit Townhome"
}

lat_map = {
    'CollgCr': 42.025, 'Veenker': 42.030, 'Crawfor': 42.022, 'NoRidge': 42.050,
    'Mitchel': 42.018, 'Somerst': 42.031, 'NWAmes': 42.040, 'OldTown': 42.016,
    'BrkSide': 42.012, 'Sawyer': 42.025, 'IDOTRR': 42.007, 'MeadowV': 42.010,
    'Edwards': 42.017, 'Timber': 42.042, 'Gilbert': 42.080, 'StoneBr': 42.055,
    'ClearCr': 42.045, 'NPkVill': 42.036, 'Blmngtn': 42.062, 'Blueste': 42.065,
    'SawyerW': 42.028, 'SWISU': 42.013, 'NridgHt': 42.054, 'NAmes': 42.053
}
lon_map = {
    'CollgCr': -93.655, 'Veenker': -93.650, 'Crawfor': -93.670, 'NoRidge': -93.635,
    'Mitchel': -93.685, 'Somerst': -93.645, 'NWAmes': -93.680, 'OldTown': -93.630,
    'BrkSide': -93.620, 'Sawyer': -93.625, 'IDOTRR': -93.610, 'MeadowV': -93.600,
    'Edwards': -93.640, 'Timber': -93.690, 'Gilbert': -93.750, 'StoneBr': -93.630,
    'ClearCr': -93.670, 'NPkVill': -93.665, 'Blmngtn': -93.690, 'Blueste': -93.700,
    'SawyerW': -93.645, 'SWISU': -93.620, 'NridgHt': -93.640, 'NAmes': -93.675
}
data["Latitude"] = data["Neighborhood"].map(lat_map).fillna(42.03)
data["Longitude"] = data["Neighborhood"].map(lon_map).fillna(-93.65)

street_names = [
    "Main St", "Oak Ave", "Pine St", "Maple Dr", "Cedar Ln",
    "Elm St", "Park Ave", "Hill Dr", "Lakeview Rd", "Sunset Blvd"
]
data["Address"] = data.apply(
    lambda r: f"{100 + int(r['Id']) % 900} {street_names[int(r['Id']) % len(street_names)]}",
    axis=1
)

PROPERTY_TYPES = ["ALL"] + sorted(data["PropertyType"].dropna().unique().tolist())

# -----------------------------
# Helpers
# -----------------------------
def money(x):
    return f"${x:,.0f}"


def display_property_type(value):
    return PROPERTY_LABELS.get(value, value)


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


def make_listing_card(row, selected_id=None):
    badge = row["PricingLabel"]
    badge_color = label_color(badge)
    is_selected = selected_id == row["Id"]

    return html.Button(
        [
            html.Div(
                "🏡",
                style={
                    "width": "102px",
                    "height": "102px",
                    "borderRadius": "20px",
                    "background": "linear-gradient(135deg, #dbeafe 0%, #bfdbfe 50%, #ffffff 100%)",
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "center",
                    "fontSize": "36px",
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
                                        style={
                                            "fontSize": "22px",
                                            "fontWeight": "900",
                                            "color": "#0f172a",
                                            "lineHeight": "1.1"
                                        }
                                    ),
                                    html.Div(
                                        f"{row['Neighborhood']} • {int(row['Bedrooms'])} bed • {display_property_type(row['PropertyType'])}",
                                        style={
                                            "fontSize": "14px",
                                            "color": "#64748b",
                                            "marginTop": "7px",
                                            "fontWeight": "600"
                                        }
                                    ),
                                ],
                                style={"flex": "1"}
                            ),
                            html.Div(
                                [
                                    html.Span(
                                        "Selected" if is_selected else badge,
                                        style={
                                            "background": label_bg(badge) if not is_selected else "rgba(59,130,246,0.14)",
                                            "color": badge_color if not is_selected else "#1e40af",
                                            "padding": "7px 12px",
                                            "borderRadius": "999px",
                                            "fontSize": "12px",
                                            "fontWeight": "800",
                                            "border": f"1px solid {badge_color}22" if not is_selected else "1px solid rgba(59,130,246,0.18)"
                                        },
                                    )
                                ]
                            )
                        ],
                        style={"display": "flex", "justifyContent": "space-between", "gap": "12px", "alignItems": "flex-start"},
                    ),
                    html.Div(
                        [
                            make_mini_metric("Listed Price", money(row["ListedPrice"])),
                            make_mini_metric("AI Price Estimate", money(row["PredictedPrice"])),
                        ],
                        style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px", "marginTop": "16px"}
                    )
                ],
                style={"flex": "1", "textAlign": "left"},
            ),
        ],
        id={"type": "listing-card", "index": int(row["Id"])} ,
        n_clicks=0,
        className="listing-hover",
        style={
            "display": "flex",
            "gap": "16px",
            "padding": "18px",
            "width": "100%",
            "background": "linear-gradient(180deg, rgba(255,255,255,0.95) 0%, rgba(248,250,252,0.9) 100%)",
            "borderRadius": "24px",
            "boxShadow": "0 12px 30px rgba(15, 23, 42, 0.08)",
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


def property_button(value, selected_value):
    selected = value == selected_value
    label = display_property_type(value)

    return html.Button(
        label,
        id={"type": "property-chip", "index": value},
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
            "minWidth": "90px",
            "textAlign": "center",
            "width": "100%"
        },
        className="chip-hover"
    )


def build_explanation(row):
    reasons = []
    if row["GrLivArea"] >= data["GrLivArea"].median():
        reasons.append("Larger living area supports a higher estimated value.")
    else:
        reasons.append("Smaller living area pulls the estimate down compared to many homes.")

    if row["OverallQual"] >= data["OverallQual"].median():
        reasons.append("Above-average quality rating helps the predicted price.")
    else:
        reasons.append("Below-average quality rating lowers the predicted price.")

    if row["Age"] >= data["Age"].median():
        reasons.append("The property is older, which can reduce price compared to newer homes.")
    else:
        reasons.append("The property is relatively newer, which supports value.")

    if row["GapPct"] > 0.08:
        reasons.append("The listed price is noticeably above the AI estimate.")
    elif row["GapPct"] < -0.08:
        reasons.append("The listed price is below the AI estimate, which may suggest a value opportunity.")
    else:
        reasons.append("The listed price is close to the AI estimate.")
    return reasons[:4]


def get_filtered_data(neighborhood, bedrooms, property_type):
    filtered = data.copy()
    if neighborhood != "ALL":
        filtered = filtered[filtered["Neighborhood"] == neighborhood]
    if bedrooms != -1:
        filtered = filtered[filtered["Bedrooms"] == bedrooms]
    if property_type != "ALL":
        filtered = filtered[filtered["PropertyType"] == property_type]
    if filtered.empty:
        filtered = data.head(10).copy()
    return filtered


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


def generate_ai_insights(selected_id, neighborhood, bedrooms, property_type):
    """Generate real-time AI insights for the selected property"""
    filtered = get_filtered_data(neighborhood, bedrooms, property_type)
    
    if selected_id is not None and selected_id in data["Id"].values:
        row = data[data["Id"] == selected_id].iloc[0]
    else:
        return None
    
    insights = []
    gap_pct = row["GapPct"] * 100
    neigh_avg = data[data["Neighborhood"] == row["Neighborhood"]]["ListedPrice"].mean()
    
    # Deal Alert
    if gap_pct < -15:
        insights.append(f"🚀 Great Deal! This home is {abs(gap_pct):.1f}% below market estimate.")
    elif gap_pct > 15:
        insights.append(f"⚠️ Overpriced Alert: {gap_pct:.1f}% above the AI estimate.")
    
    # Value Analysis
    if row["GrLivArea"] > data["GrLivArea"].quantile(0.75):
        insights.append(f"💪 Spacious: {row['GrLivArea']:.0f} sq ft (top 25% in dataset).")
    
    # Market Position
    price_diff = row["ListedPrice"] - neigh_avg
    if abs(price_diff) > data["ListedPrice"].std():
        insights.append(f"📍 Neighborhood Position: {abs(price_diff/1000):.1f}K {'above' if price_diff > 0 else 'below'} neighborhood avg.")
    
    # Condition Assessment
    if row["OverallQual"] >= 8:
        insights.append(f"✨ Premium Quality: Rating {row['OverallQual']}/10")
    elif row["OverallQual"] <= 4:
        insights.append(f"🔧 Renovation Opportunity: Quality rating {row['OverallQual']}/10")
    
    return insights[:3]


def build_chat_reply(user_text, selected_id, neighborhood, bedrooms, property_type):
    question = (user_text or "").strip().lower()
    filtered = get_filtered_data(neighborhood, bedrooms, property_type)

    if selected_id is not None and selected_id in data["Id"].values:
        row = data[data["Id"] == selected_id].iloc[0]
    else:
        row = filtered.iloc[0]

    neigh_avg = data[data["Neighborhood"] == row["Neighborhood"]]["ListedPrice"].mean()
    gap_pct = row["GapPct"] * 100

    if any(word in question for word in ["why", "reason", "label", "labeled"]):
        return " ".join(build_explanation(row)[:2])

    if any(word in question for word in ["price", "estimate", "listed", "gap", "deal", "bargain", "value"]):
        response = (
            f"{row['Address']} is listed at {money(row['ListedPrice'])}. "
            f"The AI estimate is {money(row['PredictedPrice'])}, so the price gap is {gap_pct:.1f}%. "
            f"Right now it is labeled {row['PricingLabel'].lower()}."
        )
        if abs(gap_pct) > 10:
            response += f" This is a significant {'premium' if gap_pct > 0 else 'opportunity'}."
        return response

    if any(word in question for word in ["average", "market", "neighborhood", "compare"]):
        return (
            f"For {row['Neighborhood']}, the neighborhood average is {money(neigh_avg)}. "
            f"This home has {row['GrLivArea']:.0f} sq ft, quality {row['OverallQual']}, and age {row['Age']:.0f} years. "
            f"That helps explain how it compares to nearby homes."
        )

    if any(word in question for word in ["filter", "filtered", "results", "homes"]):
        return (
            f"Your current filters show {len(filtered)} homes. "
            f"Neighborhood: {neighborhood if neighborhood != 'ALL' else 'All'}, "
            f"Bedrooms: {bedrooms if bedrooms != -1 else 'Any'}, "
            f"Property type: {display_property_type(property_type)}."
        )

    if any(word in question for word in ["recommend", "suggest", "best", "advice"]):
        best_deals = filtered.nsmallest(3, "GapPct")[["Address", "ListedPrice", "GapPct"]]
        if len(best_deals) > 0:
            top_deal = best_deals.iloc[0]
            return f"💡 Based on value: {top_deal['Address']} at {money(top_deal['ListedPrice'])} (gap: {top_deal['GapPct']*100:.1f}%)"
        return "Not enough data for recommendations yet."

    return (
        f"You can ask me about price, deals, market comparisons, or recommendations. "
        f"Right now I am focused on {row['Address']}, which is marked {row['PricingLabel'].lower()}."
    )


# -----------------------------
# App setup
# -----------------------------
app = dash.Dash(__name__)
app.title = "Real Estate Analytics"
app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
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
            .page-shell {
                animation: fadeSlideUp 0.8s ease;
            }
            .hero-title {
                animation: fadeSlideUp 0.9s ease;
            }
            .floating-badge {
                animation: floatPulse 3.8s ease-in-out infinite;
            }
            .hover-card, .listing-hover, .chip-hover, .chat-card, .glass-panel {
                animation: fadeSlideUp 0.7s ease;
            }
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
            .glass-panel:hover, .chat-card:hover {
                box-shadow: 0 22px 52px rgba(15, 23, 42, 0.10) !important;
            }
            .sparkle-dot {
                background: linear-gradient(90deg, rgba(255,255,255,0.4), rgba(255,255,255,0.95), rgba(255,255,255,0.4));
                background-size: 200% 100%;
                animation: shimmer 2.8s linear infinite;
            }
            .dash-graph {
                transition: transform 0.25s ease, box-shadow 0.25s ease;
            }
            .dash-graph:hover {
                transform: translateY(-4px);
            }
            * {
                scroll-behavior: smooth;
            }
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
        
        dcc.Store(id="property-type-store", data="ALL"),
        dcc.Store(id="selected-property-store"),
        dcc.Store(
            id="chat-history-store",
            data=[
                {
                    "role": "assistant",
                    "text": "Hi, I am the pricing assistant. Ask about the selected home, price gap, market average, or current filters."
                }
            ]
        ),

        html.Div(
            [
                html.Div(
                    [
                        html.Div("Filters", style={"fontSize": "22px", "fontWeight": "900", "color": "#0f172a", "marginBottom": "16px"}),

                        html.Div("Neighborhood", style={"fontSize": "13px", "fontWeight": "800", "marginBottom": "6px", "color": "#475569"}),
                        dcc.Dropdown(
                            id="neighborhood-filter",
                            options=[{"label": "All Neighborhoods", "value": "ALL"}] +
                                    [{"label": n, "value": n} for n in sorted(data["Neighborhood"].dropna().unique())],
                            value="ALL",
                            clearable=False,
                            style={"marginBottom": "16px"}
                        ),

                        html.Div("Bedrooms", style={"fontSize": "13px", "fontWeight": "800", "marginBottom": "6px", "color": "#475569"}),

                        dcc.Dropdown(
                            id="bedroom-filter",
                            options=[{"label": "Any", "value": -1}] +
                                    [{"label": f"{b} Bedrooms", "value": int(b)} for b in sorted(data["Bedrooms"].dropna().unique())],
                            value=-1,
                            clearable=False,
                            style={"marginBottom": "16px"}
                        ),

                        html.Div("Property Type", style={"fontSize": "13px", "fontWeight": "800", "marginBottom": "4px", "color": "#475569"}),
                        html.Div(
                            "Choose the kind of home you want to view.",
                            style={"fontSize": "11px", "color": "#64748b", "marginBottom": "8px"}
                        ),
                        html.Div(
                            id="property-type-buttons",
                            style={"display": "grid", "gridTemplateColumns": "repeat(3, minmax(80px, 1fr))", "gap": "6px", "marginBottom": "8px"}
                        ),

                        html.Div(
                            [
                                html.Div("Project demo", style={"fontSize": "12px", "color": "#94a3b8", "fontWeight": "800", "textTransform": "uppercase", "letterSpacing": "0.08em"}),
                                html.Div("AI-assisted pricing support", style={"fontSize": "18px", "fontWeight": "900", "color": "#0f172a", "marginTop": "6px"}),
                                html.Div(
                                    "Click a property on the map, scatter plot, or listing card to see detailed pricing insights.",
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
                                html.Div("Quick Q&A for the selected property", style={"fontSize": "12px", "color": "#64748b", "marginTop": "4px"}),
                                html.Div(
                                    id="chat-messages",
                                    children=[make_chat_message("assistant", "Hi, I am the pricing assistant. Ask about the selected home, price gap, market average, or current filters.")],
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
                                    placeholder="Ask about this home...",
                                    style={
                                        "width": "100%",
                                        "marginTop": "10px",
                                        "padding": "10px 12px",
                                        "borderRadius": "14px",
                                        "border": "1px solid rgba(203,213,225,0.95)",
                                        "outline": "none",
                                        "fontSize": "12px"
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
                        "minHeight": "auto",
                        "maxHeight": "auto",
                        "overflowY": "visible",
                        "overflowX": "hidden",
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
                                                html.Span("Real Estate Analytics", className="hero-title", style={"fontSize": "38px", "fontWeight": "900", "color": "#0f172a"}),
                                                html.Span(" ✦", style={"fontSize": "24px", "color": "#7c9cff", "fontWeight": "900"})
                                            ]
                                        ),
                                        html.Div(
                                            "AI-assisted pricing dashboard for listing comparison and decision support",
                                            style={"fontSize": "16px", "color": "#64748b", "marginTop": "8px"}
                                        )
                                    ]
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            [html.Span(className="sparkle-dot", style={"display": "inline-block", "width": "8px", "height": "8px", "borderRadius": "999px", "marginRight": "8px", "verticalAlign": "middle"}), "Connected • Demo mode"],
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
                            style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "24px"}
                        ),

                        html.Div(id="stats-row", style={"display": "flex", "gap": "14px", "flexWrap": "wrap", "marginBottom": "18px"}),

                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Div("Map Zoom", style={"fontWeight": "800", "marginBottom": "6px", "color": "#334155"}),
                                                dcc.Slider(
                                                    id="map-zoom-slider",
                                                    min=8,
                                                    max=16,
                                                    step=0.5,
                                                    value=11,
                                                    marks={i: str(i) for i in range(8, 17, 2)},
                                                    tooltip={"placement": "bottom", "always_visible": False}
                                                ),
                                            ],
                                            style={
                                                "marginBottom": "12px",
                                                "padding": "12px",
                                                "background": "rgba(248,250,252,0.9)",
                                                "borderRadius": "14px"
                                            }
                                        ),
                                        dcc.Graph(
                                            id="map-graph",
                                            config={"displayModeBar": False, "scrollZoom": True},
                                            className="animated-graph",
                                            style={"height": "530px"}
                                        )
                                    ],
                                    className="glass-panel",
                                    style={"flex": "1.45", **glass_card(radius="26px", padding="12px")}
                                ),
                                html.Div(
                                    id="listing-cards",
                                    className="glass-panel",
                                    style={
                                        "flex": "1",
                                        "maxHeight": "620px",
                                        "overflowY": "auto",
                                        "paddingRight": "8px"
                                    }
                                ),
                            ],
                            style={"display": "flex", "gap": "18px", "alignItems": "stretch", "marginBottom": "18px"}
                        ),

                        html.Div(
                            id="property-insight-panel",
                            className="glass-panel",
                            style={**glass_card(radius="24px", padding="18px"), "marginBottom": "18px"}
                        ),

                        html.Div(
                            [
                                html.Div(dcc.Graph(id="scatter-graph", config={"displayModeBar": False}, className="animated-graph"), className="glass-panel", style=glass_card(radius="24px", padding="12px")),
                                html.Div(dcc.Graph(id="feature-impact-graph", config={"displayModeBar": False}, className="animated-graph"), className="glass-panel", style=glass_card(radius="24px", padding="12px")),
                                html.Div(dcc.Graph(id="trend-graph", config={"displayModeBar": False}, className="animated-graph"), className="glass-panel", style=glass_card(radius="24px", padding="12px")),
                            ],
                            style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr", "gap": "16px"}
                        ),
                    ],
                    style={"flex": "1", "padding": "20px", "background": "transparent"}
                )
            ],
            className="page-shell", style={"display": "flex", "fontFamily": "Inter, Arial, sans-serif", "minHeight": "100vh"}
        )
    ]
)

# -----------------------------
# Property type chip callbacks
# -----------------------------
@app.callback(
    Output("property-type-store", "data"),
    Input({"type": "property-chip", "index": ALL}, "n_clicks"),
    prevent_initial_call=True
)
def update_property_type(_):
    ctx = dash.callback_context
    if not ctx.triggered:
        return "ALL"
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
    return ast.literal_eval(triggered_id)["index"]


@app.callback(
    Output("property-type-buttons", "children"),
    Input("property-type-store", "data")
)
def render_property_buttons(selected_property):
    return [property_button(value, selected_property) for value in PROPERTY_TYPES]


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
    if not ctx.triggered:
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
    State("neighborhood-filter", "value"),
    State("bedroom-filter", "value"),
    State("property-type-store", "data"),
    prevent_initial_call=True
)
def update_chatbot(_, __, user_text, history, selected_id, neighborhood, bedrooms, property_type):
    history = history or []
    if not user_text or not user_text.strip():
        rendered = [make_chat_message(msg["role"], msg["text"]) for msg in history]
        return history, rendered, ""

    user_text = user_text.strip()
    reply = build_chat_reply(user_text, selected_id, neighborhood, bedrooms, property_type)
    history = history + [
        {"role": "user", "text": user_text},
        {"role": "assistant", "text": reply}
    ]
    rendered = [make_chat_message(msg["role"], msg["text"]) for msg in history]
    return history, rendered, ""


# -----------------------------
# Main dashboard callback
# -----------------------------
@app.callback(
    Output("stats-row", "children"),
    Output("map-graph", "figure"),
    Output("listing-cards", "children"),
    Output("scatter-graph", "figure"),
    Output("feature-impact-graph", "figure"),
    Output("trend-graph", "figure"),
    Input("neighborhood-filter", "value"),
    Input("bedroom-filter", "value"),
    Input("property-type-store", "data"),
    Input("selected-property-store", "data"),
    Input("map-zoom-slider", "value"),
)
def update_dashboard(neighborhood, bedrooms, property_type, selected_id, zoom_value):
    filtered = get_filtered_data(neighborhood, bedrooms, property_type)


    overpriced_count = int((filtered["PricingLabel"] == "Overpriced").sum())
    fair_count = int((filtered["PricingLabel"] == "Fairly priced").sum())

    stats = [
        make_stat_card("Listings", f"{len(filtered):,}", "Current filtered view", "#8bb5ff"),
        make_stat_card("Avg Listed Price", money(filtered["ListedPrice"].mean()), "Observed listing price", "#9c9df4"),
        make_stat_card("AI Price Estimate", money(filtered["PredictedPrice"].mean()), "Model estimate", "#7adab3"),
        make_stat_card("Flagged Overpriced", f"{overpriced_count}", f"{fair_count} fairly priced homes", "#f5b06f"),
    ]

    center_lat = filtered["Latitude"].mean()
    center_lon = filtered["Longitude"].mean()
    zoom_value = 11 if zoom_value is None else zoom_value

    map_fig = px.scatter_mapbox(
        filtered,
        lat="Latitude",
        lon="Longitude",
        color="PricingLabel",
        size="ListedPrice",
        hover_name="Address",
        hover_data={
            "Neighborhood": True,
            "Bedrooms": True,
            "ListedPrice": ":,.0f",
            "PredictedPrice": ":,.0f",
            "Latitude": False,
            "Longitude": False,
        },
        custom_data=["Id"],
        zoom=zoom_value,
        height=505,
        color_discrete_map={
            "Fairly priced": "#22c55e",
            "Overpriced": "#f97316",
            "Undervalued": "#f59e0b",
        },
    )
    map_fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_center={"lat": center_lat, "lon": center_lon},
        margin=dict(l=0, r=0, t=8, b=0),
        paper_bgcolor="white",
        plot_bgcolor="white",
        uirevision="map-stable",
        legend_title_text="",
        legend=dict(orientation="h", yanchor="bottom", y=0.98, xanchor="left", x=0.02, bgcolor="rgba(255,255,255,0.7)"),
        clickmode="event+select"
    )
    map_fig.update_traces(marker={"opacity": 0.88}, selector=dict(mode="markers"))
    map_fig.update_layout(transition={"duration": 500, "easing": "cubic-in-out"})

    sorted_filtered = filtered.sort_values("GapPct", ascending=False).copy()
    if selected_id is not None and selected_id in sorted_filtered["Id"].values:
        selected_row_df = sorted_filtered[sorted_filtered["Id"] == selected_id]
        other_rows = sorted_filtered[sorted_filtered["Id"] != selected_id]
        sorted_filtered = pd.concat([selected_row_df, other_rows], axis=0)
    cards = [make_listing_card(row, selected_id=selected_id) for _, row in sorted_filtered.iterrows()]

    if selected_id is not None and selected_id in data["Id"].values:
        selected_row = data[data["Id"] == selected_id].iloc[0]
    else:
        selected_row = filtered.iloc[0]

    scatter_fig = px.scatter(
        filtered,
        x="GrLivArea",
        y="ListedPrice",
        color="PricingLabel",
        size="OverallQual",
        hover_name="Address",
        custom_data=["Id"],
        title=f"Selected Property vs Market: {selected_row['Address']}",
        labels={"GrLivArea": "Living Area (sq ft)", "ListedPrice": "Listed Price"},
        color_discrete_map={
            "Fairly priced": "#22c55e",
            "Overpriced": "#f97316",
            "Undervalued": "#f59e0b",
        }
    )
    scatter_fig.add_scatter(
        x=[selected_row["GrLivArea"]],
        y=[selected_row["ListedPrice"]],
        mode="markers+text",
        text=["Selected"],
        textposition="top center",
        marker=dict(size=24, color="#111827", symbol="star"),
        name="Selected Property"
    )
    scatter_fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=10, r=10, t=50, b=10),
        title_font_size=20,
        legend_title_text="",
        clickmode="event+select"
    )
    scatter_fig.update_traces(marker={"opacity": 0.75}, selector=dict(mode="markers"))
    scatter_fig.update_layout(transition={"duration": 450, "easing": "cubic-in-out"})

    compare_df = pd.DataFrame({
        "Feature": ["Living Area", "Overall Quality", "Age", "Garage Cars", "Bathrooms"],
        "Selected Property": [
            selected_row["GrLivArea"],
            selected_row["OverallQual"],
            selected_row["Age"],
            selected_row["GarageCars"] if pd.notna(selected_row["GarageCars"]) else 0,
            selected_row["FullBath"] if pd.notna(selected_row["FullBath"]) else 0,
        ],
        "Dataset Average": [
            data["GrLivArea"].mean(),
            data["OverallQual"].mean(),
            data["Age"].mean(),
            data["GarageCars"].fillna(0).mean(),
            data["FullBath"].fillna(0).mean(),
        ]
    })
    compare_long = compare_df.melt(id_vars="Feature", value_vars=["Selected Property", "Dataset Average"], var_name="Type", value_name="Value")
    feature_fig = px.bar(
        compare_long,
        x="Feature",
        y="Value",
        color="Type",
        barmode="group",
        title=f"Property Feature Comparison: {selected_row['Address']}",
        labels={"Value": "Value"}
    )
    feature_fig.update_layout(paper_bgcolor="white", plot_bgcolor="white", margin=dict(l=10, r=10, t=50, b=10), title_font_size=20, transition={"duration": 450, "easing": "cubic-in-out"})

    price_compare_df = pd.DataFrame({
        "Metric": ["Listed Price", "AI Price Estimate", "Neighborhood Avg", "Dataset Avg"],
        "Price": [
            selected_row["ListedPrice"],
            selected_row["PredictedPrice"],
            data[data["Neighborhood"] == selected_row["Neighborhood"]]["ListedPrice"].mean(),
            data["ListedPrice"].mean()
        ]
    })
    trend_fig = px.bar(price_compare_df, x="Metric", y="Price", title=f"Selected Property Price Context: {selected_row['Address']}", text="Price")
    trend_fig.update_traces(texttemplate="$%{y:,.0f}", textposition="outside")
    trend_fig.update_layout(paper_bgcolor="white", plot_bgcolor="white", margin=dict(l=10, r=10, t=50, b=10), title_font_size=20, showlegend=False, transition={"duration": 450, "easing": "cubic-in-out"})

    return stats, map_fig, cards, scatter_fig, feature_fig, trend_fig


# -----------------------------
# Property insight panel
# -----------------------------
@app.callback(
    Output("property-insight-panel", "children"),
    Input("selected-property-store", "data"),
    Input("neighborhood-filter", "value"),
    Input("bedroom-filter", "value"),
    Input("property-type-store", "data"),
)
def render_property_insights(selected_id, neighborhood, bedrooms, property_type):
    filtered = get_filtered_data(neighborhood, bedrooms, property_type)

    if selected_id is None:
        top = filtered.iloc[0]
        return html.Div(
            [
                html.Div("Property Insights", style={"fontSize": "22px", "fontWeight": "900", "color": "#0f172a"}),
                html.Div("Choose a home to see detailed pricing insight and comparison.", style={"color": "#64748b", "fontSize": "15px", "marginTop": "6px"}),
                html.Div(
                    [
                        make_mini_metric("Preview Home", top["Address"], f"{top['Neighborhood']} • {display_property_type(top['PropertyType'])}"),
                        make_mini_metric("Listed Price", money(top["ListedPrice"])),
                        make_mini_metric("AI Price Estimate", money(top["PredictedPrice"])),
                        make_mini_metric("Pricing Label", top["PricingLabel"]),
                    ],
                    style={"display": "grid", "gridTemplateColumns": "repeat(4, minmax(150px, 1fr))", "gap": "14px", "marginTop": "18px"}
                )
            ]
        )

    row_df = data[data["Id"] == selected_id]
    if row_df.empty:
        return html.Div("Selected property not found.")

    row = row_df.iloc[0]
    neighborhood_df = data[data["Neighborhood"] == row["Neighborhood"]]
    avg_price = data["ListedPrice"].mean()
    avg_sqft = data["GrLivArea"].mean()
    avg_quality = data["OverallQual"].mean()
    avg_age = data["Age"].mean()
    neigh_avg_price = neighborhood_df["ListedPrice"].mean()
    gap_pct = row["GapPct"] * 100
    reasons = build_explanation(row)

    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(row["Address"], style={"margin": "0", "color": "#0f172a", "fontSize": "28px", "fontWeight": "900"}),
                            html.Div(f"{row['Neighborhood']} • {int(row['Bedrooms'])} bed • {display_property_type(row['PropertyType'])}", style={"color": "#64748b", "marginTop": "4px", "fontSize": "15px"})
                        ]
                    ),
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
                style={"display": "flex", "justifyContent": "space-between", "alignItems": "flex-start", "marginBottom": "18px"}
            ),

            html.Div(
                [
                    make_mini_metric("Listed Price", money(row["ListedPrice"])),
                    make_mini_metric("AI Price Estimate", money(row["PredictedPrice"])),
                    make_mini_metric("Price Gap", f"{gap_pct:.1f}%", "Listed vs predicted"),
                    make_mini_metric("Neighborhood Avg", money(neigh_avg_price), f"Dataset avg: {money(avg_price)}"),
                ],
                style={"display": "grid", "gridTemplateColumns": "repeat(4, minmax(160px, 1fr))", "gap": "16px", "marginBottom": "24px"}
            ),

            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Why this home is labeled this way", style={"fontSize": "20px", "fontWeight": "900", "color": "#0f172a", "marginBottom": "12px", "textAlign": "center"}),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Div(f"0{i+1}", style={"fontSize": "14px", "fontWeight": "900", "color": "#1e40af", "marginBottom": "6px", "display": "inline-block", "marginRight": "10px"}),
                                            html.Div(reason, style={"color": "#334155", "lineHeight": "1.6", "fontSize": "15px", "display": "inline"})
                                        ],
                                        style={
                                            "marginBottom": "16px",
                                            "textAlign": "left"
                                        }
                                    ) for i, reason in enumerate(reasons)
                                ],
                                style={"maxWidth": "860px", "margin": "0 auto"}
                            )
                        ],
                        style={"marginBottom": "26px"}
                    ),
                    html.Div(
                        [
                            html.Div("Property Breakdown", style={"fontSize": "18px", "fontWeight": "800", "color": "#0f172a", "marginBottom": "12px", "textAlign": "center"}),
                            html.Div(
                                [
                                    make_mini_metric("Living Area", f"{row['GrLivArea']:.0f} sq ft", f"Dataset avg: {avg_sqft:.0f}"),
                                    make_mini_metric("Overall Quality", f"{row['OverallQual']}", f"Dataset avg: {avg_quality:.1f}"),
                                    make_mini_metric("Age", f"{row['Age']:.0f} years", f"Dataset avg: {avg_age:.1f}"),
                                ],
                                style={"display": "grid", "gridTemplateColumns": "repeat(3, minmax(150px, 1fr))", "gap": "14px", "maxWidth": "900px", "margin": "0 auto"}
                            )
                        ]
                    )
                ],
                style={"display": "block"}
            )
        ]
    )


# Real-time AI Insights Callback - Auto-generates insights when properties are selected
@app.callback(
    Output("chat-history-store", "data", allow_duplicate=True),
    Output("chat-messages", "children", allow_duplicate=True),
    Input("selected-property-store", "data"),
    State("chat-history-store", "data"),
    State("neighborhood-filter", "value"),
    State("bedroom-filter", "value"),
    State("property-type-store", "data"),
    prevent_initial_call=True
)
def auto_generate_insights(selected_id, history, neighborhood, bedrooms, property_type):
    """Automatically generate AI insights when property is selected"""
    if selected_id is None or selected_id not in data["Id"].values:
        return dash.no_update, dash.no_update
    
    history = history or []
    insights = generate_ai_insights(selected_id, neighborhood, bedrooms, property_type)
    if not insights:
        return dash.no_update, dash.no_update
    
    insight_text = " ".join(insights)
    history = history + [{"role": "assistant", "text": f"🤖 Real-time: {insight_text}"}]
    rendered = [make_chat_message(msg["role"], msg["text"]) for msg in history]
    return history, rendered


if __name__ == "__main__":
    app.run(debug=True)
