import os
import ast
import numpy as np
import pandas as pd
import dash
from dash import dcc, html, Input, Output
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

# Fallback display prediction if submission.csv does not align with train.csv
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


data["PricingLabel"] = data["GapPct"].apply(pricing_label)
data["PropertyType"] = data["BldgType"].fillna("House")
data["Bedrooms"] = data["BedroomAbvGr"].fillna(0).astype(int)

# User-friendly property type labels
PROPERTY_LABELS = {
    "ALL": "All Homes",
    "1Fam": "Single Family",
    "2fmCon": "Two-Family",
    "Duplex": "Duplex",
    "Twnhs": "Townhouse",
    "TwnhsE": "End-Unit Townhome"
}

# Neighborhood coordinates
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

# Demo address labels
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


def make_stat_card(title, value, subtitle="", accent="#3b82f6"):
    return html.Div(
        [
            html.Div(
                style={
                    "width": "42px",
                    "height": "42px",
                    "borderRadius": "14px",
                    "background": f"linear-gradient(135deg, {accent}, #dbeafe)",
                    "marginBottom": "14px",
                    "opacity": "0.9"
                }
            ),
            html.Div(title, style={"fontSize": "13px", "color": "#64748b", "marginBottom": "6px", "fontWeight": "600"}),
            html.Div(value, style={"fontSize": "30px", "fontWeight": "800", "color": "#0f172a"}),
            html.Div(subtitle, style={"fontSize": "12px", "color": "#94a3b8", "marginTop": "4px"}),
        ],
        style={
            "background": "linear-gradient(180deg, #ffffff 0%, #f8fbff 100%)",
            "padding": "20px",
            "borderRadius": "20px",
            "boxShadow": "0 10px 30px rgba(15, 23, 42, 0.08)",
            "border": "1px solid rgba(226,232,240,0.8)",
            "flex": "1",
            "minWidth": "190px"
        },
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
                    "width": "108px",
                    "height": "108px",
                    "borderRadius": "18px",
                    "background": "linear-gradient(135deg, #dbeafe 0%, #eff6ff 45%, #f8fafc 100%)",
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "center",
                    "fontSize": "42px",
                    "flexShrink": "0",
                    "boxShadow": "inset 0 1px 0 rgba(255,255,255,0.7)"
                },
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                row["Address"],
                                style={
                                    "fontSize": "24px",
                                    "fontWeight": "800",
                                    "color": "#0f172a",
                                    "lineHeight": "1.1"
                                }
                            ),
                            html.Span(
                                badge,
                                style={
                                    "background": badge_color,
                                    "color": "white",
                                    "padding": "6px 12px",
                                    "borderRadius": "999px",
                                    "fontSize": "12px",
                                    "fontWeight": "700"
                                },
                            ),
                        ],
                        style={
                            "display": "flex",
                            "justifyContent": "space-between",
                            "alignItems": "flex-start",
                            "gap": "12px"
                        },
                    ),
                    html.Div(
                        f"{row['Neighborhood']} • {int(row['Bedrooms'])} bed • {display_property_type(row['PropertyType'])}",
                        style={
                            "fontSize": "14px",
                            "color": "#64748b",
                            "marginTop": "8px",
                            "fontWeight": "500"
                        }
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div("Listed Price", style={"fontSize": "12px", "color": "#94a3b8"}),
                                    html.Div(money(row["ListedPrice"]), style={"fontSize": "22px", "fontWeight": "800", "color": "#111827"})
                                ]
                            ),
                            html.Div(
                                [
                                    html.Div("Predicted Price", style={"fontSize": "12px", "color": "#94a3b8"}),
                                    html.Div(money(row["PredictedPrice"]), style={"fontSize": "22px", "fontWeight": "800", "color": "#111827"})
                                ]
                            ),
                        ],
                        style={"display": "flex", "gap": "28px", "marginTop": "18px", "flexWrap": "wrap"},
                    ),
                ],
                style={"flex": "1", "textAlign": "left"},
            ),
        ],
        id={"type": "listing-card", "index": int(row["Id"])},
        n_clicks=0,
        style={
            "display": "flex",
            "gap": "16px",
            "padding": "18px",
            "width": "100%",
            "background": "linear-gradient(180deg, #ffffff 0%, #fbfdff 100%)",
            "borderRadius": "22px",
            "boxShadow": "0 12px 32px rgba(15, 23, 42, 0.08)",
            "border": "2px solid #60a5fa" if is_selected else "1px solid rgba(226,232,240,0.85)",
            "marginBottom": "16px",
            "cursor": "pointer",
            "textAlign": "left",
            "appearance": "none",
            "WebkitAppearance": "none",
            "outline": "none"
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
            "padding": "12px 18px",
            "borderRadius": "14px",
            "border": "2px solid #3b82f6" if selected else "1px solid #dbeafe",
            "background": "#3b82f6" if selected else "#ffffff",
            "color": "white" if selected else "#1e293b",
            "fontWeight": "700",
            "fontSize": "14px",
            "cursor": "pointer",
            "boxShadow": "0 4px 12px rgba(59,130,246,0.18)" if selected else "none",
            "transition": "all 0.2s ease",
            "minWidth": "140px",
            "textAlign": "center",
            "width": "100%"
        }
    )


# -----------------------------
# App setup
# -----------------------------
app = dash.Dash(__name__)
app.title = "Real Estate Analytics"

app.layout = html.Div(
    [
        dcc.Store(id="property-type-store", data="ALL"),
        dcc.Store(id="selected-property-store"),

        html.Div(
            [
                html.Div("Filters", style={"fontSize": "28px", "fontWeight": "800", "color": "#0f172a", "marginBottom": "22px"}),

                html.Div("Neighborhood", style={"fontSize": "14px", "fontWeight": "700", "marginBottom": "8px", "color": "#475569"}),
                dcc.Dropdown(
                    id="neighborhood-filter",
                    options=[{"label": "All Neighborhoods", "value": "ALL"}] +
                            [{"label": n, "value": n} for n in sorted(data["Neighborhood"].dropna().unique())],
                    value="ALL",
                    clearable=False,
                    style={"marginBottom": "20px"}
                ),

                html.Div("Bedrooms", style={"fontSize": "14px", "fontWeight": "700", "marginBottom": "8px", "color": "#475569"}),
                dcc.Dropdown(
                    id="bedroom-filter",
                    options=[{"label": "Any", "value": -1}] +
                            [{"label": f"{b} Bedrooms", "value": int(b)} for b in sorted(data["Bedrooms"].dropna().unique())],
                    value=-1,
                    clearable=False,
                    style={"marginBottom": "20px"}
                ),

                html.Div("Property Type", style={
                    "fontSize": "14px",
                    "fontWeight": "700",
                    "marginBottom": "6px",
                    "color": "#475569"
                }),
                html.Div(
                    "Choose the kind of home you want to view.",
                    style={
                        "fontSize": "12px",
                        "color": "#64748b",
                        "marginBottom": "10px"
                    }
                ),
                html.Div(
                    id="property-type-buttons",
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "repeat(2, minmax(140px, 1fr))",
                        "gap": "10px",
                        "marginBottom": "8px"
                    }
                ),

                html.Div(
                    [
                        html.Div("Project demo", style={"fontSize": "12px", "color": "#94a3b8", "fontWeight": "700", "textTransform": "uppercase", "letterSpacing": "0.08em"}),
                        html.Div("AI-assisted pricing support", style={"fontSize": "20px", "fontWeight": "800", "color": "#0f172a", "marginTop": "8px"}),
                        html.Div(
                            "Click a property on the map, scatter plot, or listing card to see detailed pricing insights.",
                            style={"fontSize": "13px", "color": "#64748b", "lineHeight": "1.6", "marginTop": "8px"}
                        ),
                    ],
                    style={
                        "marginTop": "28px",
                        "padding": "18px",
                        "borderRadius": "20px",
                        "background": "linear-gradient(135deg, #eff6ff 0%, #f8fafc 100%)",
                        "border": "1px solid #dbeafe"
                    }
                ),
            ],
            style={
                "width": "320px",
                "background": "linear-gradient(180deg, #f8fbff 0%, #f8fafc 100%)",
                "padding": "28px 22px",
                "borderRight": "1px solid #e2e8f0",
                "minHeight": "100vh",
                "position": "sticky",
                "top": "0"
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
                                        html.Span("Real Estate Analytics", style={"fontSize": "52px", "fontWeight": "900", "color": "#0f172a"}),
                                        html.Span(" ✦", style={"fontSize": "34px", "color": "#60a5fa", "fontWeight": "800"})
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
                                    "Connected • Demo mode",
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

                html.Div(id="stats-row", style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "marginBottom": "22px"}),

                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div("Map Zoom", style={"fontWeight": "700", "marginBottom": "6px", "color": "#334155"}),
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
                                        "padding": "10px",
                                        "background": "#f8fafc",
                                        "borderRadius": "12px"
                                    }
                                ),
                                dcc.Graph(
                                    id="map-graph",
                                    config={
                                        "displayModeBar": False,
                                        "scrollZoom": True
                                    },
                                    style={"height": "530px"}
                                )
                            ],
                            style={
                                "flex": "1.45",
                                "background": "linear-gradient(180deg, #ffffff 0%, #fbfdff 100%)",
                                "padding": "12px",
                                "borderRadius": "26px",
                                "boxShadow": "0 12px 36px rgba(15, 23, 42, 0.08)",
                                "border": "1px solid rgba(226,232,240,0.85)"
                            }
                        ),
                        html.Div(
                            id="listing-cards",
                            style={
                                "flex": "1",
                                "maxHeight": "620px",
                                "overflowY": "scroll",
                                "paddingRight": "8px"
                            }
                        ),
                    ],
                    style={"display": "flex", "gap": "20px", "alignItems": "stretch", "marginBottom": "22px"}
                ),

                html.Div(
                    id="property-insight-panel",
                    style={
                        "background": "linear-gradient(180deg, #ffffff 0%, #fbfdff 100%)",
                        "padding": "22px",
                        "borderRadius": "24px",
                        "boxShadow": "0 12px 32px rgba(15, 23, 42, 0.08)",
                        "border": "1px solid rgba(226,232,240,0.85)",
                        "marginBottom": "22px"
                    }
                ),

                html.Div(
                    [
                        html.Div(
                            dcc.Graph(id="scatter-graph", config={"displayModeBar": False}),
                            style={
                                "background": "linear-gradient(180deg, #ffffff 0%, #fbfdff 100%)",
                                "padding": "12px",
                                "borderRadius": "24px",
                                "boxShadow": "0 12px 32px rgba(15, 23, 42, 0.08)",
                                "border": "1px solid rgba(226,232,240,0.85)"
                            }
                        ),
                        html.Div(
                            dcc.Graph(id="feature-impact-graph", config={"displayModeBar": False}),
                            style={
                                "background": "linear-gradient(180deg, #ffffff 0%, #fbfdff 100%)",
                                "padding": "12px",
                                "borderRadius": "24px",
                                "boxShadow": "0 12px 32px rgba(15, 23, 42, 0.08)",
                                "border": "1px solid rgba(226,232,240,0.85)"
                            }
                        ),
                        html.Div(
                            dcc.Graph(id="trend-graph", config={"displayModeBar": False}),
                            style={
                                "background": "linear-gradient(180deg, #ffffff 0%, #fbfdff 100%)",
                                "padding": "12px",
                                "borderRadius": "24px",
                                "boxShadow": "0 12px 32px rgba(15, 23, 42, 0.08)",
                                "border": "1px solid rgba(226,232,240,0.85)"
                            }
                        ),
                    ],
                    style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr", "gap": "20px"}
                ),
            ],
            style={"flex": "1", "padding": "28px", "background": "#f1f5f9"}
        )
    ],
    style={"display": "flex", "fontFamily": "Inter, Arial, sans-serif", "background": "#f1f5f9"}
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
    filtered = data.copy()

    if neighborhood != "ALL":
        filtered = filtered[filtered["Neighborhood"] == neighborhood]
    if bedrooms != -1:
        filtered = filtered[filtered["Bedrooms"] == bedrooms]
    if property_type != "ALL":
        filtered = filtered[filtered["PropertyType"] == property_type]

    if filtered.empty:
        filtered = data.head(10).copy()

    overpriced_count = int((filtered["PricingLabel"] == "Overpriced").sum())
    fair_count = int((filtered["PricingLabel"] == "Fairly priced").sum())

    stats = [
        make_stat_card("Listings", f"{len(filtered):,}", "Current filtered view", "#60a5fa"),
        make_stat_card("Avg Listed Price", money(filtered["ListedPrice"].mean()), "Observed listing price", "#818cf8"),
        make_stat_card("Avg Predicted Price", money(filtered["PredictedPrice"].mean()), "Model estimate", "#34d399"),
        make_stat_card("Flagged Overpriced", f"{overpriced_count}", f"{fair_count} fairly priced homes", "#fb923c"),
    ]

    # Map
    center_lat = filtered["Latitude"].mean()
    center_lon = filtered["Longitude"].mean()

    if zoom_value is None:
        zoom_value = 11

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
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.98,
            xanchor="left",
            x=0.02,
            bgcolor="rgba(255,255,255,0.7)"
        ),
        clickmode="event+select"
    )
    map_fig.update_traces(marker={"opacity": 0.88}, selector=dict(mode="markers"))

    # Cards
    sorted_filtered = filtered.sort_values("GapPct", ascending=False).copy()

    if selected_id is not None and selected_id in sorted_filtered["Id"].values:
        selected_row_df = sorted_filtered[sorted_filtered["Id"] == selected_id]
        other_rows = sorted_filtered[sorted_filtered["Id"] != selected_id]
        sorted_filtered = pd.concat([selected_row_df, other_rows], axis=0)

    card_rows = sorted_filtered.to_dict("records")
    cards = [make_listing_card(row, selected_id=selected_id) for row in card_rows]

    # Selected property fallback
    if selected_id is not None and selected_id in data["Id"].values:
        selected_row = data[data["Id"] == selected_id].iloc[0]
    else:
        selected_row = filtered.iloc[0]

    # -----------------------------
    # Chart 1: Scatter with selected property highlighted
    # -----------------------------
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

    # -----------------------------
    # Chart 2: Selected property vs dataset average
    # -----------------------------
    compare_df = pd.DataFrame({
        "Feature": ["Living Area", "Overall Quality", "Age", "Garage Cars", "Bathrooms"],
        "Selected Property": [
            selected_row["GrLivArea"],
            selected_row["OverallQual"],
            selected_row["YrSold"] - selected_row["YearBuilt"],
            selected_row["GarageCars"] if pd.notna(selected_row["GarageCars"]) else 0,
            selected_row["FullBath"] if pd.notna(selected_row["FullBath"]) else 0,
        ],
        "Dataset Average": [
            data["GrLivArea"].mean(),
            data["OverallQual"].mean(),
            (data["YrSold"] - data["YearBuilt"]).mean(),
            data["GarageCars"].fillna(0).mean(),
            data["FullBath"].fillna(0).mean(),
        ]
    })

    compare_long = compare_df.melt(
        id_vars="Feature",
        value_vars=["Selected Property", "Dataset Average"],
        var_name="Type",
        value_name="Value"
    )

    feature_fig = px.bar(
        compare_long,
        x="Feature",
        y="Value",
        color="Type",
        barmode="group",
        title=f"Property Feature Comparison: {selected_row['Address']}",
        labels={"Value": "Value"}
    )
    feature_fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=10, r=10, t=50, b=10),
        title_font_size=20
    )

    # -----------------------------
    # Chart 3: Selected property price context
    # -----------------------------
    neighborhood_df = data[data["Neighborhood"] == selected_row["Neighborhood"]]

    price_compare_df = pd.DataFrame({
        "Metric": [
            "Listed Price",
            "Predicted Price",
            "Neighborhood Avg",
            "Dataset Avg"
        ],
        "Price": [
            selected_row["ListedPrice"],
            selected_row["PredictedPrice"],
            neighborhood_df["ListedPrice"].mean(),
            data["ListedPrice"].mean()
        ]
    })

    trend_fig = px.bar(
        price_compare_df,
        x="Metric",
        y="Price",
        title=f"Selected Property Price Context: {selected_row['Address']}",
        text="Price"
    )
    trend_fig.update_traces(texttemplate="$%{y:,.0f}", textposition="outside")
    trend_fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=10, r=10, t=50, b=10),
        title_font_size=20,
        showlegend=False
    )

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
    filtered = data.copy()

    if neighborhood != "ALL":
        filtered = filtered[filtered["Neighborhood"] == neighborhood]
    if bedrooms != -1:
        filtered = filtered[filtered["Bedrooms"] == bedrooms]
    if property_type != "ALL":
        filtered = filtered[filtered["PropertyType"] == property_type]

    if filtered.empty:
        filtered = data.copy()

    if selected_id is None:
        return html.Div(
            [
                html.H3("Property Insights", style={"marginBottom": "8px", "color": "#0f172a"}),
                html.P(
                    "Click a property on the map, scatter plot, or the listing cards on the right to see specific insights.",
                    style={"color": "#64748b", "fontSize": "15px"}
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
    avg_age = (data["YrSold"] - data["YearBuilt"]).mean()

    neigh_avg_price = neighborhood_df["ListedPrice"].mean()
    neigh_avg_sqft = neighborhood_df["GrLivArea"].mean()

    prop_age = row["YrSold"] - row["YearBuilt"]
    gap_pct = row["GapPct"] * 100

    reasons = []

    if row["GrLivArea"] > neigh_avg_sqft:
        reasons.append("This home has above-average living area for its neighborhood.")
    else:
        reasons.append("This home is smaller than the neighborhood average, which may reduce value.")

    if row["OverallQual"] > avg_quality:
        reasons.append("Its overall quality score is above the dataset average.")
    elif row["OverallQual"] < avg_quality:
        reasons.append("Its overall quality score is below the dataset average.")
    else:
        reasons.append("Its overall quality score is close to the dataset average.")

    if prop_age > avg_age:
        reasons.append("The property is older than the average home in the dataset.")
    else:
        reasons.append("The property is newer than the average home in the dataset.")

    if gap_pct > 8:
        reasons.append("The listed price is noticeably above the model-estimated fair price.")
    elif gap_pct < -8:
        reasons.append("The listed price is below the model-estimated fair price.")
    else:
        reasons.append("The listed price is close to the model-estimated fair price.")

    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H2(row["Address"], style={"margin": "0", "color": "#0f172a"}),
                            html.Div(
                                f"{row['Neighborhood']} • {int(row['Bedrooms'])} bed • {display_property_type(row['PropertyType'])}",
                                style={"color": "#64748b", "marginTop": "4px"}
                            )
                        ]
                    ),
                    html.Div(
                        row["PricingLabel"],
                        style={
                            "background": label_color(row["PricingLabel"]),
                            "color": "white",
                            "padding": "8px 14px",
                            "borderRadius": "999px",
                            "fontWeight": "700",
                            "height": "fit-content"
                        }
                    )
                ],
                style={"display": "flex", "justifyContent": "space-between", "alignItems": "flex-start", "marginBottom": "18px"}
            ),

            html.Div(
                [
                    html.Div([
                        html.Div("Listed Price", style={"fontSize": "12px", "color": "#94a3b8"}),
                        html.Div(money(row["ListedPrice"]), style={"fontSize": "26px", "fontWeight": "800"})
                    ]),
                    html.Div([
                        html.Div("Predicted Price", style={"fontSize": "12px", "color": "#94a3b8"}),
                        html.Div(money(row["PredictedPrice"]), style={"fontSize": "26px", "fontWeight": "800"})
                    ]),
                    html.Div([
                        html.Div("Price Gap", style={"fontSize": "12px", "color": "#94a3b8"}),
                        html.Div(f"{gap_pct:.1f}%", style={"fontSize": "26px", "fontWeight": "800"})
                    ]),
                ],
                style={"display": "flex", "gap": "36px", "flexWrap": "wrap", "marginBottom": "24px"}
            ),

            html.Div(
                [
                    html.Div([
                        html.Div("Living Area", style={"fontSize": "12px", "color": "#94a3b8"}),
                        html.Div(f"{row['GrLivArea']:.0f} sq ft", style={"fontWeight": "700", "fontSize": "18px"}),
                        html.Div(f"Dataset avg: {avg_sqft:.0f}", style={"fontSize": "13px", "color": "#64748b"})
                    ], style={"padding": "14px", "background": "#f8fafc", "borderRadius": "16px"}),

                    html.Div([
                        html.Div("Overall Quality", style={"fontSize": "12px", "color": "#94a3b8"}),
                        html.Div(f"{row['OverallQual']}", style={"fontWeight": "700", "fontSize": "18px"}),
                        html.Div(f"Dataset avg: {avg_quality:.1f}", style={"fontSize": "13px", "color": "#64748b"})
                    ], style={"padding": "14px", "background": "#f8fafc", "borderRadius": "16px"}),

                    html.Div([
                        html.Div("Age", style={"fontSize": "12px", "color": "#94a3b8"}),
                        html.Div(f"{prop_age:.0f} years", style={"fontWeight": "700", "fontSize": "18px"}),
                        html.Div(f"Dataset avg: {avg_age:.1f}", style={"fontSize": "13px", "color": "#64748b"})
                    ], style={"padding": "14px", "background": "#f8fafc", "borderRadius": "16px"}),

                    html.Div([
                        html.Div("Neighborhood Avg Price", style={"fontSize": "12px", "color": "#94a3b8"}),
                        html.Div(money(neigh_avg_price), style={"fontWeight": "700", "fontSize": "18px"}),
                        html.Div(f"Dataset avg: {money(avg_price)}", style={"fontSize": "13px", "color": "#64748b"})
                    ], style={"padding": "14px", "background": "#f8fafc", "borderRadius": "16px"}),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(4, minmax(160px, 1fr))",
                    "gap": "16px",
                    "marginBottom": "24px"
                }
            ),

            html.Div(
                [
                    html.H4("Why this property is labeled this way", style={"marginBottom": "10px", "color": "#0f172a"}),
                    html.Ul(
                        [html.Li(reason, style={"marginBottom": "8px", "color": "#334155"}) for reason in reasons],
                        style={"paddingLeft": "20px"}
                    )
                ]
            )
        ]
    )


if __name__ == "__main__":
    app.run(debug=True)
