"""
Transport for London — Santander Cycles: Demand Exploration & Forecasting Dashboard
Built with Dash & Plotly. Deployed on Render with uv & Gunicorn.
Themed with authentic Santander Cycles branding in pure pitch-black, horizontal prediction charts,
and streamlined controls aligned with the assignment prompt.
"""

import os
import io
import requests
import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Input, Output
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.graph_objects as go

# Import Open-Meteo weather helper
try:
    from open_meteo import open_meteo, open_meteo_history
except ImportError:
    from session10_workshop5.dashboard.open_meteo import open_meteo, open_meteo_history

# -----------------------------------------------------------------------------
# App Configuration & Server Export for Gunicorn
# -----------------------------------------------------------------------------
app = Dash(
    __name__,
    title="Santander Cycles — TfL Demand Dashboard",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    suppress_callback_exceptions=True,
)
server = app.server  # Expose server for gunicorn (render.yaml: uv run gunicorn app:server)

# -----------------------------------------------------------------------------
# Color & Branding Constants (Pure Pitch-Black & Santander Red)
# -----------------------------------------------------------------------------
BRAND = {
    "bg_page": "#000000",
    "bg_tile": "#111111",
    "bg_tile_hover": "#171717",
    "bg_input": "#0A0A0A",
    "border": "#242424",
    "border_highlight": "#333333",
    "santander_red": "#EC0000",
    "santander_flame": "#FF2E2E",
    "weekday_color": "#A1A1AA",
    "text_primary": "#FFFFFF",
    "text_secondary": "#A1A1AA",
    "text_muted": "#71717A",
    "grid": "#1E1E1E",
    "mint": "#10B981",
    "amber": "#F59E0B",
    "ice": "#38BDF8",
}

WEATHER_VARS = {
    "temp": {"label": "Temperature (°C)", "unit": "°C", "icon": "🌡️"},
    "humidity": {"label": "Relative Humidity (%)", "unit": "%", "icon": "💧"},
    "precip": {"label": "Precipitation (mm)", "unit": "mm", "icon": "🌧️"},
    "windspeed": {"label": "Wind Speed (km/h)", "unit": "km/h", "icon": "💨"},
    "cloudcover": {"label": "Cloud Cover (%)", "unit": "%", "icon": "☁️"},
}

WEEKDAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
SEASONS_ORDER = ["Spring", "Summer", "Autumn", "Winter"]

def get_plotly_layout():
    """Consistent pure pitch-black layout for all Plotly graph tiles."""
    return dict(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Outfit, sans-serif", color=BRAND["text_secondary"]),
        margin=dict(l=45, r=25, t=45, b=45),
        transition=dict(duration=300, easing="cubic-in-out"),
        xaxis=dict(
            gridcolor=BRAND["grid"],
            linecolor=BRAND["border"],
            tickfont=dict(color=BRAND["text_secondary"], size=11),
            title_font=dict(size=13, color=BRAND["text_primary"]),
        ),
        yaxis=dict(
            gridcolor=BRAND["grid"],
            linecolor=BRAND["border"],
            tickfont=dict(color=BRAND["text_secondary"], size=11),
            title_font=dict(size=13, color=BRAND["text_primary"]),
        ),
        legend=dict(
            bgcolor="rgba(17, 17, 17, 0.95)",
            bordercolor=BRAND["border"],
            borderwidth=1,
            font=dict(color=BRAND["text_primary"], size=12),
        ),
    )

# -----------------------------------------------------------------------------
# Data Loading & In-Memory Caching (with Graceful Fallback)
# -----------------------------------------------------------------------------
DATA_URL = "https://raw.githubusercontent.com/kostis-christodoulou/am01-code-sep2026/main/data/london_bikes.csv"

def load_bikes_data():
    """Load and clean the London Bikes dataset with robust fallback."""
    df = None
    try:
        resp = requests.get(DATA_URL, timeout=10)
        if resp.status_code == 200:
            df = pd.read_csv(io.StringIO(resp.text))
    except Exception as e:
        print(f"Notice: Remote data load issue ({e}). Checking local files...")

    if df is None:
        for local_path in ["data/london_bikes.csv", "../data/london_bikes.csv", "../../data/london_bikes.csv"]:
            if os.path.exists(local_path):
                df = pd.read_csv(local_path)
                break

    if df is None:
        raise RuntimeError("Could not load London bikes dataset from remote URL or local path.")

    df["date"] = pd.to_datetime(df["date"])
    df["day_of_week"] = pd.Categorical(df["day_of_week"], categories=WEEKDAY_ORDER, ordered=True)
    if "weekend" not in df.columns:
        df["weekend"] = df["day_of_week"].isin(["Sat", "Sun"])
    if "season_name" in df.columns:
        df["season_name"] = pd.Categorical(df["season_name"], categories=SEASONS_ORDER, ordered=True)

    return df

BIKES_DF = load_bikes_data()

def filter_df_by_date_range(df, start_date, end_date):
    """Filter historical rides dataframe by custom date range (tz-safe)."""
    if df is None or df.empty:
        return df
    dates = pd.to_datetime(df["date"])
    mask = pd.Series(True, index=df.index)
    if start_date:
        start_ts = pd.to_datetime(start_date)
        if dates.dt.tz is not None and start_ts.tz is None:
            start_ts = start_ts.tz_localize(dates.dt.tz)
        mask &= (dates >= start_ts)
    if end_date:
        end_ts = pd.to_datetime(end_date)
        if dates.dt.tz is not None and end_ts.tz is None:
            end_ts = end_ts.tz_localize(dates.dt.tz)
        end_ts = end_ts + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        mask &= (dates <= end_ts)
    filtered = df[mask]
    return filtered if not filtered.empty else df

# -----------------------------------------------------------------------------
# Linear Model Loading & Scoring Functions
# -----------------------------------------------------------------------------
def load_model_coefficients(filepath="model_coefficients.csv"):
    """Load linear model coefficients from CSV file."""
    paths_to_try = [
        filepath,
        os.path.join(os.path.dirname(__file__), filepath),
        os.path.join(os.path.dirname(__file__), "model_coefficients.csv"),
    ]
    for p in paths_to_try:
        if os.path.exists(p):
            return pd.read_csv(p)

    # Deployable variant fallback (adjusted R2 0.710) from MAM Team 1 notebook, Part 5.
    # Uses only predictors available from Open-Meteo plus calendar-derived terms.
    return pd.DataFrame([
        ("Intercept", 42049.03),
        ("temp_c", 614.78),
        ("temp_c2", -11.21),
        ("humidity", -51.29),
        ("log_precip", -2851.67),
        ("windspeed", -121.93),
        ("cloudcover", -73.87),
        ("sin_doy", -1011.60),
        ("cos_doy", -2834.27),
        ("temp_wknd", 222.45),
        ("rain_wknd", -734.85),
        ("xmas", -8073.41),
        ("post2023", -9874.11),
        ("p_temp_c", -496.66),
        ("p_log_precip", 1298.01),
        ("p_cloudcover", 55.79),
        ("day_Mon", 0.0),
        ("day_Tue", 2246.30),
        ("day_Wed", 2487.45),
        ("day_Thu", 2394.50),
        ("day_Fri", 763.28),
        ("day_Sat", -1017.08),
        ("day_Sun", -3279.14),
    ], columns=["term", "coefficient"])

COEFF_DESCRIPTIONS = {
    "Intercept": "Baseline expected daily hires (Monday pre-2023 reference)",
    "temp_c": "Mean daily temperature (°C) — warmer days boost hires",
    "humidity": "Relative humidity (%) — damp/muggy conditions reduce rides",
    "log_precip": "Log precipitation log(1+mm) — rain sharply reduces demand",
    "windspeed": "Mean wind speed (km/h) — strong winds discourage cycling",
    "temp_c2": "Squared temperature — the warmth effect flattens once London is hot",
    "cloudcover": "Cloud cover (%) — overcast days suppress discretionary rides",
    "sin_doy": "Annual seasonality cycle, sin(day of year)",
    "cos_doy": "Annual seasonality cycle, cos(day of year)",
    "temp_wknd": "Weekend temperature boost (leisure riders are sun-sensitive)",
    "rain_wknd": "Weekend rain penalty (discretionary weekend trips cancel when wet)",
    "covid": "Lockdown indicator — mobility reduction during pandemic restrictions",
    "xmas": "Christmas holiday shutdown — sharp drop during festive bank holidays",
    "post2023": "Structural shift indicator for dates ≥ 2023 (hybrid work / tariff change)",
    "p_temp_c": "Post-2023 temperature interaction (temp_c × post2023)",
    "p_log_precip": "Post-2023 rain interaction (log_precip × post2023)",
    "p_cloudcover": "Post-2023 cloud cover interaction (cloudcover × post2023)",
    "day_Mon": "Baseline reference day (Monday)",
    "day_Tue": "Tuesday commuter premium (relative to Monday)",
    "day_Wed": "Wednesday commuter peak (highest mid-week travel volume)",
    "day_Thu": "Thursday commuter peak (heavy in-office commuter day)",
    "day_Fri": "Friday commuter adjustment (hybrid WFH transition)",
    "day_Sat": "Saturday adjustment (fewer office commuters)",
    "day_Sun": "Sunday adjustment (lowest travel demand of the week)",
}

def prepare_weather_features(df):
    """
    Augment raw Open-Meteo weather data with derived engineering features
    required by the deployed model from MAM_Team_1_Final_Group_Project (Part 5).
    """
    df = df.copy()
    dates = pd.to_datetime(df["date"])
    doy = dates.dt.dayofyear

    if "sin_doy" not in df.columns:
        df["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    if "cos_doy" not in df.columns:
        df["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)

    if "temp_c" not in df.columns and "temp" in df.columns:
        df["temp_c"] = df["temp"] - 11.7
    if "temp_c2" not in df.columns and "temp_c" in df.columns:
        df["temp_c2"] = df["temp_c"] ** 2

    if "log_precip" not in df.columns and "precip" in df.columns:
        df["log_precip"] = np.log1p(df["precip"].clip(lower=0))

    wknd = df["day_of_week"].isin(["Sat", "Sun"]).astype(int)
    if "wknd" not in df.columns:
        df["wknd"] = wknd
    if "temp_wknd" not in df.columns and "temp_c" in df.columns:
        df["temp_wknd"] = df["temp_c"] * wknd
    if "rain_wknd" not in df.columns and "log_precip" in df.columns:
        df["rain_wknd"] = df["log_precip"] * wknd

    if "covid" not in df.columns:
        df["covid"] = 0
    if "xmas" not in df.columns:
        df["xmas"] = (
            ((dates.dt.month == 12) & (dates.dt.day >= 24)) |
            ((dates.dt.month == 1) & (dates.dt.day <= 2))
        ).astype(int)
    if "post2023" not in df.columns:
        df["post2023"] = (dates >= pd.Timestamp("2023-01-01")).astype(int)

    if "p_temp_c" not in df.columns and "temp_c" in df.columns:
        df["p_temp_c"] = df["temp_c"] * df["post2023"]
    if "p_temp_c2" not in df.columns and "temp_c2" in df.columns:
        df["p_temp_c2"] = df["temp_c2"] * df["post2023"]
    if "p_log_precip" not in df.columns and "log_precip" in df.columns:
        df["p_log_precip"] = df["log_precip"] * df["post2023"]
    if "p_cloudcover" not in df.columns and "cloudcover" in df.columns:
        df["p_cloudcover"] = df["cloudcover"] * df["post2023"]

    # NOTE: visibility and solarradiation are NOT provided by Open-Meteo, so the
    # deployed model deliberately excludes them rather than substituting invented
    # values. Every predictor below is either returned by the weather feed or
    # derived from the calendar date.

    return df

def predict_bikes(weather_df, coeffs_df):
    """
    Score daily hires using linear model contract:
    Prediction = Intercept + sum(coefficient * value) [for numeric terms] + day_<weekday>
    """
    weather_df = prepare_weather_features(weather_df)
    coef_dict = dict(zip(coeffs_df["term"], coeffs_df["coefficient"]))
    intercept = coef_dict.get("Intercept", 0.0)
    numeric_terms = [t for t in coef_dict.keys() if t != "Intercept" and not t.startswith("day_")]

    predictions = []
    for _, row in weather_df.iterrows():
        pred = intercept
        for term in numeric_terms:
            if term in row and pd.notnull(row[term]):
                pred += coef_dict[term] * float(row[term])

        weekday = str(row.get("day_of_week", "")).strip()
        day_term = f"day_{weekday}"
        if day_term in coef_dict:
            pred += coef_dict[day_term]

        predictions.append(max(0, int(round(pred))))

    return predictions

# -----------------------------------------------------------------------------
# Weather Fetching with Fallback & Memory Cache
# -----------------------------------------------------------------------------
_JAN_2026_CACHE = None

def get_january_2026_weather():
    """Fetch Jan 1-7, 2026 weather from Open-Meteo archive with offline fallback."""
    global _JAN_2026_CACHE
    if _JAN_2026_CACHE is not None:
        return _JAN_2026_CACHE[0].copy(), _JAN_2026_CACHE[1]

    try:
        df = open_meteo_history("London", "2026-01-01", "2026-01-07")
        _JAN_2026_CACHE = (df, None)
        return df, None
    except Exception:
        fallback_data = {
            "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05", "2026-01-06", "2026-01-07"]),
            "day_of_week": ["Thu", "Fri", "Sat", "Sun", "Mon", "Tue", "Wed"],
            "temp": [2.7, 1.8, 0.3, -1.0, -1.3, -1.1, 3.0],
            "humidity": [83.5, 77.2, 79.4, 82.1, 85.0, 84.2, 79.0],
            "precip": [0.0, 0.4, 0.0, 0.0, 0.0, 0.6, 7.6],
            "windspeed": [14.5, 12.5, 11.2, 10.8, 12.0, 15.1, 18.2],
            "cloudcover": [73.5, 21.0, 45.0, 30.0, 60.0, 85.0, 92.0],
        }
        fallback_df = pd.DataFrame(fallback_data)
        clean_msg = "Open-Meteo archive call failed on this host — showing cached January 2026 values stored in app.py, not a live API response."
        _JAN_2026_CACHE = (fallback_df, clean_msg)
        return fallback_df, clean_msg

def get_live_forecast_weather():
    """Fetch next 5 days forecast from Open-Meteo forecast with offline fallback."""
    try:
        df = open_meteo("London", 5)
        return df, None
    except Exception:
        today = pd.Timestamp.now().normalize()
        dates = [today + pd.Timedelta(days=i) for i in range(5)]
        fallback_data = {
            "date": dates,
            "day_of_week": [d.strftime("%a") for d in dates],
            "temp": [16.0, 17.2, 18.5, 17.8, 16.5],
            "humidity": [64.0, 68.0, 72.0, 75.0, 70.0],
            "precip": [0.0, 0.2, 1.5, 3.8, 0.5],
            "windspeed": [10.5, 11.2, 14.0, 16.2, 12.0],
            "cloudcover": [50.0, 65.0, 80.0, 75.0, 55.0],
        }
        return pd.DataFrame(fallback_data), "Open-Meteo forecast call failed on this host — showing placeholder weather stored in app.py, not a live API response."

# -----------------------------------------------------------------------------
# Component Helpers
# -----------------------------------------------------------------------------
def render_styled_table(df):
    """Build a clean responsive HTML table with no horizontal scrolling."""
    headers = [
        "Date", "Day", "Temp", "Humidity",
        "Precip", "Wind", "Cloud", "Predicted"
    ]
    header_cells = [html.Th(h) for h in headers]

    rows = []
    for _, r in df.iterrows():
        date_str = pd.to_datetime(r["date"]).strftime("%d %b")
        is_wknd = r["day_of_week"] in ["Sat", "Sun"]
        day_badge = html.Span(
            r["day_of_week"],
            className=f"table-badge {'badge-weekend' if is_wknd else 'badge-weekday'}"
        )

        cells = [
            html.Td(date_str, style={"fontWeight": "600"}),
            html.Td(day_badge),
            html.Td(f"{r['temp']:.1f}°C"),
            html.Td(f"{r['humidity']:.0f}%"),
            html.Td(f"{r['precip']:.1f}mm"),
            html.Td(f"{r['windspeed']:.0f}km/h"),
            html.Td(f"{r['cloudcover']:.0f}%"),
            html.Td(
                html.Span(f"{int(r['predicted_hires']):,}", className="table-badge badge-prediction")
            ),
        ]
        rows.append(html.Tr(cells))

    return html.Div(
        style={"overflowX": "hidden", "width": "100%"},
        children=[
            html.Table(
                className="styled-table",
                children=[
                    html.Thead(html.Tr(header_cells)),
                    html.Tbody(rows),
                ]
            )
        ]
    )

# -----------------------------------------------------------------------------
# App Layout (Pure Pitch-Black & Santander Red)
# -----------------------------------------------------------------------------
app.layout = html.Div(
    id="app-container",
    style={
        "minHeight": "100vh",
        "backgroundColor": BRAND["bg_page"],
        "color": BRAND["text_primary"],
        "padding": "24px 32px",
    },
    children=[
        # Download storage
        dcc.Download(id="download-forecast-csv"),

        # Top Header Row with Santander Branding, Animated Mascot & Radial Navigation Selector
        html.Div(
            style={
                "display": "flex",
                "justifyContent": "space-between",
                "alignItems": "center",
                "flexWrap": "wrap",
                "gap": "18px",
                "marginBottom": "20px",
                "paddingBottom": "18px",
                "borderBottom": f"1px solid {BRAND['border']}",
            },
            children=[
                # Brand with Animated Bicycle Mascot
                html.Div(
                    style={"display": "flex", "alignItems": "center", "gap": "16px"},
                    children=[
                        html.Div(
                            className="bike-mascot-container",
                            children=[
                                html.Img(
                                    src="/assets/bike_mascot.svg",
                                    style={"width": "86px", "height": "54px"},
                                ),
                            ],
                            title="Santander Cycles Animated Mascot (Hover to pedal faster!)",
                        ),
                        html.Div([
                            html.Div(
                                style={"display": "flex", "alignItems": "center", "gap": "8px", "marginBottom": "4px"},
                                children=[
                                    html.Span("SANTANDER CYCLES", style={"fontSize": "11px", "fontWeight": "800", "letterSpacing": "1.5px", "color": BRAND["santander_red"], "backgroundColor": "rgba(236, 0, 0, 0.15)", "padding": "2px 8px", "borderRadius": "4px"}),
                                    html.Span("TfL Analytics", style={"fontSize": "11px", "fontWeight": "600", "color": BRAND["text_muted"]}),
                                ],
                            ),
                            html.H1(
                                "Transport for London — Santander Cycles",
                                style={
                                    "margin": "0",
                                    "fontSize": "22px",
                                    "fontWeight": "800",
                                    "color": BRAND["text_primary"],
                                    "letterSpacing": "-0.5px",
                                },
                            ),
                            html.P(
                                "Daily Demand Exploration & Meteorological Forecasting Engine",
                                style={
                                    "margin": "3px 0 0 0",
                                    "fontSize": "12.5px",
                                    "color": BRAND["text_secondary"],
                                },
                            ),
                        ]),
                    ],
                ),

                # Right Controls: Radial Button Mode Selector & Live Status Badges
                html.Div(
                    style={"display": "flex", "alignItems": "center", "gap": "12px", "flexWrap": "wrap"},
                    children=[
                        # Executive Radial Button Selector (Saves Full Row of Screen Space)
                        dcc.RadioItems(
                            id="main-tabs",
                            options=[
                                {"label": "  📊  1. Explore Data", "value": "tab-explore"},
                                {"label": "  🔮  2. Predict Demand", "value": "tab-predict"},
                            ],
                            value="tab-explore",
                            inline=True,
                            className="executive-radio-tabs",
                        ),
                        html.Div(
                            className="santander-badge",
                            children=[
                                html.Span("🚲", style={"fontSize": "13px"}),
                                html.Span("800+ Stations"),
                            ],
                        ),
                        html.Div(
                            style={"display": "inline-flex", "alignItems": "center", "gap": "8px", "fontSize": "12px", "color": BRAND["text_secondary"], "backgroundColor": BRAND["bg_tile"], "border": f"1px solid {BRAND['border']}", "padding": "6px 14px", "borderRadius": "8px"},
                            children=[
                                html.Span(className="pulse-dot"),
                                html.Span("Live Open-Meteo", style={"fontWeight": "600", "color": BRAND["text_primary"]}),
                            ],
                        ),
                    ],
                ),
            ],
        ),

        # Dynamic Tab Content Area
        html.Div(id="tab-content"),
    ],
)

# -----------------------------------------------------------------------------
# Callback: Dynamic Tab Content
# -----------------------------------------------------------------------------
@app.callback(
    Output("tab-content", "children"),
    Input("main-tabs", "value"),
)
def render_tab_content(selected_tab):
    if selected_tab == "tab-explore":
        total_days = len(BIKES_DF)
        peak_hires = int(BIKES_DF["bikes_hired"].max())
        avg_hires = int(round(BIKES_DF["bikes_hired"].mean()))

        return html.Div([
            # Executive Metric Stat Ribbon
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(220px, 1fr))", "gap": "16px", "marginBottom": "24px"},
                children=[
                    html.Div(
                        className="kpi-stat-card",
                        children=[
                            html.Div("HISTORICAL DATASET", className="kpi-label"),
                            html.Div(f"{total_days:,} Days", className="kpi-value"),
                            html.Div("30 Jul 2010 – 31 Dec 2025 Archive", className="kpi-sub"),
                        ],
                    ),
                    html.Div(
                        className="kpi-stat-card",
                        children=[
                            html.Div("PEAK LONDON RIDERSHIP", className="kpi-label"),
                            html.Div(f"{peak_hires:,} Hires", className="kpi-value", style={"color": BRAND["santander_red"]}),
                            html.Div("Single-day TfL network record", className="kpi-sub"),
                        ],
                    ),
                    html.Div(
                        className="kpi-stat-card",
                        children=[
                            html.Div("AVERAGE DAILY DEMAND", className="kpi-label"),
                            html.Div(f"{avg_hires:,} / day", className="kpi-value"),
                            html.Div("Mean system-wide bike hires", className="kpi-sub"),
                        ],
                    ),
                    html.Div(
                        className="kpi-stat-card",
                        children=[
                            html.Div("ACTIVE REGRESSION MODEL", className="kpi-label"),
                            html.Div("Deployable variant (R² 0.710)", className="kpi-value", style={"color": BRAND["mint"]}),
                            html.Div("23-coefficient linear scoring formula", className="kpi-sub"),
                        ],
                    ),
                ],
            ),

            # Tile 1: Controls Tile (Dropdown & Colour Control per Assignment Prompt + Period Filter)
            html.Div(
                className="dashboard-tile",
                children=[
                    html.Div(
                        className="tile-header",
                        children=[
                            html.Div([
                                html.H3("Interactive Controls & Filters", className="tile-title"),
                                html.P("Pick an explanatory weather variable, color grouping, and historical timeframe", className="tile-subtitle"),
                            ]),
                        ],
                    ),
                    html.Div(
                        style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(280px, 1fr))", "gap": "24px", "alignItems": "center"},
                        children=[
                            # Dropdown: Weather Variable (Assignment requirement)
                            html.Div([
                                html.Label("Select Weather Variable (X-Axis)", className="control-label"),
                                dcc.Dropdown(
                                    id="weather-var-dropdown",
                                    options=[
                                        {"label": f"{info['icon']}  {info['label']}", "value": var}
                                        for var, info in WEATHER_VARS.items()
                                    ],
                                    value="temp",
                                    clearable=False,
                                    className="dash-dropdown",
                                ),
                            ]),

                            # Control: Color by weekend or season_name (No brackets)
                            html.Div([
                                html.Label("Colour Scatter Points By", className="control-label"),
                                dcc.RadioItems(
                                    id="color-by-selector",
                                    options=[
                                        {"label": "Weekend vs Weekday", "value": "weekend"},
                                        {"label": "Season of the Year", "value": "season_name"},
                                    ],
                                    value="weekend",
                                    inline=True,
                                    className="custom-radio-items",
                                ),
                            ]),

                            # Control: Date Range Filter (Like Avocado Dashboard)
                            html.Div([
                                html.Label("Filter Date Range", className="control-label"),
                                dcc.DatePickerRange(
                                    id="date-range-filter",
                                    start_date=BIKES_DF["date"].min().date(),
                                    end_date=BIKES_DF["date"].max().date(),
                                    min_date_allowed=BIKES_DF["date"].min().date(),
                                    max_date_allowed=BIKES_DF["date"].max().date(),
                                    display_format="YYYY-MM-DD",
                                ),
                            ]),
                        ],
                    ),
                ],
            ),

            # Visualizations Row: Two Distinct Black Graph Tiles
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(470px, 1fr))", "gap": "24px"},
                children=[
                    # Tile 2: Scatter Plot Tile
                    html.Div(
                        className="dashboard-tile",
                        children=[
                            html.Div(
                                className="tile-header",
                                children=[
                                    html.Div([
                                        html.H3("Daily Bikes Hired vs. Weather Variable", className="tile-title"),
                                        html.P("Scatter distribution with OLS linear trendline", className="tile-subtitle"),
                                    ]),
                                    html.Div(id="correlation-badge"),
                                ],
                            ),
                            dcc.Loading(dcc.Graph(id="weather-scatter-plot", config={"displayModeBar": False})),
                        ],
                    ),

                    # Tile 3: Day of Week Bar Chart Tile
                    html.Div(
                        className="dashboard-tile",
                        children=[
                            html.Div(
                                className="tile-header",
                                children=[
                                    html.Div([
                                        html.H3("Average Hires by Day of Week", className="tile-title"),
                                        html.P("Mean daily bike hires from Monday through Sunday (ordered)", className="tile-subtitle"),
                                    ]),
                                    html.Span("Weekday vs Weekend", className="table-badge badge-weekday"),
                                ],
                            ),
                            dcc.Loading(dcc.Graph(id="dow-bar-chart", config={"displayModeBar": False})),
                        ],
                    ),
                ],
            ),
        ])

    else:
        # TAB 2: PREDICT
        coeffs_df = load_model_coefficients()
        jan_weather, jan_error = get_january_2026_weather()
        live_weather, live_error = get_live_forecast_weather()

        jan_weather["predicted_hires"] = predict_bikes(jan_weather, coeffs_df)
        live_weather["predicted_hires"] = predict_bikes(live_weather, coeffs_df)

        numeric_terms = [t for t in coeffs_df["term"] if t != "Intercept" and not t.startswith("day_")]
        intercept_val = coeffs_df.loc[coeffs_df["term"] == "Intercept", "coefficient"].values[0]

        return html.Div([
            # Optional Status Notifications
            html.Div([
                html.Div(f"⚡ {jan_error}", className="alert-banner") if jan_error else html.Div(),
                html.Div(f"⚡ {live_error}", className="alert-banner") if live_error else html.Div(),
            ]),

            # Tile 4: First Week of January 2026 (Open-Meteo Archive)
            html.Div(
                className="dashboard-tile",
                children=[
                    html.Div(
                        className="tile-header",
                        children=[
                            html.Div([
                                html.H3("1. First Week of January 2026 (Open-Meteo Archive)", className="tile-title"),
                                html.P("open_meteo_history('London', '2026-01-01', '2026-01-07') evaluated through your linear model", className="tile-subtitle"),
                            ]),
                            html.Span("Historical Benchmark", className="table-badge badge-weekday"),
                        ],
                    ),

                    # Grid: Table + Horizontal Bar Chart (Each in Distinct Subtile)
                    html.Div(
                        className="prediction-split-grid",
                        children=[
                            html.Div(
                                className="dashboard-subtile",
                                children=[
                                    html.Div(
                                        className="subtile-header",
                                        children=[
                                            html.Div("📋 Historical Weather & Prediction Table", className="subtile-title"),
                                            html.Span("7-Day Archive", className="subtile-badge"),
                                        ],
                                    ),
                                    render_styled_table(jan_weather),
                                ],
                            ),
                            html.Div(
                                className="dashboard-subtile",
                                children=[
                                    html.Div(
                                        className="subtile-header",
                                        children=[
                                            html.Div("📊 Predicted Daily Hires", className="subtile-title"),
                                            html.Span("Model Output", className="subtile-badge"),
                                        ],
                                    ),
                                    dcc.Graph(
                                        figure=build_prediction_bar_chart(
                                            jan_weather,
                                            accent_color=BRAND["santander_red"],
                                        ),
                                        config={"displayModeBar": False},
                                        style={"width": "100%"},
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),

            # Tile 5: Next Five Days in London (Live Open-Meteo Forecast)
            html.Div(
                className="dashboard-tile",
                children=[
                    html.Div(
                        className="tile-header",
                        children=[
                            html.Div([
                                html.H3("2. Next Five Days in London (Live Open-Meteo Forecast)", className="tile-title"),
                                html.P("open_meteo('London', 5) real-time weather & predicted hires", className="tile-subtitle"),
                            ]),
                            html.Div(
                                style={"display": "flex", "alignItems": "center", "gap": "10px"},
                                children=[
                                    html.Button("⬇️ Export Forecast (CSV)", id="btn-export-csv", className="action-btn"),
                                    html.Span("Live 5-Day Forecast", className="table-badge badge-prediction"),
                                ],
                            ),
                        ],
                    ),

                    # Grid: Table + Horizontal Bar Chart (Each in Distinct Subtile)
                    html.Div(
                        className="prediction-split-grid",
                        children=[
                            html.Div(
                                className="dashboard-subtile",
                                children=[
                                    html.Div(
                                        className="subtile-header",
                                        children=[
                                            html.Div("📋 Live Forecast & Prediction Table", className="subtile-title"),
                                            html.Span("5-Day Live Feed", className="subtile-badge"),
                                        ],
                                    ),
                                    render_styled_table(live_weather),
                                ],
                            ),
                            html.Div(
                                className="dashboard-subtile",
                                children=[
                                    html.Div(
                                        className="subtile-header",
                                        children=[
                                            html.Div("📊 Upcoming Predicted Hires", className="subtile-title"),
                                            html.Span("Forecast Output", className="subtile-badge"),
                                        ],
                                    ),
                                    dcc.Graph(
                                        figure=build_prediction_bar_chart(
                                            live_weather,
                                            accent_color=BRAND["santander_red"],
                                        ),
                                        config={"displayModeBar": False},
                                        style={"width": "100%"},
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),

            # Tile 6: Interactive "What-If" Demand Simulator (Executive Polish)
            html.Div(
                className="dashboard-tile",
                style={"paddingBottom": "38px"},
                children=[
                    html.Div(
                        className="tile-header",
                        children=[
                            html.Div([
                                html.H3("3. Interactive 'What-If' Demand Simulator", className="tile-title", style={"color": BRAND["text_primary"]}),
                                html.P("Simulate custom weather conditions and evaluate your regression model live", className="tile-subtitle"),
                            ]),
                            html.Span("Real-Time Calculator", className="table-badge badge-weekday"),
                        ],
                    ),

                    html.Div(
                        className="simulator-layout",
                        children=[
                            # Left: All Simulation Controls (Day Bar + 2 Columns of 3 Sliders)
                            html.Div(
                                className="simulator-controls-container",
                                children=[
                                    # Full-Length Day Bar Across the Visual Top
                                    html.Div([
                                        html.Div(
                                            style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "8px"},
                                            children=[
                                                html.Label("Day of Week", className="control-label", style={"margin": "0"}),
                                                html.Span("Simulating weekly commuter cycle vs weekend demand", style={"fontSize": "12px", "color": BRAND["text_muted"]}),
                                            ],
                                        ),
                                        dcc.RadioItems(
                                            id="sim-day",
                                            options=[{"label": d, "value": d} for d in WEEKDAY_ORDER],
                                            value="Wed",
                                            inline=True,
                                            className="sim-day-bar",
                                        ),
                                    ], style={"marginBottom": "24px"}),

                                    # 2 Columns of 3 Sliders Side-by-Side (Shortened Tracks & Min/Max Only Marks)
                                    html.Div(
                                        className="simulator-sliders-2col",
                                        children=[
                                            # Column A: Primary Conditions (Temp, Precip, Humidity)
                                            html.Div([
                                                html.Div([
                                                    html.Div(
                                                        style={"display": "flex", "justifyContent": "space-between", "marginBottom": "6px", "padding": "0 6px"},
                                                        children=[
                                                            html.Label("Temperature", className="control-label", style={"margin": "0"}),
                                                            html.Span(id="sim-temp-display", style={"fontWeight": "700", "color": BRAND["ice"], "fontSize": "13px"}),
                                                        ],
                                                    ),
                                                    html.Div(
                                                        style={"padding": "0 30px"},
                                                        children=[
                                                            dcc.Slider(
                                                                id="sim-temp",
                                                                min=-5, max=35, step=1, value=20,
                                                                allow_direct_input=False,
                                                                marks={
                                                                    -5: {"label": "-5°C", "style": {"color": "#FFFFFF", "fontWeight": "600", "fontSize": "11.5px"}},
                                                                    35: {"label": "35°C", "style": {"color": "#FFFFFF", "fontWeight": "600", "fontSize": "11.5px"}},
                                                                },
                                                            ),
                                                        ],
                                                    ),
                                                ], style={"marginBottom": "26px"}),

                                                html.Div([
                                                    html.Div(
                                                        style={"display": "flex", "justifyContent": "space-between", "marginBottom": "6px", "padding": "0 6px"},
                                                        children=[
                                                            html.Label("Precipitation", className="control-label", style={"margin": "0"}),
                                                            html.Span(id="sim-precip-display", style={"fontWeight": "700", "color": BRAND["mint"], "fontSize": "13px"}),
                                                        ],
                                                    ),
                                                    html.Div(
                                                        style={"padding": "0 30px"},
                                                        children=[
                                                            dcc.Slider(
                                                                id="sim-precip",
                                                                min=0, max=25, step=1, value=0,
                                                                allow_direct_input=False,
                                                                marks={
                                                                    0: {"label": "0 mm", "style": {"color": "#FFFFFF", "fontWeight": "600", "fontSize": "11.5px"}},
                                                                    25: {"label": "25 mm", "style": {"color": "#FFFFFF", "fontWeight": "600", "fontSize": "11.5px"}},
                                                                },
                                                            ),
                                                        ],
                                                    ),
                                                ], style={"marginBottom": "26px"}),

                                                html.Div(style={"paddingBottom": "14px"}, children=[
                                                    html.Div(
                                                        style={"display": "flex", "justifyContent": "space-between", "marginBottom": "6px", "padding": "0 6px"},
                                                        children=[
                                                            html.Label("Humidity", className="control-label", style={"margin": "0"}),
                                                            html.Span(id="sim-humidity-display", style={"fontWeight": "700", "color": BRAND["amber"], "fontSize": "13px"}),
                                                        ],
                                                    ),
                                                    html.Div(
                                                        style={"padding": "0 30px"},
                                                        children=[
                                                            dcc.Slider(
                                                                id="sim-humidity",
                                                                min=30, max=100, step=5, value=65,
                                                                allow_direct_input=False,
                                                                marks={
                                                                    30: {"label": "30%", "style": {"color": "#FFFFFF", "fontWeight": "600", "fontSize": "11.5px"}},
                                                                    100: {"label": "100%", "style": {"color": "#FFFFFF", "fontWeight": "600", "fontSize": "11.5px"}},
                                                                },
                                                            ),
                                                        ],
                                                    ),
                                                ]),
                                            ]),

                                            # Column B: Atmospheric Factors (Wind, Cloud Cover, Holiday)
                                            html.Div([
                                                html.Div([
                                                    html.Div(
                                                        style={"display": "flex", "justifyContent": "space-between", "marginBottom": "6px", "padding": "0 6px"},
                                                        children=[
                                                            html.Label("Wind Speed", className="control-label", style={"margin": "0"}),
                                                            html.Span(id="sim-wind-display", style={"fontWeight": "700", "color": BRAND["ice"], "fontSize": "13px"}),
                                                        ],
                                                    ),
                                                    html.Div(
                                                        style={"padding": "0 30px"},
                                                        children=[
                                                            dcc.Slider(
                                                                id="sim-wind",
                                                                min=0, max=50, step=1, value=14,
                                                                allow_direct_input=False,
                                                                marks={
                                                                    0: {"label": "0 km/h", "style": {"color": "#FFFFFF", "fontWeight": "600", "fontSize": "11.5px"}},
                                                                    50: {"label": "50 km/h", "style": {"color": "#FFFFFF", "fontWeight": "600", "fontSize": "11.5px"}},
                                                                },
                                                            ),
                                                        ],
                                                    ),
                                                ], style={"marginBottom": "26px"}),

                                                html.Div([
                                                    html.Div(
                                                        style={"display": "flex", "justifyContent": "space-between", "marginBottom": "6px", "padding": "0 6px"},
                                                        children=[
                                                            html.Label("Cloud Cover", className="control-label", style={"margin": "0"}),
                                                            html.Span(id="sim-solar-display", style={"fontWeight": "700", "color": BRAND["amber"], "fontSize": "13px"}),
                                                        ],
                                                    ),
                                                    html.Div(
                                                        style={"padding": "0 30px"},
                                                        children=[
                                                            dcc.Slider(
                                                                id="sim-solar",
                                                                min=0, max=100, step=1, value=50,
                                                                allow_direct_input=False,
                                                                marks={
                                                                    0: {"label": "0% clear", "style": {"color": "#FFFFFF", "fontWeight": "600", "fontSize": "11.5px"}},
                                                                    100: {"label": "100% overcast", "style": {"color": "#FFFFFF", "fontWeight": "600", "fontSize": "11.5px"}},
                                                                },
                                                            ),
                                                        ],
                                                    ),
                                                ], style={"marginBottom": "26px"}),

                                                html.Div(style={"paddingBottom": "14px"}, children=[
                                                    html.Div(
                                                        style={"display": "flex", "justifyContent": "space-between", "marginBottom": "6px", "padding": "0 6px"},
                                                        children=[
                                                            html.Label("Holiday Period", className="control-label", style={"margin": "0"}),
                                                            html.Span(id="sim-vis-display", style={"fontWeight": "700", "color": BRAND["mint"], "fontSize": "13px"}),
                                                        ],
                                                    ),
                                                    html.Div(
                                                        style={"padding": "0 30px"},
                                                        children=[
                                                            dcc.Slider(
                                                                id="sim-vis",
                                                                min=0, max=1, step=1, value=0,
                                                                allow_direct_input=False,
                                                                marks={
                                                                    0: {"label": "Normal day", "style": {"color": "#FFFFFF", "fontWeight": "600", "fontSize": "11.5px"}},
                                                                    1: {"label": "Christmas week", "style": {"color": "#FFFFFF", "fontWeight": "600", "fontSize": "11.5px"}},
                                                                },
                                                            ),
                                                        ],
                                                    ),
                                                ]),
                                            ]),
                                        ],
                                    ),
                                ],
                            ),

                            # Right: Prominent & Balanced Live Prediction Output Gauge Tile
                            html.Div(
                                className="sim-gauge-tile",
                                children=[
                                    html.Div("SIMULATED DAILY PREDICTION", style={"fontSize": "11px", "fontWeight": "800", "letterSpacing": "1.4px", "color": BRAND["text_secondary"], "marginBottom": "10px"}),
                                    html.Div(
                                        id="sim-prediction-value",
                                        style={"fontSize": "48px", "fontWeight": "800", "color": BRAND["text_primary"], "fontFamily": "JetBrains Mono, monospace", "lineHeight": "1.1"},
                                    ),
                                    html.Div("bikes hired / day", style={"fontSize": "13px", "color": BRAND["text_muted"], "marginTop": "6px"}),
                                    html.Div(
                                        id="sim-status-chip",
                                        style={"marginTop": "18px"},
                                    ),
                                    html.Div(
                                        "Formula: Intercept + sum(coef * value) + day_<weekday>",
                                        style={"fontSize": "11px", "color": BRAND["text_muted"], "marginTop": "18px", "lineHeight": "1.4"},
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),

            # Tile 7: Loaded Model Terms & Formula (model_coefficients.csv)
            html.Div(
                className="dashboard-tile",
                children=[
                    html.Div(
                        className="tile-header",
                        children=[
                            html.Div([
                                html.H3("4. Loaded Model Terms & Formula (model_coefficients.csv)", className="tile-title"),
                                html.P(
                                    f"Active Model: Intercept ({intercept_val:,.1f}) + sum(coefficient * value) for {numeric_terms} + day_<weekday>",
                                    className="tile-subtitle"
                                ),
                            ]),
                            html.Span("Model Specifications", className="table-badge badge-weekday"),
                        ],
                    ),
                    html.Div(
                        style={"marginTop": "16px"},
                        children=[
                            html.Div(
                                style={"overflowX": "auto", "width": "100%"},
                                children=[
                                    html.Table(
                                        className="styled-table coeff-table",
                                        style={"width": "100%", "maxWidth": "960px"},
                                        children=[
                                            html.Thead(html.Tr([
                                                html.Th("Term", style={"width": "160px"}),
                                                html.Th("Coefficient", style={"width": "150px"}),
                                                html.Th("Descriptor & Business Interpretation"),
                                            ])),
                                            html.Tbody([
                                                html.Tr(
                                                    style={"height": "auto", "maxHeight": "none"},
                                                    children=[
                                                        html.Td(
                                                            html.Code(r["term"], style={"color": BRAND["santander_flame"], "fontWeight": "700"}),
                                                            style={"height": "auto", "whiteSpace": "nowrap"}
                                                        ),
                                                        html.Td(
                                                            f"{r['coefficient']:+,.2f}" if r["coefficient"] != 0 else "0.00 (Baseline)",
                                                            style={
                                                                "fontFamily": "JetBrains Mono, monospace",
                                                                "fontWeight": "600",
                                                                "color": BRAND["text_primary"],
                                                                "height": "auto",
                                                                "whiteSpace": "nowrap"
                                                            }
                                                        ),
                                                        html.Td(
                                                            COEFF_DESCRIPTIONS.get(r["term"], "Model term coefficient"),
                                                            style={
                                                                "color": BRAND["text_secondary"],
                                                                "fontSize": "12.5px",
                                                                "whiteSpace": "normal",
                                                                "lineHeight": "1.4",
                                                                "height": "auto"
                                                            }
                                                        ),
                                                    ]
                                                )
                                                for _, r in coeffs_df.iterrows()
                                            ]),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ])

# -----------------------------------------------------------------------------
# Callbacks: Correlation Badge
# -----------------------------------------------------------------------------
@app.callback(
    Output("correlation-badge", "children"),
    [
        Input("weather-var-dropdown", "value"),
        Input("date-range-filter", "start_date"),
        Input("date-range-filter", "end_date"),
    ],
)
def update_correlation_badge(selected_var, start_date, end_date):
    df = filter_df_by_date_range(BIKES_DF, start_date, end_date)
    if selected_var not in df.columns:
        return html.Div()
    valid = df[[selected_var, "bikes_hired"]].dropna()
    corr = valid[selected_var].corr(valid["bikes_hired"])
    corr_color = BRAND["mint"] if corr > 0.3 else (BRAND["santander_red"] if corr < -0.3 else BRAND["ice"])
    start_str = pd.to_datetime(start_date).strftime("%d %b %Y") if start_date else "Start"
    end_str = pd.to_datetime(end_date).strftime("%d %b %Y") if end_date else "End"
    return html.Span(
        f"Correlation ({start_str} – {end_str}): r = {corr:+.2f}",
        style={
            "fontSize": "12px",
            "fontWeight": "600",
            "color": corr_color,
            "backgroundColor": "rgba(236, 0, 0, 0.08)",
            "padding": "4px 10px",
            "borderRadius": "20px",
            "border": f"1px solid {BRAND['border']}",
        },
    )

# -----------------------------------------------------------------------------
# Callbacks: Tab 1 Visualizations (Scatter Plot & DOW Bar Chart)
# -----------------------------------------------------------------------------
@app.callback(
    Output("weather-scatter-plot", "figure"),
    [
        Input("weather-var-dropdown", "value"),
        Input("color-by-selector", "value"),
        Input("date-range-filter", "start_date"),
        Input("date-range-filter", "end_date"),
    ],
)
def update_scatter_plot(selected_var, color_by, start_date, end_date):
    var_info = WEATHER_VARS.get(selected_var, WEATHER_VARS["temp"])
    filtered_df = filter_df_by_date_range(BIKES_DF, start_date, end_date)

    if color_by == "weekend":
        plot_df = filtered_df.copy()
        plot_df["weekend_label"] = plot_df["weekend"].map({False: "Weekday (Mon–Fri)", True: "Weekend (Sat–Sun)"})
        color_col = "weekend_label"
        color_map = {
            "Weekday (Mon–Fri)": BRAND["weekday_color"],
            "Weekend (Sat–Sun)": BRAND["santander_red"],
        }
        labels = {"weekend_label": "Type", "bikes_hired": "Daily Bikes Hired", selected_var: var_info["label"]}
    else:
        plot_df = filtered_df
        color_col = "season_name"
        color_map = {
            "Spring": BRAND["mint"],
            "Summer": BRAND["amber"],
            "Autumn": "#F97316",
            "Winter": BRAND["ice"],
        }
        labels = {"season_name": "Season", "bikes_hired": "Daily Bikes Hired", selected_var: var_info["label"]}

    fig = px.scatter(
        plot_df,
        x=selected_var,
        y="bikes_hired",
        color=color_col,
        color_discrete_map=color_map,
        labels=labels,
        opacity=0.6,
        hover_data={"date": "|%d %b %Y", "day_of_week": True, "bikes_hired": ":,", selected_var: ":.1f"},
    )

    # Linear trendline
    valid = plot_df[[selected_var, "bikes_hired"]].dropna()
    if len(valid) > 1:
        slope, intercept = np.polyfit(valid[selected_var], valid["bikes_hired"], 1)
        x_range = np.linspace(valid[selected_var].min(), valid[selected_var].max(), 100)
        y_pred = slope * x_range + intercept
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=y_pred,
                mode="lines",
                name=f"Trendline (slope: {slope:+.1f})",
                line=dict(color="#FFFFFF", width=2.5, dash="dash"),
                hoverinfo="skip",
            )
        )

    fig.update_traces(marker=dict(size=6))
    start_str = pd.to_datetime(start_date).strftime("%d %b %Y") if start_date else "Start"
    end_str = pd.to_datetime(end_date).strftime("%d %b %Y") if end_date else "End"
    fig.update_layout(
        **get_plotly_layout(),
        height=380,
        title=dict(
            text=f"Bikes Hired vs. {var_info['label']} ({start_str} – {end_str})",
            font=dict(size=14, color=BRAND["text_primary"]),
        ),
        xaxis_title=var_info["label"],
        yaxis_title="Bikes Hired / Day",
    )
    return fig

@app.callback(
    Output("dow-bar-chart", "figure"),
    [
        Input("main-tabs", "value"),
        Input("date-range-filter", "start_date"),
        Input("date-range-filter", "end_date"),
    ],
)
def update_dow_bar_chart(_, start_date, end_date):
    filtered_df = filter_df_by_date_range(BIKES_DF, start_date, end_date)
    dow_summary = (
        filtered_df.groupby("day_of_week", observed=False)["bikes_hired"]
        .mean()
        .reindex(WEEKDAY_ORDER)
        .reset_index()
    )
    dow_summary["bikes_hired_round"] = dow_summary["bikes_hired"].round().astype(int)
    dow_summary["is_weekend"] = dow_summary["day_of_week"].isin(["Sat", "Sun"])

    # Santander Red for Weekend, Slate/Steel for Weekday
    bar_colors = [BRAND["santander_red"] if is_w else "#3F3F46" for is_w in dow_summary["is_weekend"]]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=dow_summary["day_of_week"],
            y=dow_summary["bikes_hired_round"],
            marker_color=bar_colors,
            marker_line=dict(color="rgba(255,255,255,0.15)", width=1),
            text=[f"{v:,}" for v in dow_summary["bikes_hired_round"]],
            textposition="outside",
            textfont=dict(color=BRAND["text_primary"], size=12, family="Outfit"),
            hovertemplate="<b>%{x}</b><br>Average Hires: %{y:,.0f}<extra></extra>",
        )
    )

    start_str = pd.to_datetime(start_date).strftime("%d %b %Y") if start_date else "Start"
    end_str = pd.to_datetime(end_date).strftime("%d %b %Y") if end_date else "End"
    fig.update_layout(
        **get_plotly_layout(),
        height=380,
        title=dict(
            text=f"Average Daily Hires by Day of Week ({start_str} – {end_str})",
            font=dict(size=14, color=BRAND["text_primary"]),
        ),
        xaxis_title="Day of Week",
        yaxis_title="Mean Bikes Hired",
        showlegend=False,
    )
    fig.update_yaxes(range=[0, dow_summary["bikes_hired_round"].max() * 1.18])
    return fig

# -----------------------------------------------------------------------------
# Callbacks: What-If Demand Simulator
# -----------------------------------------------------------------------------
@app.callback(
    [
        Output("sim-temp-display", "children"),
        Output("sim-precip-display", "children"),
        Output("sim-humidity-display", "children"),
        Output("sim-wind-display", "children"),
        Output("sim-solar-display", "children"),
        Output("sim-vis-display", "children"),
        Output("sim-prediction-value", "children"),
        Output("sim-status-chip", "children"),
    ],
    [
        Input("sim-day", "value"),
        Input("sim-temp", "value"),
        Input("sim-precip", "value"),
        Input("sim-humidity", "value"),
        Input("sim-wind", "value"),
        Input("sim-solar", "value"),
        Input("sim-vis", "value"),
    ],
)
def update_simulator(day, temp, precip, humidity, wind, cloud, holiday):
    coeffs_df = load_model_coefficients()
    sim_row = pd.DataFrame([{
        "date": pd.Timestamp.now().normalize(),
        "day_of_week": day,
        "temp": float(temp),
        "precip": float(precip),
        "humidity": float(humidity),
        "windspeed": float(wind),
        "cloudcover": float(cloud),
        "xmas": int(holiday),
    }])
    pred = predict_bikes(sim_row, coeffs_df)[0]

    # Category status
    if pred >= 38000:
        chip = html.Span("High Peak Demand", className="table-badge badge-prediction")
    elif pred >= 26000:
        chip = html.Span("Standard Ridership", className="table-badge badge-weekday")
    elif pred >= 16000:
        chip = html.Span("Subdued Weather Volume", className="table-badge badge-weekend")
    else:
        chip = html.Span("Low Weather Volume", className="table-badge badge-weekend")

    return (
        f"{temp}°C",
        f"{precip} mm",
        f"{humidity}%",
        f"{wind} km/h",
        f"{cloud}%",
        "Christmas week" if holiday else "Normal day",
        f"{pred:,}",
        chip,
    )

# -----------------------------------------------------------------------------
# Callback: CSV Export (Guarded by PreventUpdate to stop accidental download on tab switch)
# -----------------------------------------------------------------------------
@app.callback(
    Output("download-forecast-csv", "data"),
    Input("btn-export-csv", "n_clicks"),
    prevent_initial_call=True,
)
def export_forecast_csv(n_clicks):
    # Guard: Do not execute unless button was actively clicked
    if not n_clicks:
        raise PreventUpdate
    coeffs_df = load_model_coefficients()
    live_weather, _ = get_live_forecast_weather()
    live_weather["predicted_hires"] = predict_bikes(live_weather, coeffs_df)
    return dcc.send_data_frame(live_weather.to_csv, "london_bikes_5day_forecast.csv", index=False)

# -----------------------------------------------------------------------------
# Chart Helper: Horizontal Prediction Bar Charts for Tab 2
# -----------------------------------------------------------------------------
def build_prediction_bar_chart(df, accent_color):
    """
    Render horizontal prediction bar chart precisely matching table heights and rows.
    Each bar center lines up with its exact corresponding table row.
    Table header is 44px, and each row is locked to 46px.
    """
    n_rows = len(df)
    # Total chart height = 44px (header offset) + n_rows * 46px (row bands) + 40px (x-axis)
    chart_height = 44 + (n_rows * 46) + 40

    # Reverse so the earliest day is at the top to match table row order
    df_rev = df.iloc[::-1].copy()
    dates_str = [
        f"{pd.to_datetime(d).strftime('%a %d %b')}"
        for d in df_rev["date"]
    ]
    preds = df_rev["predicted_hires"].tolist()

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=dates_str,
            x=preds,
            orientation="h",
            marker=dict(
                color=accent_color,
                line=dict(color="rgba(255,255,255,0.2)", width=1),
            ),
            text=[f"{p:,}" for p in preds],
            textposition="outside",
            textfont=dict(color=BRAND["text_primary"], size=12, family="Outfit"),
            hovertemplate="<b>%{y}</b><br>Predicted Hires: %{x:,}<extra></extra>",
        )
    )

    layout_args = get_plotly_layout()
    # Top margin = 44px (matches 44px <th> header height of table!)
    # Bottom margin = 40px (for x-axis labels and tick numbers)
    layout_args["margin"] = dict(l=85, r=65, t=44, b=40)
    fig.update_layout(
        **layout_args,
        height=chart_height,
        bargap=0.35,
        xaxis_title="Predicted Hires / Day",
        yaxis_title="",
        showlegend=False,
    )
    if preds:
        fig.update_xaxes(range=[0, max(preds) * 1.25])

    return fig

# -----------------------------------------------------------------------------
# Local Dev Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    print(f"Starting London Bikes Dash Application on port {port}...")
    print(f"Open in browser: http://localhost:{port} or http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)
