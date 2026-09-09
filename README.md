# 🚲 London Bikes — Demand Exploration & Forecasting Dashboard

A high-performance interactive web dashboard built with **Plotly & Dash**, analyzing Transport for London (TfL) Santander Cycles ridership and predicting future demand using live meteorological data from **Open-Meteo**.

Deployed to **Render** with **uv** and **Gunicorn**.

---

## 📁 Project Structure

```
dashboard/
├── app.py                   # Main Dash application (Tabs 1 & 2, callbacks, server export)
├── open_meteo.py            # Open-Meteo historical archive & live forecast weather client
├── model_coefficients.csv   # Linear regression coefficients (data contract)
├── pyproject.toml           # Project dependencies managed via uv
├── render.yaml              # 1-Click Render blueprint configuration
├── assets/
│   └── custom.css           # Premium TfL dark theme design system
└── README.md                # Documentation and deployment walkthrough
```

---

## 🚀 Quickstart: Running Locally

### 1. Using `uv` (Recommended)
From this directory:
```bash
# Sync dependencies
uv sync

# Run the app locally
uv run python app.py
```
Open [http://localhost:8050](http://localhost:8050) in your web browser.

### 2. Using Standard Python / Virtualenv
```bash
pip install dash plotly pandas requests gunicorn
python app.py
```

---

## 🌐 Deploying to Render (Step-by-Step)

The assignment requires a **public GitHub repository** and a **live URL on Render**.

### Step 1: Create a Public GitHub Repository
1. Go to [github.com/new](https://github.com/new).
2. Name your repository (e.g. `london-bikes-dashboard`) and make sure it is set to **Public**.
3. Push this `dashboard` folder to your new repo:
   ```bash
   cd session10_workshop5/dashboard
   git init
   git add .
   git commit -m "Initial commit: London Bikes Dash application"
   git branch -M main
   git remote add origin https://github.com/<your-username>/london-bikes-dashboard.git
   git push -u origin main
   ```

### Step 2: Create a Web Service on Render
1. Go to your [Render Dashboard](https://dashboard.render.com).
2. Click **New +** → **Web Service**.
3. Connect your new `london-bikes-dashboard` GitHub repository.
4. Render will automatically detect settings or you can fill in:
   - **Name**: `london-bikes-dashboard`
   - **Environment**: `Python`
   - **Build Command**: `pip install uv && uv sync`
   - **Start Command**: `uv run gunicorn app:server`
   - **Plan Type**: `Free`
5. Click **Deploy Web Service**.
6. Once deployed, copy your live Render URL (e.g., `https://london-bikes-dashboard.onrender.com`) and paste it into the submission cell of `bikes_assignment.ipynb`!

---

## 🔄 Hot-Swapping the Final Model (`model_coefficients.csv`)

When your modeling team completes Parts 2–4 of `bikes_assignment.ipynb`:
1. Run Cell 32 in the notebook to export your final model:
   ```python
   export_coefficients(best_model, numeric_vars, path="model_coefficients.csv")
   ```
2. Copy the resulting `model_coefficients.csv` into this directory:
   - Overwrite `dashboard/model_coefficients.csv`.
3. Commit and push to GitHub:
   ```bash
   git add model_coefficients.csv
   git commit -m "Update model coefficients with final regression model"
   git push
   ```
4. Render will automatically redeploy within seconds! The dashboard will immediately apply your new model coefficients to both the January 2026 benchmark and the 5-day live forecast without touching any Python code.

---

## ✨ Features

- **Tab 1: Explore the Data**:
  - Interactive weather variable selector (Temperature, Humidity, Precipitation, Wind Speed, Cloud Cover).
  - Scatter plot with OLS trendline and toggleable color grouping (`weekend` vs `season_name`).
  - Average daily hires by day of week bar chart (Monday through Sunday).
  - Four executive KPI cards summarizing volume, temperature, and historical peaks.

- **Tab 2: Predict Demand (Open-Meteo)**:
  - **First Week of January 2026**: Evaluates your model on historical Open-Meteo archive weather (`open_meteo_history`).
  - **Live 5-Day Forecast**: Evaluates your model on real-time London forecast weather (`open_meteo`).
  - Formatted data tables with styled weekday/weekend badges.
  - Interactive bar charts comparing daily predicted bike hires.
  - Collapsible model inspector displaying active coefficients.
  - Resilient offline fallback so the app never crashes if Open-Meteo has connection issues.
