"""
streamlit_app.py
================
Unified Streamlit interface for the Tripura Crop Intelligence System.

Tabs:
  1. 🏠 Overview           — project summary & model metrics
  2. 🔴 Shortage Alerts    — browse predictions.json alert results
  3. 🌱 Crop Recommender   — interactive yield prediction
  4. 💧 Irrigation Planner — 7-day irrigation schedule (calls Open-Meteo)
  5. 📈 Model Comparison   — benchmark chart across 5 models

Run:
    streamlit run streamlit_app.py
"""

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Tripura Crop Intelligence",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CONSTANTS ─────────────────────────────────────────────────────────────────

YIELD_COL = "Yield (Tonne or Bales/Hectare)"

VALID_CROPS = [
    "Rice", "Maize", "Wheat", "Jute", "Groundnut", "Rapeseed &Mustard",
    "Masoor", "Moong(Green Gram)", "Urad", "Sugarcane", "Arhar/Tur",
    "Cotton(lint)", "Mesta", "Peas & beans (Pulses)", "Sesamum",
]

DISTRICTS = [
    "Dhalai", "Gomati", "Khowai", "North tripura",
    "Sepahijala", "South tripura", "Unakoti", "West tripura",
]

SEASONS = ["Kharif", "Rabi", "Autumn", "Summer", "Winter", "Whole Year"]

PEST_MAP = {"Low": 0, "Medium": 1, "High": 2}

WEATHER_FEATURES = [
    "weather_temp_mean", "weather_rain_total", "weather_rain_days",
    "weather_et0_total", "weather_wind_mean", "weather_solarrad_total",
]

DISTRICT_COORDS = {
    "Dhalai":        (24.17, 92.03),
    "Gomati":        (23.45, 91.65),
    "Khowai":        (24.07, 91.60),
    "North tripura": (24.45, 92.02),
    "Sepahijala":    (23.57, 91.30),
    "South tripura": (23.23, 91.73),
    "Unakoti":       (24.32, 92.08),
    "West tripura":  (23.84, 91.28),
}

# ── LOADERS ───────────────────────────────────────────────────────────────────

@st.cache_resource
def load_artefacts():
    p = Path("model_artefacts.pkl")
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_predictions():
    p = Path("predictions.json")
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)

@st.cache_data
def load_model_comparison():
    p = Path("model_comparison.json")
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)

@st.cache_data
def load_dataset():
    p = Path("merged_crop_enriched_features_del.xlsx")
    if not p.exists():
        return None
    return pd.read_excel(p)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a9/Flag_of_Tripura.svg/200px-Flag_of_Tripura.svg.png", width=120)
    st.title("🌾 Tripura Crop Intelligence")
    st.caption("AI-powered precision agriculture for Tripura")
    st.divider()

    art = load_artefacts()
    preds = load_predictions()

    if art:
        st.success(f"✅ Model loaded · {len(art['feat_cols'])} features")
    else:
        st.warning("⚠️ model_artefacts.pkl not found")

    if preds:
        s = preds["summary"]
        st.metric("Critical Alerts", s["critical"])
        st.metric("Watch Alerts", s["watch"])
        st.metric("Predictions", s["total"])
    else:
        st.info("Run generate_alerts.py to create predictions.json")

    st.divider()
    st.caption(f"Data: Open-Meteo · FAO-56 · Tripura Ag Dept")

# ── TABS ─────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Overview",
    "🔴 Shortage Alerts",
    "🌱 Crop Recommender",
    "💧 Irrigation Planner",
    "📈 Model Comparison",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.header("Tripura Crop Intelligence System")
    st.markdown("""
    An end-to-end AI platform for precision agriculture across Tripura's 8 districts.
    Built on **XGBoost** trained on district-level historical yield, weather, soil, pest, and fertilizer data.
    """)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Districts", "8")
    col2.metric("Crops", "15")
    col3.metric("Seasons", "6")
    col4.metric("Model R²", "0.9981")

    st.divider()

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("📦 System Components")
        st.markdown("""
        | Component | Description |
        |---|---|
        | `crop_yield_with_weather.py` | Model training pipeline |
        | `generate_alerts.py` | Batch alert engine |
        | `backend.py` | Flask API (crop recommender) |
        | `irrigation_backend2.py` | Flask API (irrigation) |
        | `crop_recommender.html` | Recommendation UI |
        | `alert_dashboard.html` | Alert map |
        | `crop_dashboard.html` | Analytics dashboard |
        | `irrigation_advisory1.html` | Irrigation planner |
        """)

    with col_r:
        st.subheader("🌍 District Coverage")
        df_map = pd.DataFrame([
            {"District": d, "lat": c[0], "lon": c[1]}
            for d, c in DISTRICT_COORDS.items()
        ])
        st.map(df_map, latitude="lat", longitude="lon", size=5000)

    st.divider()

    df = load_dataset()
    if df is not None:
        st.subheader("📊 Dataset Overview")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Records", len(df))
        c2.metric("Unique Crops", df["Crop"].nunique() if "Crop" in df.columns else "—")
        c3.metric("Unique Districts", df["District_Name"].nunique() if "District_Name" in df.columns else "—")
        if YIELD_COL in df.columns:
            c4.metric("Median Yield (t/ha)", f"{df[YIELD_COL].median():.2f}")

        if YIELD_COL in df.columns and "Crop" in df.columns:
            top_yields = (
                df.groupby("Crop")[YIELD_COL]
                .median()
                .sort_values(ascending=False)
                .reset_index()
            )
            top_yields.columns = ["Crop", "Median Yield (t/ha)"]
            st.bar_chart(top_yields.set_index("Crop"))

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — SHORTAGE ALERTS
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.header("🔴 Crop Shortage Alerts")

    if preds is None:
        st.warning("No predictions.json found. Run `python generate_alerts.py` to generate alerts.")
        st.code("python generate_alerts.py", language="bash")
    else:
        s = preds["summary"]
        st.caption(f"Generated: {preds.get('run_date', 'unknown')} · Model: {preds.get('model_version', '')}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Predictions", s["total"])
        c2.metric("🔴 Critical", s["critical"], delta=f"≤ {preds['critical_threshold']}%", delta_color="inverse")
        c3.metric("🟡 Watch", s["watch"], delta=f"≤ {preds['alert_threshold']}%", delta_color="inverse")
        c4.metric("🟢 Normal", s["normal"])

        st.divider()

        # Filters
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            sel_status = st.multiselect("Filter by status", ["critical", "watch", "normal"],
                                         default=["critical", "watch"])
        with col_f2:
            sel_district = st.multiselect("Filter by district", DISTRICTS)
        with col_f3:
            sel_crop = st.multiselect("Filter by crop", VALID_CROPS)

        rows = preds["predictions"]
        if sel_status:
            rows = [r for r in rows if r["status"] in sel_status]
        if sel_district:
            rows = [r for r in rows if r["district"] in sel_district]
        if sel_crop:
            rows = [r for r in rows if r["crop"] in sel_crop]

        df_alerts = pd.DataFrame([{
            "District": r["district"],
            "Crop": r["crop"],
            "Season": r["season"],
            "Predicted (t/ha)": r["predicted"],
            "Normal (t/ha)": r["normal"],
            "Anomaly %": r["anomaly"],
            "Status": r["status"].upper(),
            "Rain (mm)": r["weather"]["rain_total"],
            "Temp (°C)": r["weather"]["temp_mean"],
        } for r in rows])

        def color_status(val):
            if val == "CRITICAL":
                return "background-color: #ffcccc; color: #cc0000; font-weight: bold"
            if val == "WATCH":
                return "background-color: #fff3cd; color: #856404; font-weight: bold"
            return "background-color: #d4edda; color: #155724"

        if not df_alerts.empty:
            styled = df_alerts.style.applymap(color_status, subset=["Status"])
            st.dataframe(styled, use_container_width=True, height=500)
            st.download_button(
                "⬇️ Download filtered alerts as CSV",
                df_alerts.to_csv(index=False),
                "alerts.csv",
                "text/csv",
            )
        else:
            st.info("No alerts match the selected filters.")

        st.divider()
        st.subheader("Top 10 Worst Anomalies")
        worst = sorted(preds["predictions"], key=lambda x: x["anomaly"])[:10]
        df_worst = pd.DataFrame([{
            "District": r["district"], "Crop": r["crop"],
            "Season": r["season"], "Anomaly %": r["anomaly"],
        } for r in worst])
        st.bar_chart(
            df_worst.assign(label=df_worst["District"] + " · " + df_worst["Crop"])
            .set_index("label")["Anomaly %"]
        )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — CROP RECOMMENDER
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.header("🌱 AI Crop Recommender")
    st.markdown("Enter your field conditions and get ranked crop recommendations with predicted yields.")

    if art is None:
        st.warning("model_artefacts.pkl not found. Run `crop_yield_with_weather.py` first.")
    else:
        model = art["model"]
        feat_cols = art["feat_cols"]
        scaler = art["scaler"]
        crop_stats = art["crop_stats"]
        df_history = art["df_history"]

        col_l, col_r = st.columns([1, 2])
        with col_l:
            st.subheader("Field Inputs")
            district = st.selectbox("District", DISTRICTS)
            season   = st.selectbox("Season", SEASONS)
            area_ha  = st.number_input("Area (hectares)", min_value=0.1, max_value=1000.0, value=2.0, step=0.5)
            fert_kgha = st.number_input("Fertilizer (kg/ha)", min_value=0.0, max_value=500.0, value=80.0, step=5.0)
            pest     = st.selectbox("Pest/Disease Incidence", ["Low", "Medium", "High"])
            st.markdown("**Weather (seasonal)**")
            rain_total = st.number_input("Total Rainfall (mm)", 0.0, 3000.0, 900.0, 50.0)
            rain_days  = st.number_input("Rainy Days", 0, 200, 60, 5)
            temp_mean  = st.number_input("Mean Temperature (°C)", 10.0, 45.0, 28.0, 0.5)
            et0_total  = st.number_input("Total ET₀ (mm)", 0.0, 2000.0, 600.0, 50.0)
            wind_mean  = st.number_input("Mean Wind Speed (km/h)", 0.0, 50.0, 8.0, 0.5)
            solar_total = st.number_input("Solar Radiation (MJ/m²)", 0.0, 5000.0, 1800.0, 50.0)
            predict_btn = st.button("🔍 Get Recommendations", type="primary", use_container_width=True)

        with col_r:
            if predict_btn:
                user_wx = {
                    "weather_temp_mean":      temp_mean,
                    "weather_rain_total":     rain_total,
                    "weather_rain_days":      rain_days,
                    "weather_et0_total":      et0_total,
                    "weather_wind_mean":      wind_mean,
                    "weather_solarrad_total": solar_total,
                }

                results = []
                for crop in VALID_CROPS:
                    if crop not in crop_stats.index:
                        continue
                    mu  = crop_stats.loc[crop, "crop_mean"]
                    std = crop_stats.loc[crop, "crop_std"]
                    if std == 0:
                        continue

                    # Historical lags from df_history
                    hist = df_history[
                        (df_history["District_Name"] == district) &
                        (df_history["Crop"] == crop)
                    ] if "Crop" in df_history.columns else pd.DataFrame()

                    if len(hist) >= 3:
                        last3 = hist[YIELD_COL].values[-3:]
                    else:
                        last3 = np.array([mu, mu, mu])

                    yl1 = last3[-1]
                    yr3 = float(np.mean(last3))
                    yt  = float(np.polyfit(range(3), last3, 1)[0])

                    row = {
                        "District_Name":          district,
                        "Crop":                   crop,
                        "Area (Hectare)":         area_ha,
                        "Fertilizer_kg_per_ha":   fert_kgha,
                        "Pest_Disease_Incidence": PEST_MAP[pest],
                        "Yield_Lag1":             (yl1 - mu) / std,
                        "Yield_Roll3":            (yr3 - mu) / std,
                        "Yield_Trend":             yt       / std,
                        **user_wx,
                    }

                    row_df = pd.get_dummies(pd.DataFrame([row]), drop_first=True)
                    row_df = row_df.reindex(columns=feat_cols, fill_value=0)
                    row_sc = scaler.transform(row_df)
                    norm_pred  = model.predict(row_sc)[0]
                    pred_yield = norm_pred * std + mu
                    results.append({"crop": crop, "predicted_yield": round(float(pred_yield), 3)})

                results.sort(key=lambda x: x["predicted_yield"], reverse=True)

                st.subheader("📊 Crop Recommendations")
                df_rec = pd.DataFrame(results)
                df_rec.index = range(1, len(df_rec) + 1)
                df_rec.columns = ["Crop", "Predicted Yield (t/ha)"]

                top3_colors = ["#FFD700", "#C0C0C0", "#CD7F32"]
                for i, row in df_rec.head(3).iterrows():
                    medal = ["🥇", "🥈", "🥉"][i - 1]
                    st.success(f"{medal} **{row['Crop']}** — {row['Predicted Yield (t/ha)']} t/ha")

                st.divider()
                st.bar_chart(df_rec.set_index("Crop"), use_container_width=True)
                st.dataframe(df_rec, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — IRRIGATION PLANNER
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.header("💧 Irrigation Advisory")
    st.markdown("Get a 7-day irrigation schedule based on real weather forecast and crop growth stage.")

    CROP_WATER_NEEDS = {
        "Rice":       {"stages": ["Transplanting","Vegetative","Tillering","Flowering","Grain Filling","Maturity"],
                       "duration_days": [15,25,20,15,20,15], "kc": [1.05,1.10,1.15,1.20,1.10,0.75], "critical_pct": 75},
        "Wheat":      {"stages": ["Germination","Tillering","Stem Extension","Heading","Grain Filling","Maturity"],
                       "duration_days": [15,25,20,10,20,20], "kc": [0.4,0.7,1.15,1.15,0.75,0.4], "critical_pct": 60},
        "Maize":      {"stages": ["Germination","Vegetative","Tasseling","Silking","Grain Filling","Maturity"],
                       "duration_days": [10,30,10,10,25,15], "kc": [0.4,0.8,1.15,1.20,1.05,0.6], "critical_pct": 65},
        "Groundnut":  {"stages": ["Germination","Vegetative","Flowering","Pegging","Pod Development","Maturity"],
                       "duration_days": [10,25,20,15,25,15], "kc": [0.45,0.75,1.05,1.05,0.85,0.6], "critical_pct": 60},
        "Sugarcane":  {"stages": ["Germination","Tillering","Grand Growth","Ripening"],
                       "duration_days": [35,60,150,60], "kc": [0.55,0.80,1.25,0.75], "critical_pct": 70},
        "Jute":       {"stages": ["Germination","Vegetative","Rapid Growth","Maturity"],
                       "duration_days": [10,30,60,20], "kc": [0.5,0.8,1.15,0.8], "critical_pct": 65},
    }

    col_l, col_r = st.columns([1, 2])
    with col_l:
        irr_district = st.selectbox("District", DISTRICTS, key="irr_dist")
        irr_crop     = st.selectbox("Crop", list(CROP_WATER_NEEDS.keys()), key="irr_crop")
        days_planted = st.number_input("Days since planting", 0, 365, 30, 1)
        soil_fc      = st.number_input("Field Capacity (mm/m)", 100.0, 400.0, 250.0, 10.0)
        soil_wp      = st.number_input("Wilting Point (mm/m)", 50.0, 200.0, 120.0, 10.0)
        soil_depth   = st.number_input("Root Zone Depth (m)", 0.1, 1.5, 0.6, 0.1)
        irr_btn      = st.button("💧 Get Irrigation Schedule", type="primary", use_container_width=True)

    with col_r:
        if irr_btn:
            import requests as req_lib
            import datetime

            coords = DISTRICT_COORDS[irr_district]
            today  = datetime.date.today()
            end    = today + datetime.timedelta(days=6)
            url    = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": coords[0], "longitude": coords[1],
                "daily": "precipitation_sum,et0_fao_evapotranspiration,temperature_2m_max,temperature_2m_min",
                "timezone": "Asia/Kolkata",
                "start_date": str(today), "end_date": str(end),
            }

            with st.spinner("Fetching 7-day weather forecast from Open-Meteo..."):
                try:
                    resp = req_lib.get(url, params=params, timeout=15)
                    resp.raise_for_status()
                    daily = resp.json()["daily"]
                    forecast_ok = True
                except Exception as e:
                    st.error(f"Weather fetch failed: {e}")
                    forecast_ok = False

            if forecast_ok:
                crop_info  = CROP_WATER_NEEDS[irr_crop]
                stages     = crop_info["stages"]
                durations  = crop_info["duration_days"]
                kcs        = crop_info["kc"]
                crit_pct   = crop_info["critical_pct"]

                # Determine current growth stage
                cum = 0
                stage_idx = 0
                for idx, d in enumerate(durations):
                    cum += d
                    if days_planted <= cum:
                        stage_idx = idx
                        break

                current_stage = stages[stage_idx]
                kc            = kcs[stage_idx]

                st.info(f"🌿 Current growth stage: **{current_stage}** (day {days_planted}) · Kc = {kc}")

                taw = (soil_fc - soil_wp) * soil_depth  # total available water (mm)
                schedule = []

                for i, date_str in enumerate(daily["dates"]):
                    et0    = daily["et0_fao_evapotranspiration"][i] or 0
                    rain   = daily["precipitation_sum"][i] or 0
                    etc    = et0 * kc
                    deficit = max(0, etc - rain)
                    needs_irr = deficit > (taw * (1 - crit_pct / 100))
                    schedule.append({
                        "Date": date_str,
                        "Rain (mm)": round(rain, 1),
                        "ET₀ (mm)":  round(et0, 2),
                        "ETc (mm)":  round(etc, 2),
                        "Deficit (mm)": round(deficit, 2),
                        "Irrigate?": "✅ Yes" if needs_irr else "—",
                    })

                df_sched = pd.DataFrame(schedule)
                st.subheader(f"7-Day Schedule — {irr_crop} in {irr_district}")
                st.dataframe(df_sched, use_container_width=True)

                total_irr = df_sched[df_sched["Irrigate?"] == "✅ Yes"]["Deficit (mm)"].sum()
                irr_days  = (df_sched["Irrigate?"] == "✅ Yes").sum()
                col1, col2 = st.columns(2)
                col1.metric("Irrigation Days (next 7d)", int(irr_days))
                col2.metric("Total Estimated Need (mm)", round(total_irr, 1))

                st.line_chart(df_sched.set_index("Date")[["Rain (mm)", "ETc (mm)"]])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — MODEL COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.header("📈 Model Performance Comparison")

    mc = load_model_comparison()
    if mc is None:
        st.warning("model_comparison.json not found.")
    else:
        eval_sets = ["test_all", "test_core", "future_all", "future_core"]
        metrics   = ["r2", "mape", "rmse", "mae"]
        metric_labels = {"r2": "R²", "mape": "MAPE (%)", "rmse": "RMSE", "mae": "MAE"}

        sel_eval   = st.selectbox("Evaluation set", eval_sets, index=0)
        sel_metric = st.selectbox("Metric", metrics, format_func=lambda x: metric_labels[x])

        rows = []
        for model_name, evals in mc.items():
            if sel_eval in evals:
                rows.append({
                    "Model": model_name,
                    metric_labels[sel_metric]: round(evals[sel_eval][sel_metric], 6),
                })

        df_cmp = pd.DataFrame(rows).sort_values(
            metric_labels[sel_metric],
            ascending=(sel_metric != "r2")
        ).set_index("Model")

        st.bar_chart(df_cmp)

        st.divider()
        st.subheader("Full Benchmark Table")
        all_rows = []
        for model_name, evals in mc.items():
            for eset in eval_sets:
                if eset in evals:
                    all_rows.append({
                        "Model": model_name,
                        "Eval Set": eset,
                        **{metric_labels[m]: round(evals[eset][m], 4) for m in metrics},
                    })
        st.dataframe(pd.DataFrame(all_rows), use_container_width=True)

        st.divider()
        st.subheader("Why XGBoost was Selected")
        st.markdown("""
        XGBoost was selected as the production model based on:
        - **Best future-set R²** (0.9965) — indicating strong generalization to unseen years
        - **Balanced MAPE** across test and future sets (~10.2%)
        - **Fast inference** suitable for batch prediction over 176 combos
        - Gradient Boosting achieved slightly lower MAPE but XGBoost's tree structure
          handles the seasonal dummy features more robustly
        """)

# ── FOOTER ────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "Tripura Crop Intelligence System · "
    "Weather data: Open-Meteo · "
    "FAO-56 irrigation method · "
    "Built with Streamlit"
)
