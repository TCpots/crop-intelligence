"""
Crop Recommender — Updated Backend
====================================
Uses model_artefacts.pkl from crop_yield_with_weather.py (33 features).
Serves yield predictions AND full ranked recommendations to the HTML frontend.

Run:
    pip install flask flask-cors pandas xgboost scikit-learn openpyxl
    python backend.py

Then open crop_recommender.html in your browser.
"""

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────

ARTEFACTS_FILE = "model_artefacts.pkl"
WEATHER_FILE   = "merged_crop_enriched_features_del.xlsx"

YIELD_COL = "Yield (Tonne or Bales/Hectare)"

WEATHER_FEATURES = [
    "weather_temp_mean", "weather_rain_total", "weather_rain_days",
    "weather_et0_total", "weather_wind_mean", "weather_solarrad_total",
]

PEST_MAP = {"Low": 0, "Medium": 1, "High": 2}

# Importance-derived suitability weights (from final model run)
# Only features the user can provide that differentiate between crops
# Pest and Area excluded — they don't vary per crop in suitability scoring
# Re-normalised to sum to 1.0
SUITABILITY_WEIGHTS = {
    "weather_rain_days":    0.189,
    "Fertilizer_kg_per_ha": 0.140,
    "weather_et0_total":    0.119,
    "weather_temp_mean":    0.098,
    "weather_rain_total":   0.082,
    "weather_solarrad_total": 0.071,
    "weather_wind_mean":    0.054,
}
# Normalise so they sum to 1
_wsum = sum(SUITABILITY_WEIGHTS.values())
SUITABILITY_WEIGHTS = {k: v / _wsum for k, v in SUITABILITY_WEIGHTS.items()}

# ── LOAD ──────────────────────────────────────────────────────────────────────

print("\nLoading model artefacts...")
if not Path(ARTEFACTS_FILE).exists():
    print(f"\nERROR: {ARTEFACTS_FILE} not found.")
    print("Run crop_yield_with_weather.py first to generate it.")
    raise SystemExit(1)

with open(ARTEFACTS_FILE, "rb") as f:
    art = pickle.load(f)

model      = art["model"]
feat_cols  = art["feat_cols"]
scaler     = art["scaler"]
crop_stats = art["crop_stats"]
df_history = art["df_history"]
valid_crops = crop_stats.index.tolist()

print(f"  Model loaded — {len(valid_crops)} crops, {len(feat_cols)} features")
print(f"  Valid crops: {valid_crops}\n")

# ── LOAD FULL DATASET (for stats/correlations — df_history only has weather+yield) ──
print("Loading full dataset for stats endpoints...")
_full_df = None
if Path(WEATHER_FILE).exists():
    _full_df = pd.read_excel(WEATHER_FILE)
    # Merge weather from df_history (which has real per-season weather)
    # df_history Year is 0-based offset from 2004
    _dh = df_history.copy()
    _dh["Crop_Year"] = _dh["Year"].apply(lambda y: f"{y+2004} - {y+2005}")
    _weather_cols = [c for c in _dh.columns if c.startswith("weather_")]
    _full_df = _full_df.merge(
        _dh[["District_Name", "Crop_Year", "Crop"] + _weather_cols].drop_duplicates(
            subset=["District_Name", "Crop_Year", "Crop"]
        ),
        on=["District_Name", "Crop_Year", "Crop"], how="left"
    )
    # Encode Pest for numeric correlation
    _pest_map = {"Low": 0, "Medium": 1, "High": 2}
    if "Pest_Disease_Incidence" in _full_df.columns:
        _full_df["Pest_Disease_Incidence"] = _full_df["Pest_Disease_Incidence"].map(_pest_map).fillna(1)
    print(f"  Full dataset loaded: {len(_full_df)} rows, cols: {list(_full_df.columns)}\n")
else:
    print(f"  WARNING: {WEATHER_FILE} not found — stats endpoints will use df_history only\n")

# ── BUILD SUITABILITY PROFILES ────────────────────────────────────────────────
# For each crop: historical mean/std of suitability features + season/soil/irrigation

print("Building crop suitability profiles from historical data...")
if _full_df is None:
    print(f"  WARNING: {WEATHER_FILE} not found — soil/irrigation profiles unavailable")
    PROFILES = {}
else:
    df_wx = _full_df  # already loaded and weather-merged above

    PROFILES = {}
    for crop, grp in df_wx.groupby("Crop"):
        p = {}
        for feat in ["Fertilizer_kg_per_ha"]:
            if feat in grp.columns:
                p[feat] = {"mean": float(grp[feat].mean()), "std": float(grp[feat].std() + 1e-3)}
        for cat in ["Soil_Type", "Irrigation_Type", "Season"]:
            if cat in grp.columns:
                p[cat] = grp[cat].value_counts(normalize=True).to_dict()
        p["avg_yield"] = float(grp[YIELD_COL].mean())
        # Add weather profiles
        for feat in WEATHER_FEATURES:
            if feat in grp.columns:
                vals = grp[feat].dropna()
                if len(vals) > 0:
                    p[feat] = {"mean": float(vals.mean()), "std": float(vals.std() + 1e-3)}
        PROFILES[crop] = p

print(f"  Profiles built for {len(PROFILES)} crops\n")

# ── HELPERS ───────────────────────────────────────────────────────────────────

def gaussian(val, mean, std):
    return float(np.exp(-0.5 * ((val - mean) / std) ** 2))


def predict_yield_for_crop(crop, district, user_inputs):
    """
    Run XGBoost prediction for a specific crop.
    Returns (predicted_yield, source) where source is 'model' or 'hist_avg'.

    Real feat_cols (from get_dummies on training data, drop_first=True):
      Categorical: Crop_*, District_Name_*, Irrigation_Type_Drip,
                   Irrigation_Type_Rainfed, Soil_Type_Red Laterite
      Numeric:     Area (Hectare), Fertilizer_kg_per_ha,
                   Pest_Disease_Incidence (0/1/2),
                   Yield_Lag1, Yield_Roll3, Yield_Trend,
                   weather_temp_mean, weather_rain_total, weather_rain_days,
                   weather_et0_total, weather_solarrad_total
      NOTE: Season is dropped before training — NOT a model feature.
            Soil baseline is Alluvial (drop_first), Red Laterite is encoded.
            Irrigation baseline is Canal (drop_first), Drip/Rainfed encoded.
    """
    if crop not in valid_crops:
        avg = PROFILES.get(crop, {}).get("avg_yield", 1.0)
        return avg, "hist_avg"

    mu  = crop_stats.loc[crop, "crop_mean"]
    std = crop_stats.loc[crop, "crop_std"]

    row = {
        "District_Name":          district,
        "Crop":                   crop,
        "Soil_Type":              user_inputs.get("Soil_Type", "Alluvial"),
        "Irrigation_Type":        user_inputs.get("Irrigation_Type", "Canal"),
        "Area (Hectare)":         user_inputs.get("Area (Hectare)", 500),
        "Fertilizer_kg_per_ha":   user_inputs.get("Fertilizer_kg_per_ha", 70),
        "Pest_Disease_Incidence": PEST_MAP.get(user_inputs.get("Pest_Disease_Incidence", "Low"), 0),
        "Yield_Lag1":  0.0,
        "Yield_Roll3": 0.0,
        "Yield_Trend": 0.0,
        **{feat: user_inputs.get(feat, 0.0) for feat in WEATHER_FEATURES},
    }

    X = pd.get_dummies(pd.DataFrame([row]), drop_first=True)
    X = X.reindex(columns=feat_cols, fill_value=0)
    X_sc = scaler.transform(X)

    norm_pred  = model.predict(X_sc)[0]
    pred_yield = float(norm_pred * std + mu)
    return pred_yield, "model"


def compute_suitability(crop, user_inputs):
    """
    Gaussian profile match weighted by model feature importance.
    Returns a 0–1 suitability score.
    """
    p = PROFILES.get(crop, {})
    num, wsum = 0.0, 0.0

    for feat, wt in SUITABILITY_WEIGHTS.items():
        val = user_inputs.get(feat)
        if val is not None and feat in p:
            num  += wt * gaussian(val, p[feat]["mean"], p[feat]["std"])
            wsum += wt

    num = num / wsum if wsum > 0 else 0.0

    # Season fit (not in SUITABILITY_WEIGHTS — categorical overlay)
    season     = user_inputs.get("Season", "Kharif")
    season_fit = p.get("Season", {}).get(season, 0.0)

    # Combined: 70% numeric weather+fert match, 30% season fit
    combined = 0.70 * num + 0.30 * season_fit
    return combined, season_fit


# ── FLASK ─────────────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "crops": len(valid_crops), "features": len(feat_cols)})


@app.route("/crop_trends", methods=["GET"])
def crop_trends():
    """
    Returns real year-by-year median yield for each crop from df_history.
    Query params:
      crop (str): specific crop, or omit for all crops
    Response:
    {
      "years": [2004, 2005, ...],
      "crops": {
        "Rice": [1.82, 1.85, ...],
        "Jute": [8.1, 8.2, ...],
        ...
      },
      "overall": [2.58, 2.63, ...],
      "decade": {
        "Rice": {"early": 1.95, "recent": 2.24},
        ...
      }
    }
    """
    crop_filter = request.args.get("crop", None)

    # df_history Year is 0-based offset from min year (2004)
    # Reconstruct actual years
    min_year = 2004
    df = df_history.copy()
    df["actual_year"] = df["Year"] + min_year

    years_available = sorted(df["actual_year"].unique().tolist())

    result_crops = {}
    decade_result = {}

    crops_to_process = [crop_filter] if crop_filter else df["Crop"].unique().tolist()

    for crop in crops_to_process:
        crop_df = df[df["Crop"] == crop]
        if crop_df.empty:
            continue

        # Year-by-year median yield across all districts
        yearly = (
            crop_df.groupby("actual_year")[YIELD_COL]
            .median()
            .reindex(years_available)
        )
        # Forward-fill missing years with interpolation
        yearly = yearly.interpolate(method="linear").bfill().ffill()
        result_crops[crop] = [round(float(v), 3) for v in yearly.values]

        # Decade comparison
        early_mask  = (crop_df["actual_year"] >= 2004) & (crop_df["actual_year"] <= 2013)
        recent_mask = (crop_df["actual_year"] >= 2014) & (crop_df["actual_year"] <= 2023)
        early_avg  = float(crop_df.loc[early_mask,  YIELD_COL].median()) if early_mask.any()  else 0
        recent_avg = float(crop_df.loc[recent_mask, YIELD_COL].median()) if recent_mask.any() else 0
        decade_result[crop] = {
            "early":  round(early_avg,  3),
            "recent": round(recent_avg, 3),
            "change_pct": round((recent_avg - early_avg) / early_avg * 100, 1) if early_avg > 0 else 0,
        }

    # Overall median across all crops per year
    overall_yearly = (
        df.groupby("actual_year")[YIELD_COL]
        .median()
        .reindex(years_available)
        .interpolate(method="linear")
        .bfill()
    )

    return jsonify({
        "years":   years_available,
        "crops":   result_crops,
        "overall": [round(float(v), 3) for v in overall_yearly.values],
        "decade":  decade_result,
    })


@app.route("/predict", methods=["POST"])
def predict():
    """Single crop prediction. Used by the what-if panel."""
    data     = request.get_json()
    crop     = data.get("crop", "")
    district = data.get("district", "Dhalai")

    pred, source = predict_yield_for_crop(crop, district, data)

    # Compute anomaly vs crop historical normal
    normal = crop_stats.loc[crop, "crop_mean"] if crop in valid_crops else pred
    anomaly = round((pred - normal) / normal * 100, 1) if normal > 0 else 0.0

    return jsonify({
        "yield":   round(pred, 3),
        "normal":  round(normal, 3),
        "anomaly": anomaly,
        "source":  source,
    })


@app.route("/recommend", methods=["POST"])
def recommend():
    """
    Full recommendation run across all crops.
    Ranks by suitability score, returns top N with predicted yield + anomaly.

    Expects JSON:
    {
      "district":                "Dhalai",
      "Season":                  "Kharif",
      "Soil_Type":               "Alluvial",
      "Irrigation_Type":         "Rainfed",
      "Fertilizer_kg_per_ha":    70,
      "Area (Hectare)":          500,
      "Pest_Disease_Incidence":  "Low",
      "weather_rain_days":       170,
      "weather_rain_total":      1800,
      "weather_temp_mean":       24.5,
      "weather_et0_total":       1240,
      "weather_wind_mean":       11.2,
      "weather_solarrad_total":  5800,
      "top_n":                   7
    }
    """
    data     = request.get_json()
    district = data.get("district", "Dhalai")
    top_n    = int(data.get("top_n", 7))

    # Score all crops
    results = []
    all_crops = list(PROFILES.keys())

    for crop in all_crops:
        suit, season_fit = compute_suitability(crop, data)
        pred, source     = predict_yield_for_crop(crop, district, data)

        # Normal yield for anomaly
        if crop in valid_crops:
            normal = float(crop_stats.loc[crop, "crop_mean"])
        else:
            normal = PROFILES.get(crop, {}).get("avg_yield", pred)

        anomaly = round((pred - normal) / normal * 100, 1) if normal > 0 else 0.0

        results.append({
            "crop":        crop,
            "suit_score":  round(suit, 4),
            "season_fit":  round(season_fit, 3),
            "predicted":   round(pred, 3),
            "normal":      round(normal, 3),
            "anomaly":     anomaly,
            "source":      source,
        })

    # Sort by suitability descending
    results.sort(key=lambda x: x["suit_score"], reverse=True)

    # Normalise suitability to percentage
    max_s = results[0]["suit_score"] if results else 1.0
    for r in results:
        r["suit_pct"] = round(r["suit_score"] / max_s * 100) if max_s > 0 else 0

    return jsonify({
        "district":    district,
        "season":      data.get("Season", ""),
        "results":     results[:top_n],
        "weights_used": SUITABILITY_WEIGHTS,
    })


@app.route("/valid_crops", methods=["GET"])
def get_valid_crops():
    return jsonify({"valid_crops": valid_crops})


@app.route("/model_info", methods=["GET"])
def model_info():
    """
    Returns feature importances from the XGBoost model and
    Pearson correlations of numeric features with yield in df_history.
    """
    # Feature importances
    raw_imp = model.get_booster().get_score(importance_type="gain")
    total = sum(raw_imp.values()) or 1.0
    feat_imps = {k: round(v / total, 6) for k, v in raw_imp.items()}
    # Sort descending
    feat_imps_sorted = dict(sorted(feat_imps.items(), key=lambda x: -x[1]))

    # Pearson correlations with yield — use full dataset so Season/Soil/Irrigation/Pest/Fert are available
    corr_df = _full_df.copy() if _full_df is not None else df_history.copy()
    numeric_cols = [
        "Fertilizer_kg_per_ha", "Area (Hectare)",
        "weather_rain_total", "weather_rain_days",
        "weather_temp_mean", "weather_et0_total",
        "weather_wind_mean", "weather_solarrad_total",
        "Pest_Disease_Incidence",
    ]
    corr_result = {}
    for col in numeric_cols:
        if col in corr_df.columns:
            sub = corr_df[[col, YIELD_COL]].dropna()
            if len(sub) > 10:
                r = sub.corr().iloc[0, 1]
                corr_result[col] = round(float(r), 4)

    # Categorical correlations — encode then correlate
    for cat, label in [("Irrigation_Type", "Irrigation_Type"), ("Soil_Type", "Soil_Type"), ("Season", "Season")]:
        if cat in corr_df.columns:
            encoded = corr_df[cat].astype("category").cat.codes
            sub = pd.concat([encoded, corr_df[YIELD_COL]], axis=1).dropna()
            if len(sub) > 10:
                r = sub.corr().iloc[0, 1]
                corr_result[label] = round(float(r), 4)

    return jsonify({
        "feat_importances": feat_imps_sorted,
        "correlations": corr_result,
        "n_features": len(feat_cols),
    })


@app.route("/stats", methods=["GET"])
def stats():
    """
    Computes EDA statistics from df_history for the dashboard.
    All chart data that was previously hardcoded in the frontend.
    """
    df = _full_df.copy() if _full_df is not None else df_history.copy()

    # ── crop frequency (record count per crop) ──
    crop_freq = df["Crop"].value_counts().to_dict()

    # ── crop median yield ──
    crop_yield_med = df.groupby("Crop")[YIELD_COL].median().to_dict()

    # ── season distribution (% of records) ──
    season_counts = (df["Season"].value_counts(normalize=True) * 100).round(2).to_dict()
    season_yields = df.groupby("Season")[YIELD_COL].median().to_dict()

    # ── soil yield ──
    soil_yield = df.groupby("Soil_Type")[YIELD_COL].median().to_dict() if "Soil_Type" in df.columns else {}

    # ── irrigation yield ──
    irr_yield = df.groupby("Irrigation_Type")[YIELD_COL].median().to_dict() if "Irrigation_Type" in df.columns else {}

    # ── pest yield ──
    pest_map_inv = {0: "Low", 1: "Medium", 2: "High"}
    if "Pest_Disease_Incidence" in df.columns:
        pest_yield = (
            df.groupby("Pest_Disease_Incidence")[YIELD_COL]
            .median()
            .rename(index=lambda x: pest_map_inv.get(int(x), str(x)))
            .to_dict()
        )
    else:
        pest_yield = {}

    # ── fertilizer usage per crop ──
    fert_usage = {}
    if "Fertilizer_kg_per_ha" in df.columns:
        fert_usage = df.groupby("Crop")["Fertilizer_kg_per_ha"].median().to_dict()

    # ── binned weather vs yield ──
    def bin_yield(col, bins, labels):
        if col not in df.columns:
            return {"labels": labels, "yields": []}
        sub = df[[col, YIELD_COL]].dropna().copy()
        sub["bin"] = pd.cut(sub[col], bins=bins, labels=labels, right=False)
        result = sub.groupby("bin")[YIELD_COL].median()
        return {
            "labels": labels,
            "yields": [round(float(result.get(l, 0)), 3) for l in labels],
        }

    def scatter_col(x_col, max_pts=800):
        """Return raw {x, y} scatter points for a column vs yield, downsampled if large."""
        if x_col not in df.columns:
            return []
        pts = df[[x_col, YIELD_COL]].dropna()
        if len(pts) > max_pts:
            pts = pts.sample(max_pts, random_state=42)
        pts = pts.sort_values(x_col)
        return [{"x": round(float(r[x_col]), 3), "y": round(float(r[YIELD_COL]), 3)}
                for _, r in pts.iterrows()]

    rainfall_bins = bin_yield(
        "weather_rain_total",
        [0, 100, 150, 200, 250, 300, 400, 9999],
        ["0–100", "100–150", "150–200", "200–250", "250–300", "300–400", "400+"],
    )
    rain_scatter = scatter_col("weather_rain_total")
    temp_scatter = scatter_col("weather_temp_mean")
    et0_scatter  = scatter_col("weather_et0_total")
    fert_scatter = scatter_col("Fertilizer_kg_per_ha")

    # ── crop × season yield table ──
    crop_season = {}
    if "Season" in df.columns:
        for (crop, season), grp in df.groupby(["Crop", "Season"]):
            if crop not in crop_season:
                crop_season[crop] = {}
            crop_season[crop][season] = round(float(grp[YIELD_COL].median()), 3)

    # ── soil × irrigation matrix ──
    sxi = {}
    if "Soil_Type" in df.columns and "Irrigation_Type" in df.columns:
        for (s, ir), grp in df.groupby(["Soil_Type", "Irrigation_Type"]):
            if s not in sxi:
                sxi[s] = {}
            sxi[s][ir] = round(float(grp[YIELD_COL].median()), 3)

    # ── pest distribution per crop (%) — for stacked bar chart ──
    pest_crop_dist = {}
    if "Pest_Disease_Incidence" in df.columns and "Crop" in df.columns:
        # Pest is numeric 0/1/2 in _full_df
        pest_label_map = {0: "Low", 1: "Medium", 2: "High"}
        for crop, grp in df.groupby("Crop"):
            counts = grp["Pest_Disease_Incidence"].value_counts(normalize=True) * 100
            pest_crop_dist[crop] = {
                pest_label_map.get(int(k), str(k)): round(float(v), 1)
                for k, v in counts.items()
            }

    # ── summary stats for the overview strip ──
    n_records = int(len(df))
    n_crops = int(df["Crop"].nunique())
    n_seasons = int(df["Season"].nunique()) if "Season" in df.columns else 6
    n_districts = int(df["District_Name"].nunique()) if "District_Name" in df.columns else 8
    avg_yield = round(float(df[YIELD_COL].median()), 3)
    avg_rainfall = round(float(df["weather_rain_total"].median()), 1) if "weather_rain_total" in df.columns else None
    avg_temp = round(float(df["weather_temp_mean"].median()), 1) if "weather_temp_mean" in df.columns else None

    return jsonify({
        "summary": {
            "n_records": n_records,
            "n_crops": n_crops,
            "n_seasons": n_seasons,
            "n_districts": n_districts,
            "avg_yield": avg_yield,
            "avg_rainfall": avg_rainfall,
            "avg_temp": avg_temp,
        },
        "crop_freq": crop_freq,
        "crop_yield_med": {k: round(float(v), 3) for k, v in crop_yield_med.items()},
        "season_counts": {k: round(float(v), 2) for k, v in season_counts.items()},
        "season_yields": {k: round(float(v), 3) for k, v in season_yields.items()},
        "soil_yield": {k: round(float(v), 3) for k, v in soil_yield.items()},
        "irr_yield": {k: round(float(v), 3) for k, v in irr_yield.items()},
        "pest_yield": {k: round(float(v), 3) for k, v in pest_yield.items()},
        "pest_crop_dist": pest_crop_dist,
        "fert_usage": {k: round(float(v), 1) for k, v in fert_usage.items()},
        "rainfall_bins": rainfall_bins,
        "rain_scatter":  rain_scatter,
        "temp_scatter":  temp_scatter,
        "et0_scatter":   et0_scatter,
        "fert_scatter":  fert_scatter,
        "crop_season": crop_season,
        "soil_x_irr": sxi,
    })


@app.route("/stats/crop_scatter", methods=["GET"])
def crop_scatter():
    """
    Returns per-crop scatter points for rainfall vs yield and fertilizer vs yield.
    Used by the Conditional Yield Explorer to draw data-backed scatter + LOESS trendlines.
    Query param: crop (required)
    """
    crop = request.args.get("crop", "")
    if not crop:
        return jsonify({"error": "crop param required"}), 400

    src = _full_df if _full_df is not None else df_history
    sub = src[src["Crop"] == crop] if "Crop" in src.columns else src

    def scatter_pts(col):
        if col not in sub.columns:
            return []
        pts = sub[[col, YIELD_COL]].dropna()
        return [{"x": round(float(r[col]), 2), "y": round(float(r[YIELD_COL]), 3)}
                for _, r in pts.iterrows()]

    return jsonify({
        "crop": crop,
        "rain_scatter":  scatter_pts("weather_rain_total"),
        "fert_scatter":  scatter_pts("Fertilizer_kg_per_ha"),
        "temp_scatter":  scatter_pts("weather_temp_mean"),
        "et0_scatter":   scatter_pts("weather_et0_total"),
    })


@app.route("/profiles", methods=["GET"])
def get_profiles():
    """Expose profiles so the HTML can show crop-specific info."""
    safe = {}
    for crop, p in PROFILES.items():
        safe[crop] = {
            "avg_yield":      p.get("avg_yield", 0),
            "Season":         p.get("Season", {}),
            "Soil_Type":      p.get("Soil_Type", {}),
            "Irrigation_Type": p.get("Irrigation_Type", {}),
        }
    return jsonify(safe)


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Backend running at http://localhost:5000")
    print("Open crop_recommender.html in your browser.\n")
    app.run(port=5000, debug=False)
