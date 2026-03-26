"""
generate_alerts.py
==================
Fetches real seasonal weather from Open-Meteo, loads model_artefacts.pkl,
runs XGBoost predictions for all 176 district-crop-season combinations,
computes yield anomalies, and writes predictions.json for the dashboard.

Usage:
    python generate_alerts.py

Output:
    predictions.json  (same folder — the dashboard reads this)

Run this once per season or whenever you want fresh predictions.
Requires: model_artefacts.pkl and weather_cache.json in the same folder.
"""

import json
import time
import pickle
import datetime
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────

DATA_PATH      = "merged_crop_enriched_features_del.xlsx"
ARTEFACTS_PATH = "model_artefacts.pkl"
CACHE_PATH     = "weather_cache.json"
OUTPUT_PATH    = "predictions.json"

YIELD_COL = "Yield (Tonne or Bales/Hectare)"

ALERT_THRESHOLD   = -20.0   # % — triggers alert
CRITICAL_THRESHOLD = -30.0  # % — critical alert

VALID_CROPS = [
    "Rice", "Maize", "Wheat", "Jute", "Groundnut", "Rapeseed &Mustard",
    "Masoor", "Moong(Green Gram)", "Urad", "Sugarcane", "Arhar/Tur",
    "Cotton(lint)", "Mesta", "Peas & beans (Pulses)", "Sesamum",
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

SEASON_WINDOWS = {
    "Kharif":     ("06-01", "09-30", False),
    "Rabi":       ("10-15", "02-28", True),
    "Autumn":     ("08-01", "11-30", False),
    "Summer":     ("03-01", "06-30", False),
    "Winter":     ("11-01", "02-28", True),
    "Whole Year": ("01-01", "12-31", False),
}

WEATHER_FEATURES = [
    "weather_temp_mean", "weather_rain_total", "weather_rain_days",
    "weather_et0_total", "weather_wind_mean", "weather_solarrad_total",
]

PEST_MAP = {"Low": 0, "Medium": 1, "High": 2}


# ── HELPERS ───────────────────────────────────────────────────────────────────

def season_date_range(year_start: int, season: str):
    import calendar
    win = SEASON_WINDOWS[season]
    start_md, end_md, crosses = win
    start = f"{year_start}-{start_md}"
    end_year = year_start + 1 if crosses else year_start
    if end_md == "02-28":
        last = calendar.monthrange(end_year, 2)[1]
        end_md = f"02-{last}"
    return start, f"{end_year}-{end_md}"


def fetch_weather(lat: float, lon: float, start: str, end: str,
                  cache: dict, cache_key: str) -> dict:
    """Fetch from cache or Open-Meteo API."""
    if cache_key in cache:
        return cache[cache_key]

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start, "end_date": end,
        "daily": ",".join([
            "temperature_2m_mean", "precipitation_sum",
            "et0_fao_evapotranspiration", "windspeed_10m_max",
            "shortwave_radiation_sum",
        ]),
        "timezone": "Asia/Kolkata",
    }
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    daily = resp.json()["daily"]

    def smean(lst): v = [x for x in lst if x]; return float(np.mean(v)) if v else np.nan
    def ssum(lst):  v = [x for x in lst if x]; return float(np.sum(v)) if v else np.nan

    rain = daily.get("precipitation_sum", [])
    result = {
        "weather_temp_mean":      smean(daily.get("temperature_2m_mean", [])),
        "weather_rain_total":     ssum(rain),
        "weather_rain_days":      sum(1 for r in rain if r and r > 1.0),
        "weather_et0_total":      ssum(daily.get("et0_fao_evapotranspiration", [])),
        "weather_wind_mean":      smean(daily.get("windspeed_10m_max", [])),
        "weather_solarrad_total": ssum(daily.get("shortwave_radiation_sum", [])),
    }
    cache[cache_key] = result
    time.sleep(0.35)   # polite rate limiting
    return result


def get_season_weather(district: str, season: str, cache: dict) -> dict:
    """
    Get weather for the most recently completed instance of this season.
    For a season currently in progress, uses the most recent full year available.
    Falls back to 5-year climatology if the current year isn't available yet.
    """
    coords = DISTRICT_COORDS[district]
    today  = datetime.date.today()

    # Try last 3 years in reverse, return first successful fetch
    for years_back in range(0, 4):
        candidate_year = today.year - years_back
        start, end = season_date_range(candidate_year, season)
        season_end_date = datetime.date.fromisoformat(end)

        # Only use this year if the season has ended
        if season_end_date >= today and years_back == 0:
            continue   # Season not complete yet — try previous year

        cache_key = f"{district}|{candidate_year} - {candidate_year+1}|{season}"
        # Also check crop-year style key used by training cache
        alt_key = f"{district}|{candidate_year} - {candidate_year+1}|{season}"

        try:
            wx = fetch_weather(coords[0], coords[1], start, end, cache, alt_key)
            if not any(np.isnan(v) for v in wx.values()):
                return wx, candidate_year
        except Exception:
            continue

    # Fallback: 5-year climatology
    print(f"    Using climatology for {district} | {season}")
    records = []
    for y in range(today.year - 6, today.year - 1):
        try:
            s, e = season_date_range(y, season)
            key = f"{district}|{y} - {y+1}|{season}"
            wx = fetch_weather(coords[0], coords[1], s, e, cache, key)
            records.append(wx)
        except Exception:
            pass
    if records:
        avg = {k: float(np.nanmean([r[k] for r in records])) for k in records[0]}
        return avg, today.year - 1
    raise RuntimeError(f"Cannot fetch weather for {district} {season}")


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("TRIPURA CROP SHORTAGE ALERT GENERATOR")
    print(f"Run date: {datetime.date.today()}")
    print("=" * 60 + "\n")

    # 1. Load model artefacts
    print("Loading model artefacts...")
    with open(ARTEFACTS_PATH, "rb") as f:
        art = pickle.load(f)
    model      = art["model"]
    feat_cols  = art["feat_cols"]
    scaler     = art["scaler"]
    crop_stats = art["crop_stats"]
    df_history = art["df_history"]
    print(f"  Model loaded. Feature count: {len(feat_cols)}\n")

    # 2. Load weather cache
    cache_file = Path(CACHE_PATH)
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    print(f"Weather cache: {len(cache)} entries loaded\n")

    # 3. Build prediction combos from historical data
    print("Building prediction combos from historical data...")
    df = pd.read_excel(DATA_PATH)
    df["Year"] = df["Crop_Year"].str.split(" - ").str[0].astype(int)

    combos = []
    for (dist, crop, season), grp in df.groupby(["District_Name", "Crop", "Season"]):
        if crop not in VALID_CROPS:
            continue
        if crop not in crop_stats.index:
            continue
        grp = grp.sort_values("Year")
        if len(grp) < 3:
            continue
        last3  = grp[YIELD_COL].values[-3:]
        last_r = grp.iloc[-1]
        combos.append({
            "district":     dist,
            "crop":         crop,
            "season":       season,
            "last3_yields": [float(y) for y in last3],
            "area_ha":      float(last_r["Area (Hectare)"]),
            "fertilizer":   float(last_r["Fertilizer_kg_per_ha"]),
            "pest":         str(last_r["Pest_Disease_Incidence"]),
            "normal_yield": float(grp[YIELD_COL].values[-5:].mean()),
        })
    print(f"  {len(combos)} valid combos to predict\n")

    # 4. Predict for each combo
    print("Running predictions...\n")
    results = []
    n = len(combos)

    for i, combo in enumerate(combos):
        dist   = combo["district"]
        crop   = combo["crop"]
        season = combo["season"]

        print(f"  [{i+1:3d}/{n}] {dist:15s} | {crop:25s} | {season}", end="  ")

        # Get weather
        try:
            wx, wx_year = get_season_weather(dist, season, cache)
        except Exception as e:
            print(f"SKIP (weather error: {e})")
            continue

        # Yield lags
        last3       = combo["last3_yields"]
        yield_lag1  = last3[-1]
        yield_roll3 = float(np.mean(last3))
        yield_trend = float(np.polyfit(range(3), last3, 1)[0])
        normal      = combo["normal_yield"]

        # Normalise
        mu  = crop_stats.loc[crop, "crop_mean"]
        std = crop_stats.loc[crop, "crop_std"]

        row = {
            "District_Name":          dist,
            "Crop":                   crop,
            "Area (Hectare)":         combo["area_ha"],
            "Fertilizer_kg_per_ha":   combo["fertilizer"],
            "Pest_Disease_Incidence": PEST_MAP.get(combo["pest"], 1),
            "Yield_Lag1":             (yield_lag1  - mu) / std,
            "Yield_Roll3":            (yield_roll3 - mu) / std,
            "Yield_Trend":             yield_trend       / std,
            **{k: wx[k] for k in WEATHER_FEATURES},
        }

        row_df = pd.get_dummies(pd.DataFrame([row]), drop_first=True)
        row_df = row_df.reindex(columns=feat_cols, fill_value=0)
        row_sc = scaler.transform(row_df)

        norm_pred   = model.predict(row_sc)[0]
        pred_yield  = norm_pred * std + mu
        anomaly_pct = (pred_yield - normal) / normal * 100

        status = ("critical" if anomaly_pct <= CRITICAL_THRESHOLD
                  else "watch"    if anomaly_pct <= ALERT_THRESHOLD
                  else "normal")

        print(f"→ {pred_yield:.2f} t/ha  anomaly: {anomaly_pct:+.1f}%  [{status.upper()}]")

        results.append({
            "district":      dist,
            "crop":          crop,
            "season":        season,
            "predicted":     round(float(pred_yield), 3),
            "normal":        round(float(normal), 3),
            "anomaly":       round(float(anomaly_pct), 1),
            "status":        status,
            "weather_year":  wx_year,
            "weather": {
                "rain_total":    round(wx["weather_rain_total"], 1),
                "rain_days":     int(wx["weather_rain_days"]),
                "temp_mean":     round(wx["weather_temp_mean"], 1),
                "et0_total":     round(wx["weather_et0_total"], 1),
                "wind_mean":     round(wx["weather_wind_mean"], 1),
                "solarrad_total": round(wx["weather_solarrad_total"], 1),
            },
        })

    # 5. Save cache (new entries added)
    cache_file.write_text(json.dumps(cache, indent=2))
    print(f"\nCache updated: {len(cache)} entries")

    # 6. Summary
    critical = [r for r in results if r["status"] == "critical"]
    watch    = [r for r in results if r["status"] == "watch"]
    normal   = [r for r in results if r["status"] == "normal"]
    flagged_districts = len(set(r["district"] for r in results if r["status"] != "normal"))

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Total predictions : {len(results)}")
    print(f"  Critical alerts   : {len(critical)}  (anomaly ≤ {CRITICAL_THRESHOLD}%)")
    print(f"  Watch alerts      : {len(watch)}   (anomaly {ALERT_THRESHOLD}% to {CRITICAL_THRESHOLD}%)")
    print(f"  Normal            : {len(normal)}")
    print(f"  Districts flagged : {flagged_districts} of 8")

    if critical:
        print(f"\n  TOP CRITICAL ALERTS:")
        for r in sorted(critical, key=lambda x: x["anomaly"])[:5]:
            print(f"    {r['district']:15s} | {r['crop']:20s} | {r['season']:10s} → {r['anomaly']:+.1f}%")

    # 7. Write output JSON
    output = {
        "generated_at":   datetime.datetime.now().isoformat(),
        "run_date":       str(datetime.date.today()),
        "model_version":  "XGBoost · seasonal weather features · 33 features",
        "alert_threshold": ALERT_THRESHOLD,
        "critical_threshold": CRITICAL_THRESHOLD,
        "summary": {
            "total":            len(results),
            "critical":         len(critical),
            "watch":            len(watch),
            "normal":           len(normal),
            "districts_flagged": flagged_districts,
        },
        "predictions": results,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ predictions.json written ({len(results)} rows)")
    print(f"   Open alert_dashboard.html in your browser to view results.")


if __name__ == "__main__":
    main()
