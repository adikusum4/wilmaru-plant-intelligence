"""Yield Forecast Agent — prediksi output produksi minggu depan per line."""
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

def forecast_yield(df: pd.DataFrame, line: str, horizon_days: int = 7) -> dict:
    sub = df[df["production_line"] == line].copy()
    sub["dayofweek"] = pd.to_datetime(sub["date"]).dt.dayofweek
    sub["trend"] = range(len(sub))
    features = ["oee_pct","runtime_hours","reject_rate_pct","dayofweek","trend"]
    X = sub[features].fillna(sub[features].median())
    y = sub["actual_output_mt"]
    sc = StandardScaler()
    model = Ridge(alpha=1.0)
    model.fit(sc.fit_transform(X), y)
    last = X.iloc[-1].copy()
    preds = []
    for i in range(horizon_days):
        last["trend"] += 1
        last["dayofweek"] = (last["dayofweek"] + 1) % 7
        pred = float(model.predict(sc.transform([last]))[0])
        preds.append(max(0, pred + np.random.normal(0, pred*0.03)))
    return {
        "line": line,
        "avg_forecast_mt_day": round(np.mean(preds), 1),
        "total_7d_forecast_mt": round(sum(preds), 1),
        "daily_forecasts": [round(p, 1) for p in preds],
    }

def run_yield_forecast() -> str:
    df = pd.read_csv("data/simulated/production_quality.csv")
    lines_list = df["production_line"].unique()
    lines = ["=== YIELD FORECAST REPORT (7 hari ke depan) ===\n"]
    total = 0
    for line in lines_list:
        fc = forecast_yield(df, line)
        total += fc["total_7d_forecast_mt"]
        lines.append(f"  {fc['line']:18s} | Avg/hari: {fc['avg_forecast_mt_day']:6.1f} MT | Total 7d: {fc['total_7d_forecast_mt']:7.1f} MT")
    lines.append(f"\n  Total semua line (7d): {total:.1f} MT")
    return "\n".join(lines)
