"""
Demand Forecast Agent
Pakai Prophet (Meta, gratis) atau fallback ke simple exponential smoothing.
Forecast 30 hari ke depan per SKU.
"""
import pandas as pd
import numpy as np
from datetime import datetime

def _exp_smoothing(series: pd.Series, alpha: float = 0.3) -> list[float]:
    """Simple exponential smoothing — fallback jika Prophet tidak terinstall."""
    vals = list(series)
    if not vals: return []
    forecast = [vals[0]]
    for v in vals[1:]:
        forecast.append(alpha * v + (1 - alpha) * forecast[-1])
    last = forecast[-1]
    return [last * np.random.uniform(0.92, 1.08) for _ in range(30)]

def forecast_sku(df: pd.DataFrame, sku_id: str, horizon: int = 30) -> dict:
    sku_df = df[df["sku_id"] == sku_id].copy()
    sku_df["date"] = pd.to_datetime(sku_df["date"])
    daily = sku_df.groupby("date")["daily_demand"].mean().reset_index()
    daily.columns = ["ds","y"]
    daily = daily.sort_values("ds")

    try:
        from prophet import Prophet
        m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=False)
        m.fit(daily)
        future = m.make_future_dataframe(periods=horizon)
        fc = m.predict(future).tail(horizon)
        forecast_vals = fc["yhat"].clip(lower=0).tolist()
        method = "Prophet"
    except ImportError:
        forecast_vals = _exp_smoothing(daily["y"], alpha=0.25)
        method = "ExpSmoothing"

    avg_forecast = float(np.mean(forecast_vals))
    return {
        "sku_id": sku_id,
        "sku_name": sku_df["sku_name"].iloc[0] if len(sku_df) else sku_id,
        "unit": sku_df["unit"].iloc[0] if len(sku_df) else "",
        "avg_daily_forecast": round(avg_forecast, 2),
        "total_30d_forecast": round(avg_forecast * 30, 1),
        "method": method,
        "forecast_values": [round(v, 2) for v in forecast_vals],
    }

def run_demand_forecast() -> str:
    df = pd.read_csv("data/simulated/warehouse_inventory.csv")
    skus = df["sku_id"].unique()
    lines = ["=== DEMAND FORECAST REPORT (30 hari ke depan) ===\n"]
    for sku in skus:
        fc = forecast_sku(df, sku)
        lines.append(f"  {fc['sku_name']:28s} | Avg/hari: {fc['avg_daily_forecast']:8.2f} {fc['unit']} | Total 30d: {fc['total_30d_forecast']:10.1f} | [{fc['method']}]")
    return "\n".join(lines)
