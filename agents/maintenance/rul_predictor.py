"""
RUL Predictor Agent — Remaining Useful Life
Pakai Random Forest Regressor ditraining dari sensor history.
Dataset: machine_sensor_logs.csv (data simulasi kita sendiri)
Referensi konsep: NASA CMAPSS dataset standard industri.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

FEATURES   = ["temperature_c","vibration_mm_s","rpm","current_amp","pressure_bar","health_score"]
TARGET     = "rul_hours"

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling stats per machine — penting untuk RUL prediction."""
    df = df.sort_values(["machine_id","timestamp"]).copy()
    for col in ["temperature_c","vibration_mm_s","current_amp"]:
        df[f"{col}_roll6_mean"] = df.groupby("machine_id")[col].transform(
            lambda x: x.rolling(6, min_periods=1).mean())
        df[f"{col}_roll6_std"] = df.groupby("machine_id")[col].transform(
            lambda x: x.rolling(6, min_periods=1).std().fillna(0))
    df["vib_temp_ratio"] = df["vibration_mm_s"] / (df["temperature_c"] + 1)
    return df

def train_rul_model(df: pd.DataFrame):
    df = engineer_features(df)
    feat_cols = FEATURES + [c for c in df.columns if "_roll6_" in c or "ratio" in c]
    valid = df[df[TARGET] < 9000]   # exclude machines that won't fail (rul=9999)
    X = valid[feat_cols].fillna(0)
    y = valid[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    model = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.08, random_state=42)
    model.fit(X_train_s, y_train)
    mae = mean_absolute_error(y_test, model.predict(X_test_s))
    return model, scaler, feat_cols, mae

def predict_rul(df: pd.DataFrame) -> list[dict]:
    """Predict RUL for every machine's latest reading."""
    df_feat = engineer_features(df)
    model, scaler, feat_cols, mae = train_rul_model(df_feat)

    latest = df_feat.groupby("machine_id").last().reset_index()
    X_latest = latest[feat_cols].fillna(0)
    preds = model.predict(scaler.transform(X_latest))

    results = []
    for i, (_, row) in enumerate(latest.iterrows()):
        rul_pred = max(0, int(preds[i]))
        urgency = "URGENT (<48h)" if rul_pred < 48 else \
                  ("SOON (48-168h)" if rul_pred < 168 else \
                  ("MONITOR (1-4 weeks)" if rul_pred < 672 else "HEALTHY"))
        results.append({
            "machine_id":   row["machine_id"],
            "machine_name": row["machine_name"],
            "rul_predicted_hours": rul_pred,
            "rul_days":     round(rul_pred/24, 1),
            "urgency":      urgency,
            "health_score": round(row["health_score"], 1),
            "confidence":   f"±{int(mae/2)}h MAE",
        })
    return sorted(results, key=lambda x: x["rul_predicted_hours"])

def run_rul_analysis() -> str:
    df = pd.read_csv("data/simulated/machine_sensor_logs.csv", parse_dates=["timestamp"])
    predictions = predict_rul(df)

    lines = ["=== RUL PREDICTOR REPORT ==="]
    lines.append("(Remaining Useful Life — estimasi sisa umur mesin)\n")
    for p in predictions:
        days_str = f"{p['rul_days']}d" if p["rul_predicted_hours"] < 9000 else "N/A (healthy)"
        lines.append(f"  {p['machine_id']:22s} | RUL: {days_str:12s} | {p['urgency']:25s} | Health: {p['health_score']}%")
    urgent = [p for p in predictions if "URGENT" in p["urgency"] or "SOON" in p["urgency"]]
    if urgent:
        lines.append(f"\n⚠️  {len(urgent)} mesin butuh maintenance dalam 7 hari:")
        for p in urgent:
            lines.append(f"   → {p['machine_id']} : {p['urgency']}")
    return "\n".join(lines)
