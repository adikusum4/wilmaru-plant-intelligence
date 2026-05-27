"""
Sensor Monitor Agent
Baca machine_sensor_logs.csv → deteksi anomali → push alert.
Pakai Isolation Forest (scikit-learn, gratis).
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from core.alert_engine import push

FEATURES = ["temperature_c","vibration_mm_s","current_amp","pressure_bar"]

def load_latest(path: str = "data/simulated/machine_sensor_logs.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df.sort_values("timestamp")

def detect_anomalies(df: pd.DataFrame, contamination: float = 0.05) -> pd.DataFrame:
    """Run Isolation Forest per machine, add anomaly_score + is_anomaly columns."""
    results = []
    for mid, grp in df.groupby("machine_id"):
        X = grp[FEATURES].fillna(grp[FEATURES].median())
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        model = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
        grp = grp.copy()
        grp["anomaly_score"] = -model.fit(Xs).decision_function(Xs)   # higher = more anomalous
        grp["is_anomaly_ml"] = model.predict(Xs) == -1
        results.append(grp)
    return pd.concat(results).sort_values("timestamp")

def get_machine_summary(df: pd.DataFrame) -> list[dict]:
    """Latest health snapshot per machine."""
    latest = df.sort_values("timestamp").groupby("machine_id").last().reset_index()
    out = []
    for _, row in latest.iterrows():
        out.append({
            "machine_id":   row["machine_id"],
            "machine_name": row["machine_name"],
            "status":       row["status"],
            "health_score": round(row["health_score"], 1),
            "rul_hours":    int(row["rul_hours"]),
            "temperature":  round(row["temperature_c"], 1),
            "vibration":    round(row["vibration_mm_s"], 3),
            "is_anomaly":   bool(row.get("is_anomaly_ml", False)),
        })
    return sorted(out, key=lambda x: x["health_score"])

def run_sensor_monitor() -> str:
    df = load_latest()
    df = detect_anomalies(df)
    summary = get_machine_summary(df)

    critical = [m for m in summary if m["status"] in ("CRITICAL","FAILED")]
    warning  = [m for m in summary if m["status"] == "WARNING"]
    anomalies= [m for m in summary if m["is_anomaly"]]

    for m in critical:
        push("maintenance", "CRITICAL",
             f"Mesin {m['machine_id']} status {m['status']}",
             f"Health: {m['health_score']}% | RUL: {m['rul_hours']} jam | Vibration: {m['vibration']} mm/s")
    for m in warning:
        push("maintenance", "WARNING",
             f"Mesin {m['machine_id']} perlu perhatian",
             f"Health: {m['health_score']}% | Temp: {m['temperature']}°C")

    lines = ["=== SENSOR MONITOR REPORT ==="]
    lines.append(f"Mesin dipantau   : {len(summary)}")
    lines.append(f"Status CRITICAL  : {len(critical)}")
    lines.append(f"Status WARNING   : {len(warning)}")
    lines.append(f"Anomali ML       : {len(anomalies)}\n")
    lines.append("--- Detail per mesin (terurut health score) ---")
    for m in summary:
        flag = " ⚠️ ANOMALI" if m["is_anomaly"] else ""
        lines.append(f"  {m['machine_id']:22s} | Health: {m['health_score']:5.1f}% | RUL: {m['rul_hours']:5d}h | {m['status']}{flag}")
    return "\n".join(lines)
