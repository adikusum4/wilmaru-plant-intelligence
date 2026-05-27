"""
Quality Anomaly Detection Agent
Pakai Z-score + Isolation Forest untuk deteksi batch yang outlier.
Parameter kualitas CPO: FFA%, moisture%, color Lovibond.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from core.alert_engine import push

QUALITY_FEATURES = ["ffa_pct","moisture_pct","color_lovibond_r","reject_rate_pct"]
SPEC_LIMITS = {
    "ffa_pct":        {"max": 0.20, "name": "FFA"},
    "moisture_pct":   {"max": 0.15, "name": "Moisture"},
    "color_lovibond_r": {"max": 5.0, "name": "Color"},
    "reject_rate_pct":{"max": 4.0,  "name": "Reject Rate"},
}

def detect_quality_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Rule-based: flag berdasarkan spec limits
    df["spec_violations"] = 0
    for col, spec in SPEC_LIMITS.items():
        if col in df.columns:
            df["spec_violations"] += (df[col] > spec["max"]).astype(int)
    df["rule_fail"] = df["spec_violations"] > 0

    # ML: Isolation Forest across all lines
    X = df[QUALITY_FEATURES].fillna(df[QUALITY_FEATURES].median())
    sc = StandardScaler()
    model = IsolationForest(contamination=0.08, random_state=42)
    df["ml_anomaly"] = model.fit_predict(sc.fit_transform(X)) == -1
    df["anomaly_combined"] = df["rule_fail"] | df["ml_anomaly"]
    return df

def run_anomaly_detection() -> str:
    df = pd.read_csv("data/simulated/production_quality.csv")
    df = detect_quality_anomalies(df)

    recent = df.tail(30)
    anomalies = recent[recent["anomaly_combined"]]

    for _, row in anomalies.iterrows():
        push("quality", "CRITICAL" if row["spec_violations"] >= 2 else "WARNING",
             f"Quality anomaly: {row['production_line']} ({row['date']})",
             f"FFA: {row['ffa_pct']:.3f}% | Moisture: {row['moisture_pct']:.3f}% | Reject: {row['reject_rate_pct']:.2f}%")

    lines = ["=== QUALITY ANOMALY DETECTION REPORT ===\n"]
    lines.append(f"Periode analisis : 30 hari terakhir")
    lines.append(f"Total batch      : {len(recent)}")
    lines.append(f"Anomali terdeteksi: {len(anomalies)} ({100*len(anomalies)/len(recent):.1f}%)\n")
    if len(anomalies):
        lines.append(f"{'Tanggal':12s} {'Line':15s} {'FFA%':7s} {'Moist%':8s} {'Reject%':9s} {'Status'}")
        lines.append("-"*62)
        for _, row in anomalies.iterrows():
            lines.append(f"{row['date']:12s} {row['production_line']:15s} {row['ffa_pct']:.3f}   {row['moisture_pct']:.3f}    {row['reject_rate_pct']:.2f}     {row['quality_status']}")
    return "\n".join(lines)
