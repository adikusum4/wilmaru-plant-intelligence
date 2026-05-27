"""
Root Cause Analysis Agent
Korelasi antara kondisi mesin (sensor) dan quality anomaly.
Cross-domain insight: sensor data → quality degradation.
"""
import pandas as pd
import numpy as np

def correlate_machine_quality(sensor_df: pd.DataFrame, quality_df: pd.DataFrame) -> list[dict]:
    """Cari korelasi: mesin mana yang paling berpengaruh ke quality anomaly."""
    sensor_df["date"] = pd.to_datetime(sensor_df["timestamp"]).dt.date.astype(str)
    daily_sensor = sensor_df.groupby(["date","machine_id"]).agg(
        avg_vibration=("vibration_mm_s","mean"),
        avg_temp=("temperature_c","mean"),
        avg_health=("health_score","mean"),
    ).reset_index()

    quality_df["date"] = quality_df["date"].astype(str)
    merged = quality_df.merge(daily_sensor, on="date", how="left")

    insights = []
    for mid in daily_sensor["machine_id"].unique():
        sub = merged[merged["machine_id"] == mid].dropna(subset=["avg_vibration","reject_rate_pct"])
        if len(sub) < 10: continue
        corr_vib  = sub["avg_vibration"].corr(sub["reject_rate_pct"])
        corr_temp = sub["avg_temp"].corr(sub["ffa_pct"])
        corr_health = sub["avg_health"].corr(sub["oee_pct"])
        insights.append({
            "machine_id": mid,
            "corr_vibration_to_reject": round(corr_vib, 3),
            "corr_temp_to_ffa":         round(corr_temp, 3),
            "corr_health_to_oee":       round(corr_health, 3),
            "impact_level": "HIGH" if abs(corr_vib) > 0.6 else ("MEDIUM" if abs(corr_vib) > 0.3 else "LOW"),
        })
    return sorted(insights, key=lambda x: -abs(x["corr_vibration_to_reject"]))

def run_root_cause() -> str:
    sensor_df  = pd.read_csv("data/simulated/machine_sensor_logs.csv")
    quality_df = pd.read_csv("data/simulated/production_quality.csv")
    insights   = correlate_machine_quality(sensor_df, quality_df)

    lines = ["=== ROOT CAUSE ANALYSIS REPORT ===\n"]
    lines.append("Korelasi kondisi mesin vs parameter kualitas produksi:\n")
    lines.append(f"{'Machine':22s} {'Vib→Reject':12s} {'Temp→FFA':10s} {'Health→OEE':12s} {'Impact'}")
    lines.append("-"*68)
    for ins in insights:
        lines.append(
            f"{ins['machine_id']:22s} "
            f"{ins['corr_vibration_to_reject']:+.3f}      "
            f"{ins['corr_temp_to_ffa']:+.3f}     "
            f"{ins['corr_health_to_oee']:+.3f}       "
            f"{ins['impact_level']}"
        )
    high = [i for i in insights if i["impact_level"] == "HIGH"]
    if high:
        lines.append(f"\n🔍 Temuan utama: {len(high)} mesin berkorelasi tinggi dengan penurunan kualitas.")
        for h in high:
            lines.append(f"   → {h['machine_id']}: vibration berkorelasi {h['corr_vibration_to_reject']:+.3f} dengan reject rate")
    return "\n".join(lines)
