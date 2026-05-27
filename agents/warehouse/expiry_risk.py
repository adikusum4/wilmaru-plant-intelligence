"""
Expiry Risk Agent — penting untuk FMCG dan chemical.
Deteksi bahan yang akan expire sebelum bisa dipakai.
"""
import pandas as pd
from core.alert_engine import push

def get_expiry_risks(df: pd.DataFrame, warning_days: int = 30) -> list[dict]:
    today_str = df["date"].max()
    today_df = df[df["date"] == today_str].copy()
    risks = []
    for _, row in today_df.iterrows():
        if row["days_to_expiry"] <= warning_days:
            severity = "CRITICAL" if row["days_to_expiry"] <= 7 else "WARNING"
            push("warehouse", severity,
                 f"Expiry risk: {row['sku_id']}",
                 f"{row['sku_name']} expire dalam {row['days_to_expiry']} hari | Stock: {row['stock_qty']} {row['unit']}")
            risks.append({
                "sku_id": row["sku_id"], "sku_name": row["sku_name"],
                "days_to_expiry": int(row["days_to_expiry"]),
                "stock_qty": row["stock_qty"], "unit": row["unit"],
                "stock_value_myr": row["stock_value_myr"],
                "severity": severity,
                "action": "DISPOSE / RUSH USE" if row["days_to_expiry"] <= 7 else "PRIORITIZE IN PRODUCTION",
            })
    return sorted(risks, key=lambda x: x["days_to_expiry"])

def run_expiry_agent() -> str:
    df = pd.read_csv("data/simulated/warehouse_inventory.csv")
    risks = get_expiry_risks(df)
    lines = ["=== EXPIRY RISK REPORT ===\n"]
    if not risks:
        return lines[0] + "Tidak ada item yang akan expire dalam 30 hari."
    total_at_risk = sum(r["stock_value_myr"] for r in risks)
    lines.append(f"Item berisiko : {len(risks)}")
    lines.append(f"Nilai at risk : MYR {total_at_risk:,.0f}\n")
    for r in risks:
        lines.append(f"  [{r['severity']:8s}] {r['sku_name']:30s} | Expire: {r['days_to_expiry']:3d}d | {r['action']}")
    return "\n".join(lines)
