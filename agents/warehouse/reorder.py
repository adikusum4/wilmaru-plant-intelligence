"""
Reorder Agent — deteksi kapan perlu reorder dan generate PO otomatis.
Cross-domain: cek spare part availability sebelum maintenance dijadwalkan.
"""
import pandas as pd
import numpy as np
from core.alert_engine import push

def get_reorder_recommendations(df: pd.DataFrame) -> list[dict]:
    today_str = df["date"].max()
    today_df = df[df["date"] == today_str].copy()
    recs = []
    for _, row in today_df.iterrows():
        safety_stock = row["daily_demand"] * row["lead_time_days"] * 0.5
        reorder_point = row["daily_demand"] * row["lead_time_days"] + safety_stock
        eoq = np.sqrt((2 * row["daily_demand"] * 365 * row["unit_cost_myr"] * 0.2) /
                      (row["unit_cost_myr"] * 0.25 + 1))  # simplified EOQ
        days_of_stock = row["stock_qty"] / max(row["daily_demand"], 0.001)
        needs_reorder = row["stock_qty"] <= reorder_point
        if needs_reorder:
            push("warehouse", "WARNING" if days_of_stock > row["lead_time_days"] else "CRITICAL",
                 f"Reorder needed: {row['sku_id']}",
                 f"Stock: {row['stock_qty']} {row['unit']} | DoS: {days_of_stock:.0f}d | Lead: {row['lead_time_days']}d")
        recs.append({
            "sku_id": row["sku_id"], "sku_name": row["sku_name"],
            "current_stock": round(row["stock_qty"], 1),
            "unit": row["unit"],
            "days_of_stock": round(days_of_stock, 0),
            "reorder_point": round(reorder_point, 1),
            "suggested_qty": round(max(eoq, reorder_point * 1.5), 0),
            "estimated_cost_myr": round(max(eoq, reorder_point*1.5) * row["unit_cost_myr"], 0),
            "needs_reorder": needs_reorder,
            "urgency": "URGENT" if days_of_stock < row["lead_time_days"] else ("SOON" if needs_reorder else "OK"),
        })
    return sorted(recs, key=lambda x: x["days_of_stock"])

def run_reorder_agent() -> str:
    df = pd.read_csv("data/simulated/warehouse_inventory.csv")
    recs = get_reorder_recommendations(df)
    urgent = [r for r in recs if r["needs_reorder"]]
    lines = ["=== REORDER AGENT REPORT ===\n"]
    lines.append(f"Total SKU      : {len(recs)}")
    lines.append(f"Perlu reorder  : {len(urgent)}\n")
    lines.append(f"{'SKU':12s} {'Nama':28s} {'Stock':10s} {'DoS':6s} {'Status'}")
    lines.append("-" * 72)
    for r in recs[:15]:
        flag = " ← REORDER" if r["needs_reorder"] else ""
        lines.append(f"{r['sku_id']:12s} {r['sku_name']:28s} {str(r['current_stock'])+' '+r['unit']:10s} {str(int(r['days_of_stock']))+'d':6s} {r['urgency']}{flag}")
    return "\n".join(lines)
