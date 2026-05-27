"""
Cross-Domain Insight Engine
============================
Inilah yang membuat proyek ini UNIK dibanding portfolio biasa.
Menghubungkan insight dari 3 modul berbeda menjadi satu rekomendasi.

Contoh:
  - Mesin CENTRIFUGE-01 akan rusak dalam 48 jam (Maintenance)
  → Cek stok spare part BEARING-6205 di warehouse (Warehouse)
  → Hitung dampak downtime ke target produksi minggu ini (Quality)
  → Rekomendasikan: jadwalkan maintenance SEKARANG atau delay produksi
"""
import pandas as pd
import numpy as np

def analyze_maintenance_warehouse_impact() -> dict:
    """
    Cross-domain: mesin mau rusak → spare part tersedia? → impact ke produksi?
    """
    try:
        sensor_df  = pd.read_csv("data/simulated/machine_sensor_logs.csv")
        inventory  = pd.read_csv("data/simulated/warehouse_inventory.csv")
        quality_df = pd.read_csv("data/simulated/production_quality.csv")

        from agents.maintenance.rul_predictor import predict_rul
        from agents.maintenance.schedule import _get_spareparts
        predictions = predict_rul(sensor_df)

        urgent_machines = [p for p in predictions if p["rul_predicted_hours"] < 168]
        inv_today = inventory[inventory["date"] == inventory["date"].max()]

        insights = []
        for machine in urgent_machines:
            parts_needed = _get_spareparts(machine["machine_name"])
            spare_status = []
            for part_str in parts_needed.split(","):
                part_id = part_str.strip().split(" ")[0]
                part_row = inv_today[inv_today["sku_id"] == part_id]
                if part_row.empty:
                    spare_status.append(f"{part_id}: NOT FOUND")
                else:
                    row = part_row.iloc[0]
                    needed_qty = int(part_str.strip().split("x")[-1]) if "x" in part_str else 1
                    available = row["stock_qty"]
                    ok = available >= needed_qty
                    spare_status.append(f"{part_id}: {'OK' if ok else 'INSUFFICIENT'} ({available:.0f} available, {needed_qty} needed)")

            # Estimate daily production loss if machine is down
            avg_daily_output = quality_df["actual_output_mt"].mean()
            lines_affected = 1
            downtime_est_h = 8  # default overhaul
            production_loss_mt = (downtime_est_h / 24) * avg_daily_output * lines_affected

            insights.append({
                "machine_id": machine["machine_id"],
                "rul_hours": machine["rul_predicted_hours"],
                "urgency": machine["urgency"],
                "spare_parts_check": spare_status,
                "all_parts_available": all("OK" in s for s in spare_status),
                "production_loss_est_mt": round(production_loss_mt, 1),
                "recommendation": _make_recommendation(machine, spare_status, production_loss_mt),
            })
        return {"insights": insights, "urgent_count": len(urgent_machines)}

    except Exception as e:
        return {"insights": [], "error": str(e), "urgent_count": 0}

def _make_recommendation(machine: dict, spare_status: list, loss: float) -> str:
    parts_ok = all("OK" in s for s in spare_status)
    rul = machine["rul_predicted_hours"]
    if rul < 48 and parts_ok:
        return f"SCHEDULE IMMEDIATELY — spare parts tersedia, jadwalkan dalam 24 jam. Est. production loss: {loss:.1f} MT"
    elif rul < 48 and not parts_ok:
        return f"URGENT PROCUREMENT — parts kurang, order darurat + notify production untuk contingency plan"
    elif rul < 168 and parts_ok:
        return f"SCHEDULE THIS WEEK — parts OK, jadwalkan di shift pagi untuk minimal downtime"
    else:
        return f"MONITOR — masukkan ke jadwal maintenance reguler minggu depan"

def run_cross_domain_analysis() -> str:
    result = analyze_maintenance_warehouse_impact()
    lines = ["=== CROSS-DOMAIN INSIGHT REPORT ==="]
    lines.append("(Menghubungkan: Maintenance ↔ Warehouse ↔ Production)\n")
    if result.get("error"):
        return "\n".join(lines) + f"\nError: {result['error']}"
    if not result["insights"]:
        lines.append("Tidak ada mesin urgent dalam 7 hari ke depan.")
        return "\n".join(lines)
    lines.append(f"Mesin perlu perhatian (< 7 hari): {result['urgent_count']}\n")
    for ins in result["insights"]:
        lines.append(f"Machine  : {ins['machine_id']} ({ins['urgency']})")
        lines.append(f"RUL      : {ins['rul_hours']} jam")
        lines.append("Spare parts:")
        for sp in ins["spare_parts_check"]:
            lines.append(f"  - {sp}")
        lines.append(f"Est. downtime loss : {ins['production_loss_est_mt']} MT output")
        lines.append(f"Rekomendasi        : {ins['recommendation']}")
        lines.append("")
    return "\n".join(lines)
