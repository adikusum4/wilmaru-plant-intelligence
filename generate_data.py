"""
Wilmar Plant Intelligence — Dataset Simulator
==============================================
Jalankan PERTAMA sebelum apapun:
    python generate_data.py

Menghasilkan 3 dataset simulasi realistis di data/simulated/:
  - warehouse_inventory.csv
  - machine_sensor_logs.csv
  - production_quality.csv

Semua angka berdasarkan karakteristik industri palm oil Indonesia.
Tidak perlu internet, tidak perlu API key.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os, random

np.random.seed(42)
random.seed(42)
OUT = "data/simulated"
os.makedirs(OUT, exist_ok=True)

# ── 1. WAREHOUSE INVENTORY ───────────────────────────────────────────────────
print("Generating warehouse_inventory.csv...")

SKUS = {
    "CPO-BULK-001":    {"name": "Crude Palm Oil",         "unit": "MT",  "base_stock": 5000, "lead_days": 14, "shelf_days": 60,  "unit_cost": 3800},
    "PKO-BULK-002":    {"name": "Palm Kernel Oil",        "unit": "MT",  "base_stock": 1200, "lead_days": 21, "shelf_days": 90,  "unit_cost": 5200},
    "OLEIN-003":       {"name": "RBD Palm Olein",         "unit": "MT",  "base_stock": 2800, "lead_days": 10, "shelf_days": 180, "unit_cost": 4200},
    "FILTER-MECH-010": {"name": "Filter Press Cloth",     "unit": "PCS", "base_stock": 200,  "lead_days": 30, "shelf_days": 730, "unit_cost": 450},
    "BEARING-6205":    {"name": "Bearing 6205 ZZ",        "unit": "PCS", "base_stock": 150,  "lead_days": 21, "shelf_days": 1825,"unit_cost": 85},
    "SEAL-PUMP-007":   {"name": "Centrifugal Pump Seal",  "unit": "PCS", "base_stock": 80,   "lead_days": 28, "shelf_days": 1825,"unit_cost": 320},
    "CAUSTIC-SODA":    {"name": "Caustic Soda Flakes",    "unit": "KG",  "base_stock": 8000, "lead_days": 7,  "shelf_days": 365, "unit_cost": 6.5},
    "PHOSPHORIC-ACID": {"name": "Phosphoric Acid 85%",    "unit": "L",   "base_stock": 3000, "lead_days": 14, "shelf_days": 365, "unit_cost": 18},
    "PACKAGING-20L":   {"name": "Jerry Can 20L HDPE",     "unit": "PCS", "base_stock": 15000,"lead_days": 21, "shelf_days": 730, "unit_cost": 12},
    "PACKAGING-1L":    {"name": "Bottle 1L PET",          "unit": "PCS", "base_stock": 40000,"lead_days": 14, "shelf_days": 730, "unit_cost": 3.2},
}

rows = []
today = datetime.now()
for sku_id, info in SKUS.items():
    for i in range(90):
        date = today - timedelta(days=89-i)
        daily_demand = max(0, np.random.normal(info["base_stock"]*0.015, info["base_stock"]*0.004))
        received = info["base_stock"]*0.018 if i % info["lead_days"] == 0 else 0
        stock = max(0, info["base_stock"] + received - daily_demand*(i%info["lead_days"]) + np.random.normal(0, info["base_stock"]*0.02))
        expiry_date = date + timedelta(days=info["shelf_days"] - random.randint(0, info["shelf_days"]//3))
        days_to_expiry = (expiry_date - date).days
        reorder_point = daily_demand * info["lead_days"] * 1.3
        rows.append({
            "date": date.strftime("%Y-%m-%d"),
            "sku_id": sku_id, "sku_name": info["name"],
            "unit": info["unit"],
            "stock_qty": round(stock, 1),
            "daily_demand": round(daily_demand, 2),
            "lead_time_days": info["lead_days"],
            "reorder_point": round(reorder_point, 1),
            "days_to_expiry": max(1, days_to_expiry),
            "unit_cost_myr": info["unit_cost"],
            "stock_value_myr": round(stock * info["unit_cost"], 0),
            "status": "CRITICAL" if stock < reorder_point*0.5 else ("LOW" if stock < reorder_point else "OK"),
        })

pd.DataFrame(rows).to_csv(f"{OUT}/warehouse_inventory.csv", index=False)
print(f"  -> {len(rows)} rows, {len(SKUS)} SKUs x 90 days")

# ── 2. MACHINE SENSOR LOGS ───────────────────────────────────────────────────
print("Generating machine_sensor_logs.csv...")

MACHINES = {
    "M-CENTRIFUGE-01": {"name": "Decanter Centrifuge",    "base_temp": 68,  "base_vib": 2.1, "base_rpm": 3500, "failure_at_day": 75},
    "M-CENTRIFUGE-02": {"name": "Decanter Centrifuge",    "base_temp": 70,  "base_vib": 2.3, "base_rpm": 3500, "failure_at_day": None},
    "M-PRESS-01":      {"name": "Screw Press",            "base_temp": 85,  "base_vib": 3.8, "base_rpm": 1450, "failure_at_day": None},
    "M-PRESS-02":      {"name": "Screw Press",            "base_temp": 83,  "base_vib": 3.5, "base_rpm": 1450, "failure_at_day": 55},
    "M-PUMP-01":       {"name": "Transfer Pump",          "base_temp": 55,  "base_vib": 1.2, "base_rpm": 2900, "failure_at_day": None},
    "M-REFINERY-01":   {"name": "Bleaching Reactor",      "base_temp": 110, "base_vib": 0.8, "base_rpm": 120,  "failure_at_day": 80},
    "M-DRYER-01":      {"name": "Flash Dryer",            "base_temp": 145, "base_vib": 1.5, "base_rpm": 850,  "failure_at_day": None},
}

sensor_rows = []
for mid, minfo in MACHINES.items():
    fail_day = minfo["failure_at_day"]
    for day in range(90):
        for hour in range(0, 24, 4):
            ts = today - timedelta(days=89-day, hours=hour)
            # Degradation: readings worsen approaching failure
            if fail_day and day > fail_day - 15:
                deg = (day - (fail_day-15)) / 15.0
            else:
                deg = 0.0
            temp  = minfo["base_temp"]  * (1 + deg*0.18) + np.random.normal(0, 1.5)
            vib   = minfo["base_vib"]   * (1 + deg*0.55) + np.random.normal(0, 0.12)
            rpm   = minfo["base_rpm"]   * (1 - deg*0.08) + np.random.normal(0, minfo["base_rpm"]*0.005)
            current = 45 * (1 + deg*0.12) + np.random.normal(0, 1.2)
            pressure = 4.5  * (1 + deg*0.20) + np.random.normal(0, 0.15)
            rul = max(0, (fail_day - day)*24 - hour) if fail_day else 9999
            health = max(0, 100 - deg*70 + np.random.normal(0, 3))
            status = "FAILED" if (fail_day and day >= fail_day) else \
                     ("CRITICAL" if deg > 0.7 else ("WARNING" if deg > 0.3 else "NORMAL"))
            sensor_rows.append({
                "timestamp": ts.strftime("%Y-%m-%d %H:%M"),
                "machine_id": mid, "machine_name": minfo["name"],
                "temperature_c": round(temp, 1),
                "vibration_mm_s": round(max(0, vib), 3),
                "rpm": round(max(0, rpm), 0),
                "current_amp": round(max(0, current), 2),
                "pressure_bar": round(max(0, pressure), 2),
                "health_score": round(max(0, min(100, health)), 1),
                "rul_hours": int(rul),
                "status": status,
            })

pd.DataFrame(sensor_rows).to_csv(f"{OUT}/machine_sensor_logs.csv", index=False)
print(f"  -> {len(sensor_rows)} rows, {len(MACHINES)} machines x 90 days x 6 readings/day")

# ── 3. PRODUCTION QUALITY ────────────────────────────────────────────────────
print("Generating production_quality.csv...")

LINES = ["LINE-A-CPO", "LINE-B-CPO", "LINE-C-PKO", "LINE-D-OLEIN"]
qual_rows = []
for day in range(90):
    date = today - timedelta(days=89-day)
    for line in LINES:
        # Simulate quality drift events (bad batches)
        bad_event = (day in [22, 45, 67, 78] and line in ["LINE-A-CPO", "LINE-C-PKO"])
        planned_output = {"LINE-A-CPO":320,"LINE-B-CPO":310,"LINE-C-PKO":180,"LINE-D-OLEIN":240}[line]
        ffa = np.random.normal(0.12, 0.02) if not bad_event else np.random.normal(0.28, 0.04)
        moisture = np.random.normal(0.08, 0.01) if not bad_event else np.random.normal(0.19, 0.03)
        color_lov = np.random.normal(3.1, 0.3) if not bad_event else np.random.normal(5.8, 0.6)
        reject_rate = np.random.normal(0.8, 0.3) if not bad_event else np.random.normal(6.5, 1.2)
        actual_output = planned_output * (1 - reject_rate/100) * np.random.uniform(0.92, 0.99)
        runtime_h = np.random.uniform(20, 23.5)
        downtime_h = 24 - runtime_h
        oee = (actual_output / planned_output) * (runtime_h / 24) * np.random.uniform(0.88, 0.97)
        qual_rows.append({
            "date": date.strftime("%Y-%m-%d"),
            "production_line": line,
            "planned_output_mt": planned_output,
            "actual_output_mt": round(actual_output, 1),
            "reject_rate_pct": round(max(0, reject_rate), 2),
            "ffa_pct": round(max(0, ffa), 3),
            "moisture_pct": round(max(0, moisture), 3),
            "color_lovibond_r": round(max(0, color_lov), 2),
            "runtime_hours": round(runtime_h, 1),
            "downtime_hours": round(downtime_h, 1),
            "oee_pct": round(max(0, min(100, oee*100)), 1),
            "quality_status": "FAIL" if (ffa>0.2 or moisture>0.15 or reject_rate>4) else "PASS",
            "is_anomaly": int(bad_event),
        })

pd.DataFrame(qual_rows).to_csv(f"{OUT}/production_quality.csv", index=False)
print(f"  -> {len(qual_rows)} rows, {len(LINES)} lines x 90 days")
print("\nDone! Files in data/simulated/")
print("  warehouse_inventory.csv")
print("  machine_sensor_logs.csv")
print("  production_quality.csv")
