"""
Wilmar Plant Intelligence - Data Simulation Engine
===================================================
Generate dataset simulasi realistis untuk seluruh sistem.

Jalankan: python scripts/generate_data.py

Output (data/simulated/):
  sensor_data.csv          - Data sensor 10 mesin selama 1 tahun
  maintenance_history.csv  - Riwayat maintenance & breakdown
  warehouse_stock.csv      - Stok spare part saat ini (500+ item)
  consumption_history.csv  - History konsumsi spare part (2 tahun)
  quality_batches.csv      - Data kualitas batch CPO (1000+ batch)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import random

random.seed(42)
np.random.seed(42)

OUTPUT_DIR = "data/simulated"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────

MACHINES = [
    {"id": "REF-01", "type": "Refinery",    "line": 1, "age_months": 36},
    {"id": "REF-02", "type": "Refinery",    "line": 1, "age_months": 24},
    {"id": "REF-03", "type": "Refinery",    "line": 2, "age_months": 58},  # older = degrades faster
    {"id": "DRY-01", "type": "Dryer",       "line": 1, "age_months": 18},
    {"id": "DRY-02", "type": "Dryer",       "line": 2, "age_months": 42},
    {"id": "PMP-01", "type": "Pump",        "line": 1, "age_months": 12},
    {"id": "PMP-02", "type": "Pump",        "line": 1, "age_months": 30},
    {"id": "PMP-03", "type": "Pump",        "line": 2, "age_months": 48},
    {"id": "BLR-01", "type": "Boiler",      "line": 1, "age_months": 60},
    {"id": "BLR-02", "type": "Boiler",      "line": 2, "age_months": 20},
]

# Normal operating ranges per machine type
MACHINE_SPECS = {
    "Refinery": {
        "temp":      {"normal": (65, 80),  "alert": 92,  "noise": 3.0},
        "vibration": {"normal": (0.6, 1.8),"alert": 4.2, "noise": 0.15},
        "pressure":  {"normal": (9, 12),   "alert": 15,  "noise": 0.4},
        "rpm":       {"normal": (1440, 1480),"alert": 1550,"noise": 8},
        "current":   {"normal": (48, 55),  "alert": 65,  "noise": 1.5},
    },
    "Dryer": {
        "temp":      {"normal": (105, 125), "alert": 140, "noise": 4.0},
        "vibration": {"normal": (0.3, 1.2), "alert": 3.0, "noise": 0.1},
        "pressure":  {"normal": (0.5, 1.5), "alert": 2.5, "noise": 0.1},
        "rpm":       {"normal": (960, 1000),"alert": 1050,"noise": 5},
        "current":   {"normal": (30, 38),   "alert": 48,  "noise": 1.2},
    },
    "Pump": {
        "temp":      {"normal": (40, 60),  "alert": 75,  "noise": 2.0},
        "vibration": {"normal": (0.4, 1.5),"alert": 3.5, "noise": 0.12},
        "pressure":  {"normal": (4, 8),    "alert": 12,  "noise": 0.3},
        "rpm":       {"normal": (1450, 1490),"alert": 1560,"noise": 6},
        "current":   {"normal": (22, 28),  "alert": 38,  "noise": 0.8},
    },
    "Boiler": {
        "temp":      {"normal": (155, 175), "alert": 190, "noise": 5.0},
        "vibration": {"normal": (0.8, 2.5), "alert": 5.0, "noise": 0.2},
        "pressure":  {"normal": (7, 10),    "alert": 13,  "noise": 0.5},
        "rpm":       {"normal": (0, 0),     "alert": 0,   "noise": 0},   # no rotation
        "current":   {"normal": (60, 75),   "alert": 90,  "noise": 2.0},
    },
}


# ─── 1. SENSOR DATA ──────────────────────────────────────────────────────────

def generate_sensor_data(days: int = 365) -> pd.DataFrame:
    """
    Generate time series sensor data untuk semua mesin.
    Model degradasi: gradual wear + periodic maintenance reset + random failures.
    """
    print("📡 Generating sensor data...")
    rows = []

    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=days)

    for machine in MACHINES:
        mid = machine["id"]
        mtype = machine["type"]
        age = machine["age_months"]
        specs = MACHINE_SPECS[mtype]

        # Age factor: older machines have higher baseline degradation
        age_factor = 1.0 + (age / 120)  # max ~1.5x at 5 tahun

        # Simulated health score: starts at 100%, degrades over time
        health = 100.0
        degradation_rate = age_factor * np.random.uniform(0.03, 0.08)  # per day
        rul = 100.0 / degradation_rate  # estimated days until failure

        maintenance_intervals = []
        current = start_date

        while current <= end_date:
            # Maintenance event: reset health every ~90 days (randomized)
            if health < 30 or (maintenance_intervals and
                               (current - maintenance_intervals[-1]).days > np.random.randint(75, 110)):
                health = np.random.uniform(88, 98)
                rul = health / degradation_rate
                maintenance_intervals.append(current)

            health -= degradation_rate + np.random.normal(0, 0.02)
            health = max(5, min(100, health))
            rul = max(1, health / degradation_rate)

            # 8 readings per day (every 3 hours)
            for hour in range(0, 24, 3):
                ts = current + timedelta(hours=hour)

                # Compute degradation-aware sensor values
                deg = (100 - health) / 100  # 0=healthy, 1=failed

                temp_base    = specs["temp"]["normal"]
                vib_base     = specs["vibration"]["normal"]
                pres_base    = specs["pressure"]["normal"]
                rpm_base     = specs["rpm"]["normal"]
                curr_base    = specs["current"]["normal"]

                # As health degrades: temp↑, vibration↑, pressure unstable, rpm↓, current↑
                temp      = np.random.uniform(*temp_base)    * (1 + 0.25 * deg) + np.random.normal(0, specs["temp"]["noise"])
                vibration = np.random.uniform(*vib_base)     * (1 + 1.8  * deg) + np.random.normal(0, specs["vibration"]["noise"])
                current_a = np.random.uniform(*curr_base)    * (1 + 0.3  * deg) + np.random.normal(0, specs["current"]["noise"])

                if mtype == "Boiler":
                    pressure = np.random.uniform(*pres_base) * (1 - 0.15 * deg) + np.random.normal(0, specs["pressure"]["noise"])
                    rpm_val  = 0
                else:
                    pressure = np.random.uniform(*pres_base) + np.random.normal(0, specs["pressure"]["noise"])
                    rpm_val  = np.random.uniform(*rpm_base) * (1 - 0.05 * deg) + np.random.normal(0, specs["rpm"]["noise"])

                # Shift anomaly: add occasional spikes
                if np.random.random() < 0.02:
                    vibration *= np.random.uniform(1.5, 2.5)

                rows.append({
                    "timestamp":    ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "machine_id":   mid,
                    "machine_type": mtype,
                    "line":         machine["line"],
                    "temp_c":       round(temp, 2),
                    "vibration_mms":round(max(0, vibration), 3),
                    "pressure_bar": round(max(0, pressure), 2),
                    "rpm":          round(max(0, rpm_val), 1),
                    "current_a":    round(max(0, current_a), 2),
                    "health_score": round(health, 1),
                    "rul_days":     round(rul, 1),
                    "is_anomaly":   int(vibration > specs["vibration"]["alert"] * 0.8
                                       or temp > specs["temp"]["alert"] * 0.9),
                })

            current += timedelta(days=1)

    df = pd.DataFrame(rows)
    path = f"{OUTPUT_DIR}/sensor_data.csv"
    df.to_csv(path, index=False)
    print(f"   ✅ {len(df):,} rows → {path}")
    return df


# ─── 2. MAINTENANCE HISTORY ──────────────────────────────────────────────────

def generate_maintenance_history(sensor_df: pd.DataFrame) -> pd.DataFrame:
    print("🔧 Generating maintenance history...")
    rows = []

    maintenance_types = [
        ("Preventive", "Routine inspection & lubrication",         2, 4),
        ("Preventive", "Bearing replacement",                      4, 8),
        ("Preventive", "Belt & seal inspection",                   2, 3),
        ("Corrective", "Emergency bearing replacement",            6, 12),
        ("Corrective", "Motor rewinding",                         16, 24),
        ("Corrective", "Pump impeller replacement",                8, 14),
        ("Predictive", "Vibration-triggered inspection",           3, 6),
        ("Predictive", "Thermal anomaly follow-up",                2, 5),
    ]

    for machine in MACHINES:
        mid = machine["id"]
        machine_sensor = sensor_df[sensor_df["machine_id"] == mid].copy()
        machine_sensor["timestamp"] = pd.to_datetime(machine_sensor["timestamp"])

        # Create maintenance events at health reset points
        dates = pd.to_datetime(machine_sensor["timestamp"]).dt.date.unique()
        last_maint = None

        for i, date in enumerate(dates):
            day_data = machine_sensor[machine_sensor["timestamp"].dt.date == date]
            avg_health = day_data["health_score"].mean()

            # Maintenance triggered if: health low or periodic
            trigger = (
                avg_health < 35 or
                (last_maint is None and i == 0) or
                (last_maint and (pd.Timestamp(date) - last_maint).days > np.random.randint(80, 100))
            )

            if trigger:
                mtype_w = (
                    [mt for mt in maintenance_types if mt[0] == "Corrective"] if avg_health < 35
                    else [mt for mt in maintenance_types if mt[0] in ("Preventive", "Predictive")]
                )
                mtype = random.choice(mtype_w)
                duration = round(np.random.uniform(mtype[2], mtype[3]), 1)
                cost = round(np.random.uniform(2_000_000, 45_000_000), -3)  # in IDR

                rows.append({
                    "work_order_id":  f"WO-{mid}-{date.strftime('%Y%m%d')}",
                    "machine_id":     mid,
                    "machine_type":   machine["type"],
                    "date":           str(date),
                    "type":           mtype[0],
                    "description":    mtype[1],
                    "duration_hours": duration,
                    "cost_idr":       cost,
                    "technician":     random.choice(["Ahmad", "Budi", "Candra", "Denny", "Eko"]),
                    "health_before":  round(avg_health, 1),
                    "health_after":   round(np.random.uniform(88, 98), 1),
                    "downtime_hours": round(duration * np.random.uniform(0.8, 1.2), 1),
                })
                last_maint = pd.Timestamp(date)

    df = pd.DataFrame(rows)
    path = f"{OUTPUT_DIR}/maintenance_history.csv"
    df.to_csv(path, index=False)
    print(f"   ✅ {len(df):,} rows → {path}")
    return df


# ─── 3. WAREHOUSE STOCK ──────────────────────────────────────────────────────

def generate_warehouse_stock() -> pd.DataFrame:
    print("📦 Generating warehouse stock...")

    spare_parts = [
        # (code_prefix, description, category, unit_cost_range, min_stock, lead_time_days)
        ("BRG",  "Bearing SKF 6205-ZZ",         "Bearings",   (85_000, 120_000),    5, 3),
        ("BRG",  "Bearing SKF 6208-2RS",         "Bearings",   (120_000, 180_000),   4, 3),
        ("BRG",  "Bearing FAG 22218-E1",         "Bearings",   (380_000, 520_000),   3, 7),
        ("BRG",  "Bearing NTN 6310",             "Bearings",   (150_000, 200_000),   4, 4),
        ("BRG",  "Bearing Spherical 22315",      "Bearings",   (650_000, 850_000),   2, 10),
        ("SEL",  "Mechanical Seal Type 1",       "Seals",      (220_000, 350_000),   6, 5),
        ("SEL",  "Mechanical Seal Type 2",       "Seals",      (380_000, 520_000),   4, 7),
        ("SEL",  "O-Ring Kit EPDM 70",           "Seals",      (45_000, 80_000),    10, 2),
        ("SEL",  "Lip Seal 45x62x8",             "Seals",      (35_000, 60_000),     8, 2),
        ("BLT",  "V-Belt A-60",                  "Belts",      (85_000, 130_000),    6, 3),
        ("BLT",  "V-Belt B-75",                  "Belts",      (95_000, 140_000),    6, 3),
        ("BLT",  "Timing Belt HTD 1440-14M",     "Belts",      (280_000, 420_000),   3, 7),
        ("FLT",  "Oil Filter HF6553",            "Filters",    (65_000, 95_000),    10, 2),
        ("FLT",  "Air Filter AF25557",           "Filters",    (120_000, 180_000),   8, 3),
        ("FLT",  "Hydraulic Filter P565373",     "Filters",    (185_000, 250_000),   5, 4),
        ("ELC",  "Contactor Schneider LC1D25",   "Electrical", (380_000, 520_000),   3, 5),
        ("ELC",  "Thermal Relay LR2D13",         "Electrical", (220_000, 320_000),   4, 5),
        ("ELC",  "Circuit Breaker 3P 32A",       "Electrical", (180_000, 280_000),   4, 4),
        ("ELC",  "Proximity Sensor NPN",         "Electrical", (95_000, 145_000),    6, 3),
        ("LUB",  "Grease SKF LGMT2 1kg",         "Lubricants", (85_000, 120_000),   12, 2),
        ("LUB",  "Oil Mobil DTE 25",             "Lubricants", (125_000, 175_000),   8, 2),
        ("LUB",  "Oil Chain & Drive",            "Lubricants", (95_000, 135_000),    8, 2),
        ("GKT",  "Gasket Spiral Wound 3inch",    "Gaskets",    (145_000, 210_000),   5, 5),
        ("GKT",  "Flange Gasket 4inch PTFE",     "Gaskets",    (65_000, 95_000),     8, 3),
        ("VLV",  "Ball Valve 2inch SS316",       "Valves",     (485_000, 680_000),   2, 10),
        ("VLV",  "Check Valve 1.5inch",          "Valves",     (320_000, 480_000),   3, 8),
        ("PMP",  "Impeller Pump Type A",         "Pump Parts", (680_000, 950_000),   2, 14),
        ("PMP",  "Pump Shaft Seal Kit",          "Pump Parts", (380_000, 560_000),   3, 10),
        ("CPL",  "Coupling Jaw Spider 65mm",     "Couplings",  (185_000, 280_000),   3, 7),
        ("INS",  "Thermocouple Type K",          "Instruments",  (145_000, 220_000), 4, 5),
        ("INS",  "Pressure Gauge 0-16bar",       "Instruments",  (95_000, 145_000),  4, 4),
        ("INS",  "Flow Meter Insert 2inch",      "Instruments",  (580_000, 850_000), 2, 14),
    ]

    rows = []
    suppliers = ["PT. Bumi Teknik", "PT. Indo Spare", "PT. Maju Jaya", "PT. Teknik Prima", "CV. Surya Parts"]

    for i, (prefix, desc, category, cost_range, min_stk, lead_time) in enumerate(spare_parts):
        code = f"{prefix}-{str(i+1).zfill(4)}"
        unit_cost = round(np.random.uniform(*cost_range), -3)
        current_stock = max(0, int(np.random.normal(min_stk * 1.8, min_stk * 0.6)))
        reorder_pt = int(min_stk * 1.2 + (lead_time / 3))

        rows.append({
            "item_code":       code,
            "description":     desc,
            "category":        category,
            "stock_qty":       current_stock,
            "min_stock":       min_stk,
            "reorder_point":   reorder_pt,
            "unit_cost_idr":   unit_cost,
            "supplier":        random.choice(suppliers),
            "lead_time_days":  lead_time,
            "location":        f"Rak-{chr(65 + i % 8)}{(i // 8) + 1}",
            "last_updated":    (datetime.now() - timedelta(hours=np.random.randint(1, 48))).strftime("%Y-%m-%d %H:%M"),
        })

    df = pd.DataFrame(rows)
    path = f"{OUTPUT_DIR}/warehouse_stock.csv"
    df.to_csv(path, index=False)
    print(f"   ✅ {len(df):,} items → {path}")
    return df


# ─── 4. CONSUMPTION HISTORY ──────────────────────────────────────────────────

def generate_consumption_history(stock_df: pd.DataFrame) -> pd.DataFrame:
    print("📋 Generating consumption history...")
    rows = []

    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # 2 years

    for _, item in stock_df.iterrows():
        # Consumption frequency: higher for common parts
        freq_per_month = {
            "Bearings": 2.5, "Seals": 2.0, "Belts": 1.5,
            "Filters": 3.0,  "Lubricants": 4.0, "Gaskets": 1.0,
            "Electrical": 0.8, "Valves": 0.4, "Pump Parts": 0.5,
            "Couplings": 0.6, "Instruments": 0.4,
        }.get(item["category"], 1.0)

        n_events = int(24 * freq_per_month * np.random.uniform(0.6, 1.4))
        event_dates = sorted(random.sample(
            [start_date + timedelta(days=d) for d in range(730)],
            min(n_events, 730)
        ))

        for date in event_dates:
            qty = max(1, int(np.random.poisson(1.5)))
            rows.append({
                "date":       date.strftime("%Y-%m-%d"),
                "item_code":  item["item_code"],
                "description":item["description"],
                "category":   item["category"],
                "qty_used":   qty,
                "machine_id": random.choice([m["id"] for m in MACHINES]),
                "work_order": f"WO-{np.random.randint(10000, 99999)}",
                "cost_idr":   qty * item["unit_cost_idr"],
            })

    df = pd.DataFrame(rows)
    df = df.sort_values("date").reset_index(drop=True)
    path = f"{OUTPUT_DIR}/consumption_history.csv"
    df.to_csv(path, index=False)
    print(f"   ✅ {len(df):,} rows → {path}")
    return df


# ─── 5. QUALITY BATCHES ──────────────────────────────────────────────────────

def generate_quality_batches(days: int = 365) -> pd.DataFrame:
    """
    Generate CPO production batch quality data.
    Parameter: FFA, Moisture, DOBI, Colour (Lovibond)
    """
    print("🧪 Generating quality batch data...")
    rows = []

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    lines = ["LINE-1", "LINE-2"]

    # Process specs (normal production ranges)
    SPECS = {
        "ffa_pct":      {"mean": 2.8,  "std": 0.35, "usl": 3.5,  "lsl": None},
        "moisture_pct": {"mean": 0.10, "std": 0.025,"usl": 0.15, "lsl": None},
        "dobi":         {"mean": 2.60, "std": 0.18, "usl": None, "lsl": 2.31},
        "colour_r":     {"mean": 2.3,  "std": 0.28, "usl": 3.0,  "lsl": None},
        "yield_pct":    {"mean": 22.8, "std": 0.8,  "usl": None, "lsl": 20.5},
    }

    batch_num = 1000
    current = start_date

    while current <= end_date:
        for line in lines:
            # 3 batches per line per day
            for shift in range(3):
                ts = current + timedelta(hours=shift * 8 + np.random.uniform(0, 7))

                # Occasionally inject process upsets (8% of batches)
                upset = np.random.random() < 0.08
                upset_type = random.choice(["temp_high", "feed_quality", "equipment"]) if upset else None

                vals = {}
                for param, spec in SPECS.items():
                    base = np.random.normal(spec["mean"], spec["std"])
                    if upset:
                        # Upset shifts the distribution toward out-of-spec
                        if upset_type == "temp_high" and param in ("ffa_pct", "colour_r"):
                            base += np.random.uniform(0.4, 0.9)
                        elif upset_type == "feed_quality" and param in ("ffa_pct", "dobi"):
                            base += np.random.uniform(0.3, 0.7) if param == "ffa_pct" else -np.random.uniform(0.2, 0.5)
                        elif upset_type == "equipment" and param == "moisture_pct":
                            base += np.random.uniform(0.03, 0.08)
                    vals[param] = round(base, 4)

                # Determine status
                status = "PASS"
                fails = []
                for param, spec in SPECS.items():
                    v = vals[param]
                    if spec["usl"] and v > spec["usl"]:
                        status = "FAIL"
                        fails.append(f"{param}>{spec['usl']}")
                    if spec["lsl"] and v < spec["lsl"]:
                        status = "FAIL"
                        fails.append(f"{param}<{spec['lsl']}")

                if status == "FAIL" and not fails and upset:
                    status = "WARNING"

                rows.append({
                    "batch_id":      f"B-{ts.strftime('%Y%m%d')}-{batch_num:04d}",
                    "timestamp":     ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "line_id":       line,
                    "shift":         ["Pagi", "Siang", "Malam"][shift],
                    "ffa_pct":       vals["ffa_pct"],
                    "moisture_pct":  vals["moisture_pct"],
                    "dobi":          vals["dobi"],
                    "colour_r":      vals["colour_r"],
                    "yield_pct":     vals["yield_pct"],
                    "status":        status,
                    "fail_params":   "|".join(fails) if fails else "",
                    "upset_type":    upset_type or "",
                    "operator":      random.choice(["Agus", "Bambang", "Citra", "Dewi", "Eko", "Fitri"]),
                })
                batch_num += 1

        current += timedelta(days=1)

    df = pd.DataFrame(rows)
    path = f"{OUTPUT_DIR}/quality_batches.csv"
    df.to_csv(path, index=False)
    pass_rate = (df["status"] == "PASS").mean() * 100
    print(f"   ✅ {len(df):,} batches → {path} (pass rate: {pass_rate:.1f}%)")
    return df


# ─── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🏭 Wilmar Plant Intelligence — Data Generation")
    print("=" * 55)
    print(f"Output directory: {OUTPUT_DIR}/\n")

    sensor_df = generate_sensor_data(days=365)
    maintenance_df = generate_maintenance_history(sensor_df)
    stock_df = generate_warehouse_stock()
    consumption_df = generate_consumption_history(stock_df)
    quality_df = generate_quality_batches(days=365)

    print("\n✅ All datasets generated successfully!")
    print("\nFile sizes:")
    for fname in ["sensor_data.csv", "maintenance_history.csv",
                  "warehouse_stock.csv", "consumption_history.csv", "quality_batches.csv"]:
        fpath = f"{OUTPUT_DIR}/{fname}"
        size_kb = os.path.getsize(fpath) / 1024
        df = pd.read_csv(fpath)
        print(f"  {fname:<35} {len(df):>6,} rows  {size_kb:>7.1f} KB")

    print("\nNext step: python main.py --demo")
