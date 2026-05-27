"""
Maintenance Schedule Agent
Input: RUL predictions → output: jadwal maintenance yang optimal.
Logic: prioritas berdasarkan RUL, spare part availability, shift calendar.
"""
import pandas as pd
from datetime import datetime, timedelta
from agents.maintenance.rul_predictor import predict_rul
from core.alert_engine import push

SHIFT_HOURS = {"PAGI": "07:00-15:00", "SIANG": "15:00-23:00", "MALAM": "23:00-07:00"}
MAINTENANCE_DURATION = {  # jam estimasi per jenis mesin
    "Decanter Centrifuge": 8, "Screw Press": 6, "Transfer Pump": 3,
    "Bleaching Reactor": 10, "Flash Dryer": 5,
}

def generate_schedule(predictions: list[dict]) -> list[dict]:
    """Buat jadwal maintenance, prioritaskan RUL paling kecil."""
    schedule = []
    today = datetime.now()
    for p in sorted(predictions, key=lambda x: x["rul_predicted_hours"]):
        if p["rul_predicted_hours"] >= 9000:
            continue
        deadline = today + timedelta(hours=p["rul_predicted_hours"] * 0.85)  # 15% safety buffer
        # Find best window: maintenance on shift PAGI saat rul hampir habis
        sched_date = deadline - timedelta(hours=24)
        dur = MAINTENANCE_DURATION.get(p["machine_name"], 6)
        schedule.append({
            "priority": len(schedule) + 1,
            "machine_id":   p["machine_id"],
            "machine_name": p["machine_name"],
            "scheduled_date": sched_date.strftime("%Y-%m-%d"),
            "shift":          "PAGI",
            "duration_hours": dur,
            "rul_at_schedule": p["rul_predicted_hours"],
            "urgency":        p["urgency"],
            "recommended_action": _get_action(p),
            "spare_parts_needed": _get_spareparts(p["machine_name"]),
        })
        if "URGENT" in p["urgency"]:
            push("maintenance", "CRITICAL",
                 f"Jadwal darurat: {p['machine_id']}",
                 f"Maintenance dijadwalkan {sched_date.strftime('%Y-%m-%d')} | RUL {p['rul_predicted_hours']}h")
    return schedule

def _get_action(p: dict) -> str:
    if p["health_score"] < 40: return "Major overhaul — ganti bearing, seal, check shaft alignment"
    if p["health_score"] < 65: return "Minor service — lubrication, vibration alignment, filter clean"
    return "Preventive inspection — check fasteners, lubrication levels"

def _get_spareparts(machine_name: str) -> str:
    parts = {
        "Decanter Centrifuge": "BEARING-6205 x2, SEAL-PUMP-007 x1",
        "Screw Press":         "BEARING-6205 x4, FILTER-MECH-010 x2",
        "Transfer Pump":       "SEAL-PUMP-007 x2",
        "Bleaching Reactor":   "FILTER-MECH-010 x4, SEAL-PUMP-007 x1",
        "Flash Dryer":         "BEARING-6205 x2",
    }
    return parts.get(machine_name, "Cek SOP manual")

def run_schedule_agent() -> str:
    df = pd.read_csv("data/simulated/machine_sensor_logs.csv", parse_dates=["timestamp"])
    from agents.maintenance.rul_predictor import predict_rul
    predictions = predict_rul(df)
    schedule = generate_schedule(predictions)

    lines = ["=== MAINTENANCE SCHEDULE REPORT ===\n"]
    if not schedule:
        return lines[0] + "Semua mesin dalam kondisi sehat. Tidak ada maintenance terjadwal."
    lines.append(f"{'#':>2}  {'Machine':22s}  {'Tanggal':12s}  {'Durasi':8s}  {'Urgency'}")
    lines.append("-" * 72)
    for s in schedule:
        lines.append(f"{s['priority']:>2}  {s['machine_id']:22s}  {s['scheduled_date']:12s}  {s['duration_hours']}h       {s['urgency']}")
        lines.append(f"    Aksi: {s['recommended_action']}")
        lines.append(f"    Spare parts: {s['spare_parts_needed']}\n")
    return "\n".join(lines)
