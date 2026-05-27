"""Tests untuk warehouse module."""
import pytest, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def test_reorder_logic():
    import pandas as pd
    from agents.warehouse.reorder import get_reorder_recommendations
    df = pd.DataFrame([{
        "date": "2024-12-01", "sku_id": "TEST-001", "sku_name": "Test Item",
        "unit": "PCS", "stock_qty": 10, "daily_demand": 5.0, "lead_time_days": 7,
        "reorder_point": 40, "days_to_expiry": 365, "unit_cost_myr": 100, "stock_value_myr": 1000, "status": "CRITICAL"
    }])
    recs = get_reorder_recommendations(df)
    assert len(recs) == 1
    assert recs[0]["needs_reorder"] == True

def test_expiry_detection():
    import pandas as pd
    from agents.warehouse.expiry_risk import get_expiry_risks
    df = pd.DataFrame([{
        "date": "2024-12-01", "sku_id": "EXP-001", "sku_name": "Expiring Soon",
        "unit": "KG", "stock_qty": 500, "daily_demand": 10, "lead_time_days": 7,
        "reorder_point": 80, "days_to_expiry": 5, "unit_cost_myr": 10, "stock_value_myr": 5000, "status": "OK"
    }])
    from core.alert_engine import clear; clear()
    risks = get_expiry_risks(df, warning_days=30)
    assert len(risks) == 1
    assert risks[0]["severity"] == "CRITICAL"
