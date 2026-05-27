"""Tests untuk maintenance module."""
import pytest, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def test_alert_engine():
    from core.alert_engine import push, get_all, clear
    clear()
    push("test", "WARNING", "Test alert", "detail")
    alerts = get_all("WARNING")
    assert len(alerts) == 1
    assert alerts[0].title == "Test alert"
    clear()

def test_sensor_feature_engineering():
    import pandas as pd, numpy as np
    from agents.maintenance.rul_predictor import engineer_features
    df = pd.DataFrame({
        "machine_id": ["M1"]*20, "timestamp": pd.date_range("2024-01-01", periods=20, freq="4h"),
        "temperature_c": np.random.normal(70, 2, 20),
        "vibration_mm_s": np.random.normal(2.1, 0.2, 20),
        "rpm": np.random.normal(3500, 50, 20),
        "current_amp": np.random.normal(45, 2, 20),
        "pressure_bar": np.random.normal(4.5, 0.2, 20),
        "health_score": np.linspace(90, 60, 20),
        "rul_hours": np.linspace(500, 100, 20),
        "status": ["NORMAL"]*20,
    })
    result = engineer_features(df)
    assert "temperature_c_roll6_mean" in result.columns
    assert "vib_temp_ratio" in result.columns

def test_isolation_forest_runs():
    import pandas as pd, numpy as np
    from agents.maintenance.sensor_monitor import detect_anomalies
    df = pd.DataFrame({
        "machine_id": ["M1"]*50, "timestamp": pd.date_range("2024-01-01", periods=50, freq="h"),
        "temperature_c": np.append(np.random.normal(70,2,45), np.random.normal(95,2,5)),
        "vibration_mm_s": np.append(np.random.normal(2,0.2,45), np.random.normal(8,0.5,5)),
        "current_amp": np.random.normal(45,2,50),
        "pressure_bar": np.random.normal(4.5,0.2,50),
        "machine_name": ["Test"]*50, "rpm": [3500]*50, "health_score": [80]*50,
        "rul_hours": [500]*50, "status": ["NORMAL"]*50,
    })
    result = detect_anomalies(df, contamination=0.1)
    assert "is_anomaly_ml" in result.columns
    assert result["is_anomaly_ml"].sum() > 0
