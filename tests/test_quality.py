"""Tests untuk quality module."""
import pytest, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def test_quality_anomaly_detection():
    import pandas as pd, numpy as np
    from agents.quality.anomaly_detect import detect_quality_anomalies
    df = pd.DataFrame({
        "date": ["2024-01-01"]*50, "production_line": ["LINE-A"]*50,
        "ffa_pct": np.append(np.random.normal(0.12,0.01,45), np.random.normal(0.35,0.05,5)),
        "moisture_pct": np.random.normal(0.08,0.01,50),
        "color_lovibond_r": np.random.normal(3.1,0.3,50),
        "reject_rate_pct": np.random.normal(0.8,0.3,50),
        "quality_status": ["PASS"]*50, "actual_output_mt": [300]*50,
        "planned_output_mt": [320]*50, "oee_pct": [85]*50,
        "runtime_hours": [22]*50, "downtime_hours": [2]*50, "is_anomaly": [0]*50,
    })
    result = detect_quality_anomalies(df)
    assert "rule_fail" in result.columns
    assert result["rule_fail"].sum() >= 4  # at least the 5 high FFA batches

def test_yield_forecast():
    import pandas as pd, numpy as np
    from agents.quality.yield_forecast import forecast_yield
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=60).astype(str),
        "production_line": ["LINE-A"]*60,
        "oee_pct": np.random.normal(82, 3, 60),
        "runtime_hours": np.random.normal(22, 1, 60),
        "reject_rate_pct": np.random.normal(1, 0.3, 60),
        "actual_output_mt": np.random.normal(300, 15, 60),
        "planned_output_mt": [320]*60,
    })
    result = forecast_yield(df, "LINE-A", horizon_days=7)
    assert "total_7d_forecast_mt" in result
    assert result["total_7d_forecast_mt"] > 0
    assert len(result["daily_forecasts"]) == 7
