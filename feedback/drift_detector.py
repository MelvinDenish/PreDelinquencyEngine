# pyre-ignore-all-errors
"""
Drift Detection with Evidently AI
Monitors model performance and detects data/prediction drift.
Triggers retraining when accuracy degrades beyond threshold.
"""
import os
import sys
import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import psycopg2
from sqlalchemy import create_engine

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config.settings import PostgresConfig

logger = logging.getLogger(__name__)


def compute_drift_metrics() -> dict:
    """
    Compute drift metrics by comparing current prediction distributions
    against baseline using Evidently AI.
    """
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
        from evidently.metrics import (
            DatasetDriftMetric,
            DataDriftTable,
        )
    except ImportError:
        logger.warning("Evidently AI not installed. Using basic drift detection.")
        return _basic_drift_detection()

    engine = create_engine(PostgresConfig.get_url())

    # Get recent scores (last 7 days as current)
    current_df = pd.read_sql(
        """SELECT risk_score, xgboost_score, lstm_score, ensemble_score
           FROM risk_scores
           WHERE scored_at > NOW() - INTERVAL '7 days'""",
        engine,
    )

    # Get baseline scores (8-30 days ago)
    baseline_df = pd.read_sql(
        """SELECT risk_score, xgboost_score, lstm_score, ensemble_score
           FROM risk_scores
           WHERE scored_at BETWEEN NOW() - INTERVAL '30 days'
                 AND NOW() - INTERVAL '7 days'""",
        engine,
    )

    if current_df.empty or baseline_df.empty:
        return {"status": "insufficient_data", "drift_detected": False}

    # Fill NaN for LSTM scores
    for df in [current_df, baseline_df]:
        df.fillna(0, inplace=True)

    # Evidently drift report
    try:
        report = Report(metrics=[
            DatasetDriftMetric(),
            DataDriftTable(),
        ])
        report.run(reference_data=baseline_df, current_data=current_df)
        results = report.as_dict()

        # Extract drift metrics
        metrics = results.get("metrics", [])
        drift_detected = False
        drift_score = 0.0

        for metric in metrics:
            result = metric.get("result", {})
            if "dataset_drift" in result:
                drift_detected = result["dataset_drift"]
            if "drift_share" in result:
                drift_score = result["drift_share"]

        return {
            "status": "computed",
            "drift_detected": drift_detected,
            "drift_score": drift_score,
            "current_samples": len(current_df),
            "baseline_samples": len(baseline_df),
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Evidently drift analysis failed: {e}")
        return _basic_drift_detection()


def _basic_drift_detection() -> dict:
    """Fallback basic drift detection using statistical tests."""
    engine = create_engine(PostgresConfig.get_url())

    # Get recent and baseline risk scores
    recent = pd.read_sql(
        "SELECT risk_score FROM risk_scores WHERE scored_at > NOW() - INTERVAL '7 days'",
        engine,
    )
    baseline = pd.read_sql(
        """SELECT risk_score FROM risk_scores
           WHERE scored_at BETWEEN NOW() - INTERVAL '30 days' AND NOW() - INTERVAL '7 days'""",
        engine,
    )

    if recent.empty or baseline.empty:
        return {"status": "insufficient_data", "drift_detected": False}

    # KS test for distribution shift
    from scipy.stats import ks_2samp
    ks_stat, p_value = ks_2samp(baseline["risk_score"], recent["risk_score"])

    drift_detected = p_value < 0.05
    mean_shift = abs(recent["risk_score"].mean() - baseline["risk_score"].mean())

    return {
        "status": "computed_basic",
        "drift_detected": drift_detected,
        "ks_statistic": float(ks_stat),
        "p_value": float(p_value),
        "mean_shift": float(mean_shift),
        "current_mean": float(recent["risk_score"].mean()),
        "baseline_mean": float(baseline["risk_score"].mean()),
        "timestamp": datetime.now().isoformat(),
    }


def log_drift_result(drift_result: dict):
    """Log drift detection result to PostgreSQL."""
    conn = psycopg2.connect(
        host=PostgresConfig.HOST, port=PostgresConfig.PORT,
        user=PostgresConfig.USER, password=PostgresConfig.PASSWORD,
        dbname=PostgresConfig.DB,
    )
    cursor = conn.cursor()

    action = "none"
    if drift_result.get("drift_detected"):
        action = "retrain_triggered"

    cursor.execute(
        """INSERT INTO drift_logs
           (drift_score, action_taken, details)
           VALUES (%s, %s, %s)""",
        (
            drift_result.get("drift_score", 0),
            action,
            json.dumps(drift_result),
        )
    )

    conn.commit()
    cursor.close()
    conn.close()


def check_and_alert():
    """Run drift detection and trigger retraining if needed."""
    logger.info("[DriftDetector] Running drift detection...")
    result = compute_drift_metrics()
    log_drift_result(result)

    if result.get("drift_detected"):
        logger.warning("[DriftDetector] DRIFT DETECTED! Triggering retraining...")
        return True
    else:
        logger.info("[DriftDetector] No significant drift detected.")
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    check_and_alert()
