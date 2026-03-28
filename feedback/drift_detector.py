# pyre-ignore-all-errors
"""
Drift Detection with Evidently AI + Concept Drift Monitoring
Monitors model performance and detects data/prediction drift.
Concept drift tracks whether the relationship between features and
outcomes is changing (e.g., model AUC degrading over time).
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
        """SELECT risk_score, xgboost_score, ensemble_score
           FROM risk_scores
           WHERE scored_at > NOW() - INTERVAL '7 days'""",
        engine,
    )

    # Get baseline scores (8-30 days ago)
    baseline_df = pd.read_sql(
        """SELECT risk_score, xgboost_score, ensemble_score
           FROM risk_scores
           WHERE scored_at BETWEEN NOW() - INTERVAL '30 days'
                 AND NOW() - INTERVAL '7 days'""",
        engine,
    )

    if current_df.empty or baseline_df.empty:
        return {"status": "insufficient_data", "drift_detected": False}

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


def compute_concept_drift(auc_drop_threshold: float = 0.05) -> dict:
    """
    Detect concept drift by comparing recent model performance against baseline.

    Concept drift = the relationship between features and outcomes changes,
    even if the feature distributions stay the same. Detected by tracking
    rolling AUC on recent predictions with known outcomes (from feedback_events).

    Args:
        auc_drop_threshold: Flag concept drift if AUC drops by more than this
                            relative to the baseline (default 5%)

    Returns:
        dict with concept drift status, rolling AUC, baseline AUC, and delta
    """
    from sklearn.metrics import roc_auc_score

    engine = create_engine(PostgresConfig.get_url())

    # Get recent predictions that have known outcomes (from feedback_events)
    # feedback_events records whether a scored customer actually defaulted
    recent_df = pd.read_sql("""
        SELECT rs.risk_score AS predicted_score,
               CASE WHEN fe.outcome = 'defaulted' THEN 1 ELSE 0 END AS actual_label
        FROM risk_scores rs
        INNER JOIN feedback_events fe
            ON rs.customer_id = fe.customer_id
        WHERE rs.scored_at > NOW() - INTERVAL '14 days'
          AND fe.created_at > NOW() - INTERVAL '14 days'
    """, engine)

    # Get baseline performance from model_registry (set during training)
    baseline_df = pd.read_sql("""
        SELECT metrics->>'test_auc' AS baseline_auc
        FROM model_registry
        WHERE model_name = 'ensemble'
        ORDER BY registered_at DESC
        LIMIT 1
    """, engine)

    # Fallback baseline AUC if model_registry doesn't have it
    baseline_auc = 0.85
    if not baseline_df.empty and baseline_df.iloc[0]["baseline_auc"]:
        try:
            baseline_auc = float(baseline_df.iloc[0]["baseline_auc"])
        except (ValueError, TypeError):
            pass

    if recent_df.empty or len(recent_df) < 30:
        return {
            "status": "insufficient_data",
            "concept_drift_detected": False,
            "message": f"Need ≥30 scored+outcome pairs, have {len(recent_df)}",
            "timestamp": datetime.now().isoformat(),
        }

    n_pos = int(recent_df["actual_label"].sum())
    if n_pos == 0 or n_pos == len(recent_df):
        return {
            "status": "insufficient_class_variance",
            "concept_drift_detected": False,
            "message": f"All labels are {'positive' if n_pos > 0 else 'negative'} — cannot compute AUC",
            "timestamp": datetime.now().isoformat(),
        }

    rolling_auc = roc_auc_score(recent_df["actual_label"], recent_df["predicted_score"])
    auc_delta = baseline_auc - rolling_auc
    concept_drift_detected = auc_delta > auc_drop_threshold

    result = {
        "status": "computed",
        "concept_drift_detected": concept_drift_detected,
        "rolling_auc": round(float(rolling_auc), 4),
        "baseline_auc": round(float(baseline_auc), 4),
        "auc_delta": round(float(auc_delta), 4),
        "auc_drop_threshold": auc_drop_threshold,
        "n_samples": len(recent_df),
        "n_positive": n_pos,
        "timestamp": datetime.now().isoformat(),
    }

    if concept_drift_detected:
        logger.warning(
            f"[DriftDetector] CONCEPT DRIFT: AUC dropped {auc_delta:.3f} "
            f"({baseline_auc:.3f} → {rolling_auc:.3f}), threshold={auc_drop_threshold}"
        )
    else:
        logger.info(
            f"[DriftDetector] No concept drift: AUC {rolling_auc:.3f} "
            f"(baseline {baseline_auc:.3f}, delta {auc_delta:.3f})"
        )

    return result


def check_and_alert():
    """Run data drift + concept drift detection and trigger retraining if needed."""
    logger.info("[DriftDetector] Running drift detection...")

    # Data/prediction drift (distribution shifts)
    data_drift = compute_drift_metrics()
    log_drift_result(data_drift)

    # Concept drift (model performance degradation)
    concept_drift = compute_concept_drift()
    log_drift_result(concept_drift)

    data_drifted = data_drift.get("drift_detected", False)
    concept_drifted = concept_drift.get("concept_drift_detected", False)

    if concept_drifted:
        logger.warning("[DriftDetector] CONCEPT DRIFT DETECTED — model performance degraded. Retraining recommended.")
    if data_drifted:
        logger.warning("[DriftDetector] DATA DRIFT DETECTED — input distributions shifted.")

    if data_drifted or concept_drifted:
        return True
    else:
        logger.info("[DriftDetector] No significant drift detected.")
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    check_and_alert()
