"""
Fairness & Bias Auditing
Evaluates model fairness across demographic groups using Fairlearn.
"""
import os
import sys
import numpy as np
import pandas as pd
import logging
from typing import Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

logger = logging.getLogger(__name__)


def compute_fairness_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: pd.DataFrame,
) -> Dict:
    """
    Compute fairness metrics across demographic groups.
    Uses Fairlearn's MetricFrame for group-level analysis.
    """
    try:
        from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference
        from sklearn.metrics import accuracy_score, precision_score, recall_score
    except ImportError:
        logger.warning("Fairlearn not installed. Skipping fairness audit.")
        return {"error": "Fairlearn not installed"}

    results = {}

    for col in sensitive_features.columns:
        sf = sensitive_features[col].values

        metric_frame = MetricFrame(
            metrics={
                "accuracy": accuracy_score,
                "precision": lambda yt, yp: precision_score(yt, yp, zero_division=0),
                "recall": lambda yt, yp: recall_score(yt, yp, zero_division=0),
                "selection_rate": selection_rate,
            },
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sf,
        )

        group_metrics = metric_frame.by_group.to_dict()
        overall_metrics = metric_frame.overall.to_dict()

        try:
            dpd = demographic_parity_difference(y_true, y_pred, sensitive_features=sf)
        except Exception:
            dpd = None

        results[col] = {
            "overall": overall_metrics,
            "by_group": {k: {gk: float(gv) for gk, gv in v.items()} for k, v in group_metrics.items()},
            "demographic_parity_difference": float(dpd) if dpd is not None else None,
            "is_fair": abs(dpd) < 0.1 if dpd is not None else None,
        }

        logger.info(f"[Fairness] {col}: DPD = {dpd:.4f}" if dpd else f"[Fairness] {col}: computed")

    return results


def run_bias_audit(model, X: np.ndarray, y: np.ndarray,
                   demographics: pd.DataFrame) -> Dict:
    """
    Run complete bias audit on a model.
    demographics DataFrame should have columns like 'age_group', 'gender', 'region'.
    """
    y_pred = model.predict(X)

    sensitive_cols = [c for c in demographics.columns
                     if c in ("gender", "region", "income_bracket", "age_group")]

    if not sensitive_cols:
        return {"warning": "No sensitive features found for fairness audit"}

    sensitive_df = demographics[sensitive_cols].reset_index(drop=True)

    # Create age groups if 'age' exists
    if "age" in demographics.columns and "age_group" not in demographics.columns:
        sensitive_df["age_group"] = pd.cut(
            demographics["age"],
            bins=[0, 30, 45, 60, 100],
            labels=["18-30", "31-45", "46-60", "60+"]
        )

    results = compute_fairness_metrics(y, y_pred, sensitive_df)

    # Overall verdict
    all_fair = all(
        r.get("is_fair", True) for r in results.values()
        if isinstance(r, dict) and "is_fair" in r
    )
    results["verdict"] = "PASS" if all_fair else "FAIL - Review required"

    return results
