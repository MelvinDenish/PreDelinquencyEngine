# pyre-ignore-all-errors
"""
Cost-Sensitive Threshold Optimization
Finds optimal risk tier thresholds by minimizing total business cost
rather than using arbitrary cutoffs (e.g., 0.7/0.4).

Value: Missing a delinquent customer costs ~£10,000 (loan loss).
       An unnecessary RM call costs ~£5. The optimal threshold
       should be heavily biased toward catching delinquents.
       This module finds that business-optimal operating point.
"""
import logging
import numpy as np
import joblib
from typing import Dict

logger = logging.getLogger(__name__)


def _find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray,
                             cost_fn: float, cost_fp: float) -> float:
    """Find single threshold minimizing total cost.

    Args:
        y_true: Ground truth binary labels
        y_proba: Calibrated probabilities
        cost_fn: Cost of false negative (missed delinquent)
        cost_fp: Cost of false positive (unnecessary intervention)
    """
    best_threshold, best_cost = 0.5, float('inf')

    for threshold in np.arange(0.01, 0.95, 0.01):
        preds = (y_proba >= threshold).astype(int)
        fn = ((preds == 0) & (y_true == 1)).sum()
        fp = ((preds == 1) & (y_true == 0)).sum()
        total_cost = fn * cost_fn + fp * cost_fp
        if total_cost < best_cost:
            best_cost = total_cost
            best_threshold = float(threshold)

    return round(best_threshold, 2)


def optimize_thresholds(y_true: np.ndarray, y_proba: np.ndarray,
                         cost_missed_default: float = 10000.0,
                         cost_unnecessary_call: float = 5.0,
                         cost_unnecessary_sms: float = 0.50) -> Dict:
    """Find optimal critical and watch thresholds.

    Two thresholds:
    - Critical: triggers RM call / restructuring offer (high intervention cost)
    - Watch: triggers automated SMS/push nudge (low intervention cost)

    Args:
        y_true: Ground truth binary labels
        y_proba: Calibrated probabilities
        cost_missed_default: Cost of missing a delinquent customer (avg loan loss)
        cost_unnecessary_call: Cost of unnecessary RM call
        cost_unnecessary_sms: Cost of unnecessary automated SMS

    Returns:
        dict with critical_threshold, watch_threshold, and cost metrics
    """
    # Critical threshold: RM call costs £5, missing default costs £10,000
    critical = _find_optimal_threshold(y_true, y_proba, cost_missed_default, cost_unnecessary_call)

    # Watch threshold: SMS costs £0.50, missing default still costs £10,000
    watch = _find_optimal_threshold(y_true, y_proba, cost_missed_default, cost_unnecessary_sms)

    # Ensure watch <= critical (watch is a looser trigger)
    if watch >= critical:
        watch = round(critical * 0.65, 2)

    # Compute metrics at optimal thresholds
    critical_preds = (y_proba >= critical).astype(int)
    watch_preds = (y_proba >= watch).astype(int)

    n_pos = int(y_true.sum())
    n_total = len(y_true)

    critical_tp = int(((critical_preds == 1) & (y_true == 1)).sum())
    critical_fp = int(((critical_preds == 1) & (y_true == 0)).sum())
    watch_tp = int(((watch_preds == 1) & (y_true == 1)).sum())
    watch_fp = int(((watch_preds == 1) & (y_true == 0)).sum())

    result = {
        "critical_threshold": critical,
        "watch_threshold": watch,
        "cost_assumptions": {
            "missed_default": cost_missed_default,
            "unnecessary_call": cost_unnecessary_call,
            "unnecessary_sms": cost_unnecessary_sms,
        },
        "critical_recall": round(critical_tp / max(n_pos, 1), 3),
        "critical_flagged": critical_tp + critical_fp,
        "watch_recall": round(watch_tp / max(n_pos, 1), 3),
        "watch_flagged": watch_tp + watch_fp,
        "n_delinquent": n_pos,
        "n_total": n_total,
    }

    logger.info(
        f"[ThresholdOptimizer] Critical={critical:.2f} (recall={result['critical_recall']:.1%}), "
        f"Watch={watch:.2f} (recall={result['watch_recall']:.1%})"
    )
    return result


def save_thresholds(thresholds: Dict, path: str):
    """Persist optimized thresholds to disk."""
    joblib.dump(thresholds, path)
    logger.info(f"[ThresholdOptimizer] Saved to {path}")


def load_thresholds(path: str) -> Dict:
    """Load optimized thresholds from disk."""
    import os
    if not os.path.exists(path):
        return None
    return joblib.load(path)
