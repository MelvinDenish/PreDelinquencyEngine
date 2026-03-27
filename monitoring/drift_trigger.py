# pyre-ignore-all-errors
"""
PSI Drift Trigger (M9)
Monitors Population Stability Index (PSI) for feature drift.
Triggers automated retraining when drift exceeds threshold.
"""
import logging
import numpy as np
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

PSI_THRESHOLD = 0.20  # PSI > 0.20 = significant drift → trigger retrain
PSI_WARNING = 0.10    # PSI > 0.10 = moderate shift → log warning


def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Compute Population Stability Index between two distributions.

    PSI < 0.10 → No significant shift
    PSI 0.10-0.20 → Moderate shift, monitor
    PSI > 0.20 → Significant shift, retrain needed

    Args:
        expected: baseline distribution (training data)
        actual: current distribution (scoring data)
        bins: number of histogram bins

    Returns:
        PSI value (float)
    """
    # Create bins from expected distribution
    breakpoints = np.quantile(expected, np.linspace(0, 1, bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    # Normalise to proportions (add small epsilon to avoid log(0))
    eps = 1e-6
    expected_pct = expected_counts / max(len(expected), 1) + eps
    actual_pct = actual_counts / max(len(actual), 1) + eps

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def compute_feature_drift(baseline_features: Dict[str, np.ndarray],
                           current_features: Dict[str, np.ndarray]) -> Dict[str, dict]:
    """
    Compute PSI for each feature column.

    Returns:
        Dict[feature_name → {psi, status, needs_retrain}]
    """
    results = {}

    for feature_name in baseline_features:
        if feature_name not in current_features:
            continue

        baseline = baseline_features[feature_name]
        current = current_features[feature_name]

        if len(baseline) < 50 or len(current) < 50:
            continue

        psi = compute_psi(baseline, current)

        if psi > PSI_THRESHOLD:
            status = "CRITICAL"
            needs_retrain = True
        elif psi > PSI_WARNING:
            status = "WARNING"
            needs_retrain = False
        else:
            status = "STABLE"
            needs_retrain = False

        results[feature_name] = {
            "psi": round(psi, 4),
            "status": status,
            "needs_retrain": needs_retrain,
        }

    return results


def check_drift_and_trigger(baseline_features: Dict[str, np.ndarray],
                              current_features: Dict[str, np.ndarray]) -> dict:
    """
    Main drift check entry point. Returns drift report and trigger decision.

    Returns:
        {
            "trigger_retrain": bool,
            "drifted_features": list,
            "feature_drift": dict,
            "overall_psi_avg": float,
        }
    """
    drift_report = compute_feature_drift(baseline_features, current_features)

    drifted = [f for f, d in drift_report.items() if d["needs_retrain"]]
    all_psi = [d["psi"] for d in drift_report.values()]
    avg_psi = np.mean(all_psi) if all_psi else 0.0

    trigger = len(drifted) >= 3 or avg_psi > PSI_THRESHOLD

    if trigger:
        logger.warning(
            f"[DriftTrigger] RETRAIN TRIGGERED — "
            f"{len(drifted)} features drifted (avg PSI: {avg_psi:.4f}): "
            f"{', '.join(drifted[:5])}"
        )
    elif drifted:
        logger.info(
            f"[DriftTrigger] Warning — {len(drifted)} features drifted "
            f"but below retrain threshold"
        )
    else:
        logger.info(f"[DriftTrigger] All features stable (avg PSI: {avg_psi:.4f})")

    return {
        "trigger_retrain": trigger,
        "drifted_features": drifted,
        "feature_drift": drift_report,
        "overall_psi_avg": round(avg_psi, 4),
    }
