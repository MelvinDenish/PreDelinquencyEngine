# pyre-ignore-all-errors
"""
Conformal Prediction Intervals — P6
Split conformal calibration for uncertainty quantification.

Value: Every risk score now has calibrated bounds (e.g., 0.72 ± 0.09).
       RMs know when to trust the model and when to use human judgment.
       High uncertainty → escalate to RM regardless of tier.
       Narrow interval → high confidence, safe to automate.
"""
import os
import sys
import logging
import numpy as np
import joblib
from typing import Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
logger = logging.getLogger(__name__)


class ConformalPredictor:
    """
    Split conformal prediction for risk score intervals.

    Method: Inductive (split) conformal — calibrate nonconformity scores
    on a held-out calibration set. No assumptions on model internals.
    Works with any scoring function.

    Output: (lower_bound, upper_bound) at a given confidence level.
    """

    def __init__(self, confidence: float = 0.90):
        """
        Args:
            confidence: Target coverage level (0.90 = 90% coverage guaranteed).
        """
        self.confidence = confidence
        self.quantile_threshold = None  # q_hat
        self._is_calibrated = False

    def calibrate(self, true_labels: np.ndarray, predicted_proba: np.ndarray) -> dict:
        """
        Calibrate on a held-out set.

        Args:
            true_labels:     Ground-truth binary labels (0/1)
            predicted_proba: Model's predicted probabilities
        Returns:
            dict with empirical coverage and threshold
        """
        n = len(true_labels)
        if n < 30:
            logger.warning("[Conformal] Calibration set too small (<30). Using fixed 0.15 margin.")
            self.quantile_threshold = 0.15
            self._is_calibrated = True
            return {"threshold": 0.15, "empirical_coverage": None}

        # Nonconformity scores: |label - predicted_prob|
        nonconformity = np.abs(true_labels.astype(float) - predicted_proba)

        # Conformal quantile at (1 - alpha) * (1 + 1/n) level
        alpha = 1.0 - self.confidence
        level = np.ceil((1 - alpha) * (n + 1)) / n
        level = min(level, 1.0)
        self.quantile_threshold = float(np.quantile(nonconformity, level))
        self._is_calibrated = True

        # Empirical coverage check
        covered = np.mean(nonconformity <= self.quantile_threshold)

        logger.info(
            f"[Conformal] Calibrated. q_hat={self.quantile_threshold:.4f}, "
            f"empirical_coverage={covered:.3f} (target={self.confidence})"
        )
        return {
            "threshold": self.quantile_threshold,
            "empirical_coverage": float(covered),
            "target_coverage": self.confidence,
            "n_calibration": n,
        }

    def predict_interval(self, predicted_proba: float) -> Tuple[float, float]:
        """
        Compute prediction interval for a single predicted probability.

        Returns:
            (lower, upper) — both clipped to [0, 1]
        """
        if not self._is_calibrated:
            q = 0.15  # default margin
        else:
            q = self.quantile_threshold

        lower = max(0.0, predicted_proba - q)
        upper = min(1.0, predicted_proba + q)
        return round(lower, 4), round(upper, 4)

    def predict_intervals_batch(self, predicted_probas: np.ndarray) -> np.ndarray:
        """
        Batch prediction intervals.
        Returns array of shape (n, 2) — columns: [lower, upper]
        """
        if not self._is_calibrated:
            q = 0.15
        else:
            q = self.quantile_threshold

        lower = np.clip(predicted_probas - q, 0.0, 1.0)
        upper = np.clip(predicted_probas + q, 0.0, 1.0)
        return np.stack([lower, upper], axis=1)

    def uncertainty_flag(self, lower: float, upper: float) -> str:
        """
        Classify uncertainty level from interval width.
        Returns: 'low', 'medium', or 'high'
        """
        width = upper - lower
        if width < 0.10:
            return "low"        # High confidence
        elif width < 0.20:
            return "medium"     # Moderate confidence
        else:
            return "high"       # Human review recommended

    def save(self, path: str):
        joblib.dump({
            "confidence": self.confidence,
            "quantile_threshold": self.quantile_threshold,
            "is_calibrated": self._is_calibrated,
        }, path)
        logger.info(f"[Conformal] Saved to {path}")

    def load(self, path: str):
        if not os.path.exists(path):
            return
        data = joblib.load(path)
        self.confidence = data["confidence"]
        self.quantile_threshold = data["quantile_threshold"]
        self._is_calibrated = data["is_calibrated"]
        logger.info(f"[Conformal] Loaded from {path}")
