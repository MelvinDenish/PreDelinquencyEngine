# pyre-ignore-all-errors
"""
Probability Calibration — IFRS 9 Compatible PD Estimates
Transforms raw ensemble risk scores into calibrated probabilities using
isotonic regression. When the model says 0.65, approximately 65% of
customers with that score actually default.

Value: Raw model scores are relative rankings, not probabilities.
       IFRS 9 requires calibrated PD (Probability of Default) estimates
       for expected credit loss calculations. This module ensures
       risk scores are meaningful to Relationship Managers and
       compliant with Basel III / FCA PRA SS1/23 model risk standards.
"""
import os
import logging
import numpy as np
import joblib
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

logger = logging.getLogger(__name__)


class ProbabilityCalibrator:
    """Isotonic regression calibrator for ensemble risk scores.

    Isotonic regression fits a non-parametric monotonic function that maps
    raw scores to calibrated probabilities. More flexible than Platt scaling
    (logistic sigmoid) for non-linear distortions.

    Usage:
        1. Train ensemble models on training set
        2. Get ensemble scores on CALIBRATION set (held-out, not train or test)
        3. Fit calibrator: calibrator.fit(raw_scores, true_labels)
        4. At inference: calibrator.calibrate(raw_score) → calibrated PD
    """

    def __init__(self):
        self.calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
        self._is_fitted = False

    def fit(self, raw_scores: np.ndarray, true_labels: np.ndarray) -> dict:
        """Fit calibrator on held-out calibration set.

        Args:
            raw_scores: Raw ensemble probabilities (uncalibrated)
            true_labels: Ground truth binary labels (0/1)

        Returns:
            dict with Brier scores before/after calibration
        """
        self.calibrator.fit(raw_scores, true_labels)
        self._is_fitted = True

        calibrated = self.calibrator.predict(raw_scores)
        brier_before = brier_score_loss(true_labels, raw_scores)
        brier_after = brier_score_loss(true_labels, calibrated)

        logger.info(
            f"[Calibrator] Brier score: {brier_before:.4f} → {brier_after:.4f} "
            f"(improvement: {(brier_before - brier_after) / max(brier_before, 1e-8) * 100:.1f}%)"
        )
        return {
            "brier_before": round(float(brier_before), 4),
            "brier_after": round(float(brier_after), 4),
            "improvement_pct": round(float((brier_before - brier_after) / max(brier_before, 1e-8) * 100), 1),
            "n_calibration": len(raw_scores),
        }

    def calibrate(self, raw_score: float) -> float:
        """Transform single raw score to calibrated probability."""
        if not self._is_fitted:
            return raw_score
        return float(self.calibrator.predict([raw_score])[0])

    def calibrate_batch(self, raw_scores: np.ndarray) -> np.ndarray:
        """Transform batch of raw scores to calibrated probabilities."""
        if not self._is_fitted:
            return raw_scores
        return self.calibrator.predict(raw_scores)

    def save(self, path: str):
        """Persist calibrator to disk."""
        joblib.dump({
            "calibrator": self.calibrator,
            "is_fitted": self._is_fitted,
        }, path)
        logger.info(f"[Calibrator] Saved to {path}")

    def load(self, path: str):
        """Load calibrator from disk."""
        if not os.path.exists(path):
            logger.warning(f"[Calibrator] File not found: {path}")
            return
        data = joblib.load(path)
        self.calibrator = data["calibrator"]
        self._is_fitted = data["is_fitted"]
        logger.info(f"[Calibrator] Loaded from {path}")
