# pyre-ignore-all-errors
"""
Uplift / Causal ML — P9
T-Learner meta-learner for heterogeneous treatment effect estimation.

Value: Conventional models predict risk; uplift models predict the BENEFIT of intervening.
       A customer with risk=0.80 who would pay without intervention is a wasted call.
       A customer with risk=0.60 who has high uplift is your best target.
       Formula: uplift(x) = P(pay | treated, x) - P(pay | untreated, x)
       Gate interventions: only act if uplift_score > threshold (avoidable defaulters).
"""
import os
import sys
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
logger = logging.getLogger(__name__)


class UpliftModel:
    """
    T-Learner (Two-model) uplift estimator.

    T-Learner trains two separate models:
    - mu1(x): P(positive outcome | treated, x)  — trained on treated group
    - mu0(x): P(positive outcome | control, x)  — trained on control group
    Uplift = mu1(x) - mu0(x)

    Requires A/B data (ab_holdout.py must have run for ≥ 30 days).
    Falls back to a heuristic uplift proxy if no A/B data is available.
    """

    MIN_SAMPLES_REQUIRED = 100  # Each arm: treated and control

    def __init__(self) -> None:
        self.mu1: Any = None   # Treated arm model
        self.mu0: Any = None   # Control arm model
        self._is_fitted: bool = False
        self.feature_names: Optional[List[str]] = None

    def fit(
        self,
        X: Any,
        y: Any,
        treatment_mask: Any,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Convenience method: split X/y by treatment mask and delegate to train().

        Args:
            X:              Full feature matrix
            y:              Full target vector (1=paid, 0=defaulted)
            treatment_mask: Boolean array (True=treated, False=control)
            feature_names:  Optional feature names for interpretability

        Returns:
            dict with AUC per arm, mean uplift, and sample counts
        """
        X_treated = X[treatment_mask]
        y_treated = y[treatment_mask]
        X_control = X[~treatment_mask]
        y_control = y[~treatment_mask]
        names = feature_names if feature_names is not None else []
        return self.train(X_treated, y_treated, X_control, y_control, names)

    def train(
        self,
        X_treated: Any,
        y_treated: Any,
        X_control: Any,
        y_control: Any,
        feature_names: List[str],
    ) -> Dict[str, Any]:
        """
        Train T-Learner on separated treated/control arms.

        Args:
            X_treated:     Features for customers who received intervention
            y_treated:     1=paid, 0=defaulted, for treated group
            X_control:     Features for holdout customers
            y_control:     1=paid, 0=defaulted, for control group
            feature_names: Feature names for interpretability

        Returns:
            dict with AUC per arm and example uplift scores
        """
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.metrics import roc_auc_score

        self.feature_names = feature_names

        if len(X_treated) < self.MIN_SAMPLES_REQUIRED or len(X_control) < self.MIN_SAMPLES_REQUIRED:
            logger.warning(
                f"[Uplift] Insufficient A/B data "
                f"(treated={len(X_treated)}, control={len(X_control)}). "
                f"Need {self.MIN_SAMPLES_REQUIRED} each. Using heuristic uplift."
            )
            self._is_fitted = False
            return {"status": "insufficient_data", "fallback": "heuristic"}

        logger.info(f"[Uplift] Training T-Learner (treated={len(X_treated)}, control={len(X_control)})")

        # Train treated-arm model (mu1)
        self.mu1 = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42
        )
        self.mu1.fit(X_treated, y_treated)
        auc_t = float(roc_auc_score(y_treated, self.mu1.predict_proba(X_treated)[:, 1]))

        # Train control-arm model (mu0)
        self.mu0 = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42
        )
        self.mu0.fit(X_control, y_control)
        auc_c = float(roc_auc_score(y_control, self.mu0.predict_proba(X_control)[:, 1]))

        self._is_fitted = True

        # Sample uplift distribution
        sample = X_treated[:min(100, len(X_treated))]
        sample_uplift = self.predict_uplift(sample)

        logger.info(
            f"[Uplift] Training complete. AUC-treated={auc_t:.4f}, AUC-control={auc_c:.4f}, "
            f"mean_uplift={float(np.mean(sample_uplift)):.4f}"
        )

        return {
            "treated_auc": round(auc_t, 4),
            "control_auc": round(auc_c, 4),
            "mean_uplift": round(float(np.mean(sample_uplift)), 4),
            "n_treated": len(X_treated),
            "n_control": len(X_control),
        }

    def predict_uplift(self, X: Any) -> Any:
        """
        Predict uplift scores for a batch.
        uplift = P(pay | treated) - P(pay | control)
        Positive = intervention helps; Negative = intervention hurts.
        """
        if not self._is_fitted or self.mu1 is None or self.mu0 is None:
            return self._heuristic_uplift(X)

        try:
            prob_treated = self.mu1.predict_proba(X)[:, 1]
            prob_control = self.mu0.predict_proba(X)[:, 1]
            return prob_treated - prob_control
        except Exception as e:
            logger.warning(f"[Uplift] predict_uplift failed: {e}")
            return self._heuristic_uplift(X)

    def predict_uplift_single(self, features: Any) -> float:
        """Predict uplift for a single customer's feature vector."""
        result = self.predict_uplift(features.reshape(1, -1))
        return float(result[0])

    def _heuristic_uplift(self, X: Any) -> Any:
        """
        Fallback heuristic when no A/B data available.
        Assumption: mid-risk customers benefit most from intervention;
        very-high risk (would default anyway) and very-low risk (would pay anyway)
        have lower marginal uplift.
        """
        # Use first feature column as a risk proxy if available
        if X.ndim == 2 and X.shape[1] > 0:
            raw = X[:, 0]  # Assume first column is risk score proxy
            # Inverted parabola peaking at risk=0.6
            uplift = 0.3 - 4 * (raw - 0.6) ** 2
            return np.clip(uplift, -0.1, 0.4)
        return np.full(len(X), 0.10)

    def should_intervene(self, uplift_score: float, threshold: float = 0.05) -> bool:
        """
        Gate intervention decision: only act if expected uplift is positive and meaningful.
        threshold=0.05 means: only intervene if we expect ≥5% increase in payment rate.
        """
        return uplift_score >= threshold

    def save(self, path: str) -> None:
        joblib.dump({
            "mu1": self.mu1, "mu0": self.mu0,
            "feature_names": self.feature_names,
            "is_fitted": self._is_fitted,
        }, path)

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            return
        data = joblib.load(path)
        self.mu1 = data.get("mu1")
        self.mu0 = data.get("mu0")
        self.feature_names = data.get("feature_names")
        self._is_fitted = data.get("is_fitted", False)
        logger.info(f"[Uplift] Model loaded from {path}")
