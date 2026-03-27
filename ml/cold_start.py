# pyre-ignore-all-errors
"""
Cold-Start Scorer
Handles customers with insufficient transaction history (< 30 days)
for TFT/LSTM temporal models. Uses a simplified XGBoost-only score
on available demographic + available streaming features.
"""
import logging
import numpy as np
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class ColdStartScorer:
    """
    Handles customers with insufficient history for temporal models.
    Uses a simplified XGBoost-only score on available features.
    """

    MIN_HISTORY_DAYS = 30
    MAX_RISK_TIER = "watch"  # Cold-start customers capped at watch, never critical

    # Features usable even with limited history
    COLD_START_FEATURES = [
        "credit_score",
        "age",
        "tenure_months",
        "product_count",
        "has_credit_card",
        "has_personal_loan",
        "has_mortgage",
        "income_bracket",
    ]

    def is_cold_start(self, available_days: int) -> bool:
        """Check if a customer has insufficient history."""
        return available_days < self.MIN_HISTORY_DAYS

    def get_confidence_flag(self, available_days: int) -> str:
        """Return confidence level based on history length."""
        if available_days < 7:
            return "cold_start"
        elif available_days < self.MIN_HISTORY_DAYS:
            return "limited_history"
        return "full"

    def score_cold_start(self, features: dict, xgb_model=None) -> dict:
        """
        Score a cold-start customer using only available features.

        Returns:
            dict with risk_score, risk_tier, confidence_flag
        """
        # Build feature vector from available features only
        available_features = {}
        for feat in self.COLD_START_FEATURES:
            if feat in features and features[feat] is not None:
                available_features[feat] = features[feat]

        # If we have an XGBoost model, use it for scoring
        if xgb_model is not None:
            try:
                # Create full feature vector with missing features set to 0
                full_features = features.copy()
                for key in full_features:
                    if full_features[key] is None:
                        full_features[key] = 0
                risk_score = float(xgb_model.predict_proba(
                    np.array([list(full_features.values())])
                )[0])
            except Exception as e:
                logger.warning(f"[ColdStart] XGBoost scoring failed: {e}")
                risk_score = self._heuristic_score(available_features)
        else:
            risk_score = self._heuristic_score(available_features)

        # Cap at watch tier — never escalate cold-start to critical
        if risk_score > 0.70:
            risk_score = min(risk_score, 0.699)

        risk_tier = self._determine_tier(risk_score)

        return {
            "risk_score": round(risk_score, 4),
            "risk_tier": risk_tier,
            "confidence_flag": "cold_start",
            "xgboost_score": round(risk_score, 4),
            "tft_score": None,
            "lstm_score": None,
            "meta_learner_used": False,
        }

    def _heuristic_score(self, features: dict) -> float:
        """
        Simple rule-based scoring when no model is available.
        Uses credit score, DTI, and product holdings.
        """
        score = 0.3  # Base risk

        credit_score = features.get("credit_score", 700)
        if credit_score < 550:
            score += 0.25
        elif credit_score < 650:
            score += 0.15
        elif credit_score < 700:
            score += 0.05
        elif credit_score > 800:
            score -= 0.10

        # Loan load
        if features.get("has_personal_loan", False):
            score += 0.05
        if features.get("product_count", 1) > 5:
            score += 0.05

        # Tenure stability
        tenure = features.get("tenure_months", 12)
        if tenure < 6:
            score += 0.10
        elif tenure > 60:
            score -= 0.05

        return max(0.01, min(0.699, score))

    def _determine_tier(self, risk_score: float) -> str:
        # Cold-start customers: cap at watch
        if risk_score >= 0.50:
            return "watch"
        return "stable"
