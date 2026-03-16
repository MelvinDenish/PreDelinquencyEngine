"""
Ensemble Scorer
Combines XGBoost (tabular) and LSTM (temporal) predictions
into a unified delinquency risk score.
"""
import os
import sys
import numpy as np
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config.settings import ModelConfig

logger = logging.getLogger(__name__)


class EnsembleScorer:
    """
    Weighted ensemble that combines XGBoost and LSTM predictions.
    Default weights: 0.6 XGBoost + 0.4 LSTM (as per project spec).
    """

    def __init__(self, xgboost_weight: float = None, lstm_weight: float = None):
        self.xgboost_weight = xgboost_weight or ModelConfig.XGBOOST_WEIGHT
        self.lstm_weight = lstm_weight or ModelConfig.LSTM_WEIGHT

        # Ensure weights sum to 1
        total = self.xgboost_weight + self.lstm_weight
        self.xgboost_weight /= total
        self.lstm_weight /= total

    def combine(self, xgboost_prob: float, lstm_prob: float = None) -> float:
        """
        Combine XGBoost and LSTM probabilities.
        If LSTM is not available, fall back to XGBoost only.
        """
        if lstm_prob is None:
            return float(xgboost_prob)

        ensemble_score = (
            self.xgboost_weight * xgboost_prob +
            self.lstm_weight * lstm_prob
        )
        return float(np.clip(ensemble_score, 0, 1))

    def combine_batch(self, xgboost_probs: np.ndarray,
                      lstm_probs: np.ndarray = None) -> np.ndarray:
        """Combine batch predictions."""
        if lstm_probs is None:
            return xgboost_probs.astype(float)

        # Align lengths if different
        min_len = min(len(xgboost_probs), len(lstm_probs))
        xg = xgboost_probs[:min_len]
        ls = lstm_probs[:min_len]

        ensemble = self.xgboost_weight * xg + self.lstm_weight * ls
        return np.clip(ensemble, 0, 1).astype(float)

    @staticmethod
    def score_to_risk_tier(score: float) -> str:
        """Map ensemble score to risk tier."""
        if score >= ModelConfig.RISK_CRITICAL_THRESHOLD:
            return "critical"
        elif score >= ModelConfig.RISK_WATCH_THRESHOLD:
            return "watch"
        else:
            return "stable"

    @staticmethod
    def score_to_credit_score(probability: float) -> int:
        """
        Map delinquency probability to credit score.
        Formula: 850 - probability * 550 (as per project spec).
        """
        credit_score = int(850 - probability * 550)
        return max(300, min(850, credit_score))
