"""
Ensemble Scorer
Combines XGBoost, LightGBM, and LSTM predictions into a single risk score.
Uses weighted averaging with configurable weights for each model.
"""
import os
import sys
import numpy as np
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config.settings import ModelConfig

logger = logging.getLogger(__name__)


class EnsembleScorer:
    """3-model ensemble: XGBoost + LightGBM + LSTM."""

    def __init__(self, xgb_weight: float = None, lgb_weight: float = None,
                 lstm_weight: float = None):
        """
        Initialize ensemble with configurable weights.
        Default: XGBoost 0.40, LightGBM 0.30, LSTM 0.30
        Falls back to 2-model or 1-model if components are missing.
        """
        self.xgb_weight = xgb_weight or ModelConfig.ENSEMBLE_XGB_WEIGHT
        self.lgb_weight = lgb_weight or getattr(ModelConfig, 'ENSEMBLE_LGB_WEIGHT', 0.30)
        self.lstm_weight = lstm_weight or ModelConfig.ENSEMBLE_LSTM_WEIGHT

    def combine(self, xgb_prob: float = None, lgb_prob: float = None,
                lstm_prob: float = None) -> float:
        """
        Combine model predictions using weighted average.
        Handles missing models by redistributing weights.
        """
        scores = []
        weights = []

        if xgb_prob is not None:
            scores.append(xgb_prob)
            weights.append(self.xgb_weight)

        if lgb_prob is not None:
            scores.append(lgb_prob)
            weights.append(self.lgb_weight)

        if lstm_prob is not None:
            scores.append(lstm_prob)
            weights.append(self.lstm_weight)

        if not scores:
            return 0.5  # Default when no models available

        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        ensemble_score = sum(s * w for s, w in zip(scores, normalized_weights))
        return float(np.clip(ensemble_score, 0, 1))

    def combine_batch(self, xgb_probs: np.ndarray = None,
                      lgb_probs: np.ndarray = None,
                      lstm_probs: np.ndarray = None) -> np.ndarray:
        """Combine batch predictions."""
        available = []
        weights = []

        if xgb_probs is not None:
            available.append(xgb_probs)
            weights.append(self.xgb_weight)

        if lgb_probs is not None:
            available.append(lgb_probs)
            weights.append(self.lgb_weight)

        if lstm_probs is not None:
            available.append(lstm_probs)
            weights.append(self.lstm_weight)

        if not available:
            return np.full(len(xgb_probs or lgb_probs or lstm_probs), 0.5)

        total = sum(weights)
        result = sum(p * (w / total) for p, w in zip(available, weights))
        return np.clip(result, 0, 1)

    @staticmethod
    def score_to_risk_tier(score: float) -> str:
        """Map ensemble score to risk tier."""
        if score >= 0.7:
            return "critical"
        elif score >= 0.5:
            return "watch"
        return "stable"

    @staticmethod
    def score_to_credit_score(score: float) -> int:
        """Map ensemble risk score to credit score equivalent (300-900)."""
        return int(900 - (score * 600))

    def get_model_contributions(self, xgb_prob: float = None,
                                 lgb_prob: float = None,
                                 lstm_prob: float = None) -> dict:
        """Return individual model contributions to the ensemble."""
        ensemble = self.combine(xgb_prob, lgb_prob, lstm_prob)
        contributions = {}

        if xgb_prob is not None:
            contributions["xgboost"] = {
                "raw_score": float(xgb_prob),
                "weight": self.xgb_weight,
                "contribution": float(xgb_prob * self.xgb_weight),
            }
        if lgb_prob is not None:
            contributions["lightgbm"] = {
                "raw_score": float(lgb_prob),
                "weight": self.lgb_weight,
                "contribution": float(lgb_prob * self.lgb_weight),
            }
        if lstm_prob is not None:
            contributions["lstm"] = {
                "raw_score": float(lstm_prob),
                "weight": self.lstm_weight,
                "contribution": float(lstm_prob * self.lstm_weight),
            }

        contributions["ensemble_score"] = float(ensemble)
        contributions["risk_tier"] = self.score_to_risk_tier(ensemble)
        return contributions
