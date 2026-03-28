# pyre-ignore-all-errors
"""
Ensemble Scorer — StackingEnsemble + Fixed-Weight Fallback
Combines XGBoost, LightGBM, and TFT predictions (3-model ensemble).
LSTM removed — TFT (Temporal Fusion Transformer) is the superior temporal model.
Supports meta-learner stacking (LogisticRegression) with OOF training
or fixed-weight fallback if meta-learner is unavailable.
"""
import os
import sys
import joblib
import numpy as np
import logging
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config.settings import ModelConfig

logger = logging.getLogger(__name__)

# Load optimized thresholds if available (from cost-sensitive threshold tuning)
_OPTIMIZED_THRESHOLDS = None
_THRESHOLDS_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'thresholds.joblib')
if os.path.exists(_THRESHOLDS_PATH):
    try:
        _OPTIMIZED_THRESHOLDS = joblib.load(_THRESHOLDS_PATH)
        logger.info(
            f"[Ensemble] Loaded optimized thresholds: "
            f"critical={_OPTIMIZED_THRESHOLDS['critical_threshold']}, "
            f"watch={_OPTIMIZED_THRESHOLDS['watch_threshold']}"
        )
    except Exception as e:
        logger.warning(f"[Ensemble] Failed to load thresholds: {e}")


class EnsembleScorer:
    """3-model ensemble: XGBoost + LightGBM + TFT (fixed weights)."""

    def __init__(self, xgb_weight: float = None, lgb_weight: float = None,
                 tft_weight: float = None):
        """
        Initialize ensemble with configurable weights.
        Default: XGBoost 0.35, LightGBM 0.40, TFT 0.25
        Falls back to fewer models if components are missing.
        """
        self.xgb_weight = xgb_weight or ModelConfig.ENSEMBLE_XGB_WEIGHT
        self.lgb_weight = lgb_weight or getattr(ModelConfig, 'ENSEMBLE_LGB_WEIGHT', 0.40)
        self.tft_weight = tft_weight or getattr(ModelConfig, 'ENSEMBLE_TFT_WEIGHT', 0.25)

    def combine(self, xgb_prob: float = None, lgb_prob: float = None,
                tft_prob: float = None, **kwargs) -> float:
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

        if tft_prob is not None:
            scores.append(tft_prob)
            weights.append(self.tft_weight)

        if not scores:
            return 0.5  # Default when no models available

        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        ensemble_score = sum(s * w for s, w in zip(scores, normalized_weights))
        return float(np.clip(ensemble_score, 0, 1))

    def combine_batch(self, xgb_probs: np.ndarray = None,
                      lgb_probs: np.ndarray = None,
                      tft_probs: np.ndarray = None, **kwargs) -> np.ndarray:
        """Combine batch predictions."""
        available = []
        weights = []

        if xgb_probs is not None:
            available.append(xgb_probs)
            weights.append(self.xgb_weight)

        if lgb_probs is not None:
            available.append(lgb_probs)
            weights.append(self.lgb_weight)

        if tft_probs is not None:
            available.append(tft_probs)
            weights.append(self.tft_weight)

        if not available:
            return np.full(1, 0.5)

        total = sum(weights)
        result = sum(p * (w / total) for p, w in zip(available, weights))
        return np.clip(result, 0, 1)

    @staticmethod
    def score_to_risk_tier(score: float, segment_type: str = None,
                           segment_thresholds: dict = None) -> str:
        """Map ensemble score to risk tier using cost-optimized thresholds.

        Priority: segment_thresholds > optimized thresholds > config defaults.
        """
        # Default from config
        watch = ModelConfig.RISK_WATCH_THRESHOLD
        critical = ModelConfig.RISK_CRITICAL_THRESHOLD

        # Override with cost-optimized thresholds if available
        if _OPTIMIZED_THRESHOLDS:
            critical = _OPTIMIZED_THRESHOLDS["critical_threshold"]
            watch = _OPTIMIZED_THRESHOLDS["watch_threshold"]

        # Segment-level overrides take highest priority
        if segment_thresholds:
            watch = segment_thresholds.get("watch", watch)
            critical = segment_thresholds.get("critical", critical)

        if score >= critical:
            return "critical"
        elif score >= watch:
            return "watch"
        return "stable"

    @staticmethod
    def score_to_credit_score(score: float) -> int:
        """Map ensemble risk score to credit score equivalent (300-900)."""
        return int(900 - (score * 600))

    def get_model_contributions(self, xgb_prob: float = None,
                                 lgb_prob: float = None,
                                 tft_prob: float = None, **kwargs) -> dict:
        """Return individual model contributions to the ensemble."""
        ensemble = self.combine(xgb_prob, lgb_prob, tft_prob)
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
        if tft_prob is not None:
            contributions["tft"] = {
                "raw_score": float(tft_prob),
                "weight": self.tft_weight,
                "contribution": float(tft_prob * self.tft_weight),
            }

        contributions["ensemble_score"] = float(ensemble)
        contributions["risk_tier"] = self.score_to_risk_tier(ensemble)
        return contributions


# ═══════════════════════════════════════════════════════
# M4: Meta-Learner Stacking Ensemble
# ═══════════════════════════════════════════════════════

class StackingEnsemble:
    """
    Meta-learner stacking ensemble (M4).
    Trains a LogisticRegression on out-of-fold predictions from
    XGBoost, LightGBM, TFT plus customer meta-features.

    Meta-feature vector (7 inputs):
        [xgb_score, lgb_score, tft_score,
         income_bracket_encoded, segment_type_encoded,
         tenure_months_normalised, credit_score_normalised]
    """

    # Encoding maps for categorical meta-features
    INCOME_BRACKET_MAP = {
        "ews": 0, "low": 1, "lower_middle": 2, "middle": 3,
        "upper_middle": 4, "high": 5, "ultra_high": 6,
    }
    SEGMENT_TYPE_MAP = {
        "salaried": 0, "self_employed": 1, "gig_worker": 2,
        "retiree": 3, "agricultural": 4, "nri": 5, "student": 6,
    }

    def __init__(self, meta_learner_path: str = None):
        self.meta_learner = None
        self.fallback_ensemble = EnsembleScorer()
        if meta_learner_path and os.path.exists(meta_learner_path):
            self.load_meta_learner(meta_learner_path)

    def build_meta_features(self, xgb_prob: float = None, lgb_prob: float = None,
                             tft_prob: float = None,
                             income_bracket: str = "middle",
                             segment_type: str = "salaried",
                             tenure_months: int = 36,
                             credit_score: int = 700, **kwargs) -> np.ndarray:
        """Construct the 7-feature meta-input vector."""
        return np.array([
            xgb_prob if xgb_prob is not None else 0.5,
            lgb_prob if lgb_prob is not None else 0.5,
            tft_prob if tft_prob is not None else 0.5,
            self.INCOME_BRACKET_MAP.get(income_bracket, 3) / 6.0,   # Normalised
            self.SEGMENT_TYPE_MAP.get(segment_type, 0) / 6.0,       # Normalised
            min(tenure_months, 300) / 300.0,                         # Cap at 25yr
            max(0, min(credit_score, 900) - 300) / 600.0,           # 300-900 → 0-1
        ], dtype=np.float64)

    def build_meta_features_batch(self, xgb_probs: np.ndarray = None,
                                    lgb_probs: np.ndarray = None,
                                    tft_probs: np.ndarray = None,
                                    income_brackets: list = None,
                                    segment_types: list = None,
                                    tenure_months_arr: np.ndarray = None,
                                    credit_scores: np.ndarray = None, **kwargs) -> np.ndarray:
        """Build meta-features for a batch of customers."""
        n = len(xgb_probs) if xgb_probs is not None else len(lgb_probs)
        meta_X = np.zeros((n, 7), dtype=np.float64)

        meta_X[:, 0] = xgb_probs if xgb_probs is not None else 0.5
        meta_X[:, 1] = lgb_probs if lgb_probs is not None else 0.5
        meta_X[:, 2] = tft_probs if tft_probs is not None else 0.5

        if income_brackets:
            meta_X[:, 3] = [self.INCOME_BRACKET_MAP.get(b, 3) / 6.0 for b in income_brackets]
        else:
            meta_X[:, 3] = 0.5

        if segment_types:
            meta_X[:, 4] = [self.SEGMENT_TYPE_MAP.get(s, 0) / 6.0 for s in segment_types]
        else:
            meta_X[:, 4] = 0.0

        if tenure_months_arr is not None:
            meta_X[:, 5] = np.clip(tenure_months_arr, 0, 300) / 300.0
        else:
            meta_X[:, 5] = 0.5

        if credit_scores is not None:
            meta_X[:, 6] = np.clip(credit_scores - 300, 0, 600) / 600.0
        else:
            meta_X[:, 6] = 0.5

        return meta_X

    def train_meta_learner(self, meta_X: np.ndarray, y: np.ndarray) -> dict:
        """
        Train the meta-learner on OOF predictions + meta-features.

        Args:
            meta_X: (N, 8) meta-feature matrix from build_meta_features_batch
            y: (N,) binary labels

        Returns:
            dict with training metrics
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import roc_auc_score

        self.meta_learner = LogisticRegression(
            C=1.0, penalty="l2", solver="lbfgs",
            max_iter=500, class_weight="balanced",
        )

        # Cross-validated AUC
        cv_scores = cross_val_score(
            self.meta_learner, meta_X, y,
            cv=5, scoring="roc_auc"
        )

        # Fit on full data
        self.meta_learner.fit(meta_X, y)

        # Final AUC on training data (for comparison only)
        train_probs = self.meta_learner.predict_proba(meta_X)[:, 1]
        train_auc = roc_auc_score(y, train_probs)

        metrics = {
            "cv_auc_mean": float(np.mean(cv_scores)),
            "cv_auc_std": float(np.std(cv_scores)),
            "train_auc": float(train_auc),
            "coefficients": dict(zip(
                ["xgb", "lgb", "tft", "income", "segment", "tenure", "credit"],
                self.meta_learner.coef_[0].tolist()
            )),
        }

        logger.info(
            f"[StackingEnsemble] Meta-learner trained | "
            f"CV AUC: {metrics['cv_auc_mean']:.4f} ± {metrics['cv_auc_std']:.4f}"
        )
        return metrics

    def combine_stacked(self, xgb_prob: float = None, lgb_prob: float = None,
                         tft_prob: float = None,
                         customer_meta: dict = None, **kwargs) -> float:
        """
        Use meta-learner if available, else fallback to fixed weights.

        Args:
            customer_meta: dict with income_bracket, segment_type, tenure_months, credit_score
        """
        if self.meta_learner is None:
            return self.fallback_ensemble.combine(xgb_prob, lgb_prob, tft_prob)

        meta = customer_meta or {}
        meta_features = self.build_meta_features(
            xgb_prob=xgb_prob, lgb_prob=lgb_prob,
            tft_prob=tft_prob,
            income_bracket=meta.get("income_bracket", "middle"),
            segment_type=meta.get("segment_type", "salaried"),
            tenure_months=meta.get("tenure_months", 36),
            credit_score=meta.get("credit_score", 700),
        )

        prob = self.meta_learner.predict_proba(meta_features.reshape(1, -1))[0][1]
        return float(np.clip(prob, 0, 1))

    def save_meta_learner(self, path: str):
        """Save meta-learner to disk."""
        if self.meta_learner is None:
            raise ValueError("No meta-learner trained")
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        joblib.dump(self.meta_learner, path)
        logger.info(f"[StackingEnsemble] Meta-learner saved to {path}")

    def load_meta_learner(self, path: str):
        """Load meta-learner from disk."""
        if os.path.exists(path):
            self.meta_learner = joblib.load(path)
            logger.info(f"[StackingEnsemble] Meta-learner loaded from {path}")
        else:
            logger.warning(f"[StackingEnsemble] Meta-learner not found at {path}")

    @staticmethod
    def score_to_risk_tier(score: float, segment_thresholds: dict = None) -> str:
        """Map score to risk tier."""
        return EnsembleScorer.score_to_risk_tier(score, segment_thresholds=segment_thresholds)
