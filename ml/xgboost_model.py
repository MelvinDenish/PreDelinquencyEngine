# pyre-ignore-all-errors
"""
XGBoost Delinquency Risk Model
Trains an XGBoost classifier on tabular behavioral features.
"""
import os
import sys
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, roc_auc_score, precision_recall_curve,
    average_precision_score, confusion_matrix,
)
import joblib
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)


class XGBoostDelinquencyModel:
    """XGBoost classifier for delinquency risk prediction."""

    def __init__(self, params: dict = None):
        self.params = params or {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_estimators": 300,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 0.1,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "scale_pos_weight": 3.0,  # Handle class imbalance
            "random_state": 42,
            "n_jobs": -1,
        }
        self.model = None
        self.feature_names = None

    def train(self, X: np.ndarray, y: np.ndarray, feature_names: list = None,
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> dict:
        """
        Train the XGBoost model.
        Returns metrics dict.
        """
        self.feature_names = feature_names

        self.model = xgb.XGBClassifier(**self.params)

        eval_set = []
        if X_val is not None:
            eval_set = [(X_val, y_val)]

        self.model.fit(
            X, y,
            eval_set=eval_set if eval_set else None,
            verbose=False,
        )

        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring="roc_auc")

        # Predictions
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        y_pred = self.model.predict(X)

        metrics = {
            "train_auc": roc_auc_score(y, y_pred_proba),
            "cv_auc_mean": cv_scores.mean(),
            "cv_auc_std": cv_scores.std(),
            "avg_precision": average_precision_score(y, y_pred_proba),
            "classification_report": classification_report(y, y_pred, output_dict=True),
        }

        if X_val is not None:
            val_pred_proba = self.model.predict_proba(X_val)[:, 1]
            metrics["val_auc"] = roc_auc_score(y_val, val_pred_proba)

        # Feature importance
        if feature_names:
            importances = self.model.feature_importances_
            importance_dict = dict(zip(feature_names, importances.tolist()))
            metrics["feature_importance"] = dict(
                sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            )

        logger.info(f"[XGBoost] Training AUC: {metrics['train_auc']:.4f}, "
                    f"CV AUC: {metrics['cv_auc_mean']:.4f} +/- {metrics['cv_auc_std']:.4f}")

        return metrics

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of delinquency."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary prediction."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X)

    def save(self, path: str = None):
        """Save model to disk."""
        path = path or os.path.join(MODEL_DIR, "xgboost_model.joblib")
        joblib.dump({
            "model": self.model,
            "params": self.params,
            "feature_names": self.feature_names,
        }, path)
        logger.info(f"[XGBoost] Model saved to {path}")

    def load(self, path: str = None):
        """Load model from disk."""
        path = path or os.path.join(MODEL_DIR, "xgboost_model.joblib")
        data = joblib.load(path)
        self.model = data["model"]
        self.params = data["params"]
        self.feature_names = data["feature_names"]
        logger.info(f"[XGBoost] Model loaded from {path}")

    def get_booster(self):
        """Get the underlying XGBoost booster for SHAP."""
        return self.model

