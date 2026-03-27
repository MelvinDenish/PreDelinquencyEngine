# pyre-ignore-all-errors
"""
LightGBM Delinquency Risk Model
Gradient boosting with LightGBM for tabular delinquency prediction.
Complements XGBoost in the ensemble with different boosting strategy.
"""
import os
import sys
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report
import joblib
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)


class LightGBMDelinquencyModel:
    """LightGBM classifier for delinquency prediction."""

    def __init__(self, params: dict = None):
        self.params = params or {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "num_leaves": 63,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_child_samples": 20,
            "max_depth": 8,
            "lambda_l1": 0.1,
            "lambda_l2": 0.1,
            "verbose": -1,
            "n_jobs": -1,
            "seed": 42,
        }
        self.model = None
        self.feature_names = None
        self.feature_importances_ = None

    def train(self, X: np.ndarray, y: np.ndarray,
              feature_names: list = None,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              num_boost_round: int = 500, early_stopping_rounds: int = 50) -> dict:
        """
        Train LightGBM model with cross-validation and early stopping.
        """
        self.feature_names = feature_names or [f"f_{i}" for i in range(X.shape[1])]

        # Handle class imbalance
        pos_count = y.sum()
        neg_count = len(y) - pos_count
        if pos_count > 0:
            self.params["scale_pos_weight"] = neg_count / pos_count

        # Cross-validation
        cv_aucs = []
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train, y_fold_val = y[train_idx], y[val_idx]

            train_data = lgb.Dataset(X_fold_train, label=y_fold_train,
                                     feature_name=self.feature_names)
            val_data = lgb.Dataset(X_fold_val, label=y_fold_val,
                                   feature_name=self.feature_names, reference=train_data)

            callbacks = [
                lgb.early_stopping(stopping_rounds=early_stopping_rounds),
                lgb.log_evaluation(period=0),
            ]

            fold_model = lgb.train(
                self.params,
                train_data,
                num_boost_round=num_boost_round,
                valid_sets=[val_data],
                callbacks=callbacks,
            )

            val_pred = fold_model.predict(X_fold_val)
            fold_auc = roc_auc_score(y_fold_val, val_pred)
            cv_aucs.append(fold_auc)
            logger.info(f"[LightGBM] Fold {fold+1} AUC: {fold_auc:.4f}")

        # Train final model on all data
        if X_val is not None:
            train_data = lgb.Dataset(X, label=y, feature_name=self.feature_names)
            val_data = lgb.Dataset(X_val, label=y_val,
                                   feature_name=self.feature_names, reference=train_data)
            callbacks = [
                lgb.early_stopping(stopping_rounds=early_stopping_rounds),
                lgb.log_evaluation(period=0),
            ]
            self.model = lgb.train(
                self.params,
                train_data,
                num_boost_round=num_boost_round,
                valid_sets=[val_data],
                callbacks=callbacks,
            )
        else:
            train_data = lgb.Dataset(X, label=y, feature_name=self.feature_names)
            self.model = lgb.train(
                self.params,
                train_data,
                num_boost_round=num_boost_round,
            )

        # Feature importances
        self.feature_importances_ = dict(
            zip(self.feature_names,
                self.model.feature_importance(importance_type="gain"))
        )

        # Metrics
        train_pred = self.model.predict(X)
        train_auc = roc_auc_score(y, train_pred)

        val_auc = None
        if X_val is not None:
            val_pred = self.model.predict(X_val)
            val_auc = roc_auc_score(y_val, val_pred)

        metrics = {
            "train_auc": train_auc,
            "cv_auc_mean": float(np.mean(cv_aucs)),
            "cv_auc_std": float(np.std(cv_aucs)),
            "val_auc": val_auc,
            "num_trees": self.model.num_trees(),
            "top_features": dict(sorted(self.feature_importances_.items(),
                                       key=lambda x: x[1], reverse=True)[:10]),
        }

        logger.info(f"[LightGBM] Train AUC: {train_auc:.4f}, "
                    f"CV AUC: {np.mean(cv_aucs):.4f} +/- {np.std(cv_aucs):.4f}")

        return metrics

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of delinquency."""
        if self.model is None:
            raise ValueError("Model not trained or loaded.")
        return self.model.predict(X)

    def get_booster(self):
        """Return the underlying LightGBM booster (for SHAP/LIME)."""
        return self.model

    def save(self, path: str = None):
        """Save model to disk."""
        path = path or os.path.join(MODEL_DIR, "lightgbm_model.joblib")
        joblib.dump({
            "model": self.model,
            "feature_names": self.feature_names,
            "feature_importances": self.feature_importances_,
            "params": self.params,
        }, path)
        logger.info(f"[LightGBM] Model saved to {path}")

    def load(self, path: str = None):
        """Load model from disk."""
        path = path or os.path.join(MODEL_DIR, "lightgbm_model.joblib")
        data = joblib.load(path)
        self.model = data["model"]
        self.feature_names = data["feature_names"]
        self.feature_importances_ = data["feature_importances"]
        self.params = data["params"]
        logger.info(f"[LightGBM] Model loaded from {path}")
