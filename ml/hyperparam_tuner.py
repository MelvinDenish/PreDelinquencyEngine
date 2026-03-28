# pyre-ignore-all-errors
"""
Optuna Hyperparameter Tuner (Fix 3)
====================================
Automated hyperparameter optimization for XGBoost and LightGBM
using Bayesian optimization via Optuna.

Usage:
    python -m ml.hyperparam_tuner
"""
import os, sys
import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import joblib
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ml.dataset_builder import build_training_dataset

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')


def tune_xgboost(X, y, n_trials=50) -> dict:
    """Tune XGBoost hyperparameters with Optuna."""
    print(f"\n[Optuna] Tuning XGBoost ({n_trials} trials)...")

    def objective(trial):
        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 3.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 5.0),
            "random_state": 42,
            "n_jobs": -1,
            "use_label_encoder": False,
        }

        model = xgb.XGBClassifier(**params)
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=kf, scoring="roc_auc", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize", study_name="xgboost_tuning")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"[Optuna] XGBoost best AUC: {study.best_value:.4f}")
    print(f"[Optuna] Best params: {study.best_params}")

    return study.best_params


def tune_lightgbm(X, y, n_trials=50) -> dict:
    """Tune LightGBM hyperparameters with Optuna."""
    print(f"\n[Optuna] Tuning LightGBM ({n_trials} trials)...")

    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 2.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 2.0),
            "verbose": -1,
            "n_jobs": -1,
            "seed": 42,
        }

        pos = y.sum()
        neg = len(y) - pos
        params["scale_pos_weight"] = neg / max(pos, 1)

        train_data = lgb.Dataset(X, label=y)
        cv_result = lgb.cv(
            params, train_data, num_boost_round=500,
            nfold=5, stratified=True,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
            return_cvbooster=False,
        )
        best_auc = max(cv_result["valid auc-mean"])
        return best_auc

    study = optuna.create_study(direction="maximize", study_name="lightgbm_tuning")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"[Optuna] LightGBM best AUC: {study.best_value:.4f}")
    print(f"[Optuna] Best params: {study.best_params}")

    return study.best_params


def run_tuning(n_trials=30):
    """Run full hyperparameter tuning pipeline."""
    print("=" * 60)
    print("Optuna Hyperparameter Tuning")
    print("=" * 60)

    X, y, feature_names, _ = build_training_dataset()
    if X is None:
        print("ERROR: Could not build dataset")
        return

    # Tune XGBoost
    xgb_best = tune_xgboost(X, y, n_trials=n_trials)
    xgb_best_path = os.path.join(MODEL_DIR, "xgb_best_params.joblib")
    joblib.dump(xgb_best, xgb_best_path)
    print(f"[Optuna] XGBoost params saved: {xgb_best_path}")

    # Tune LightGBM
    lgb_best = tune_lightgbm(X, y, n_trials=n_trials)
    lgb_best_path = os.path.join(MODEL_DIR, "lgb_best_params.joblib")
    joblib.dump(lgb_best, lgb_best_path)
    print(f"[Optuna] LightGBM params saved: {lgb_best_path}")

    return xgb_best, lgb_best


if __name__ == "__main__":
    run_tuning(n_trials=30)
