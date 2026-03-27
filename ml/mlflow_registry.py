# pyre-ignore-all-errors
"""
MLflow Model Registry
Manages model versioning, experiment tracking, and stage transitions.
Integrates with the ML training pipeline for reproducibility.
"""
import os
import sys
import json
import logging
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "pdi-delinquency-prediction"


def setup_mlflow():
    """Set up MLflow tracking."""
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        logger.info(f"[MLflow] Tracking URI: {MLFLOW_TRACKING_URI}")
        logger.info(f"[MLflow] Experiment: {EXPERIMENT_NAME}")
        return mlflow
    except ImportError:
        logger.warning("MLflow not installed. Using local file logging.")
        return None


def log_training_run(
    model_name: str,
    params: dict,
    metrics: dict,
    artifacts_dir: str = None,
    tags: dict = None,
) -> str:
    """
    Log a complete training run to MLflow.
    Returns the run ID.
    """
    mlflow = setup_mlflow()
    if mlflow is None:
        # Fallback: log to local file
        run_id = f"local_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        log_entry = {
            "run_id": run_id,
            "model_name": model_name,
            "params": params,
            "metrics": metrics,
            "tags": tags or {},
            "timestamp": datetime.now().isoformat(),
        }
        log_dir = os.path.join(os.path.dirname(__file__), '..', 'models', 'mlflow_logs')
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, f"{run_id}.json"), "w") as f:
            json.dump(log_entry, f, indent=2, default=str)
        logger.info(f"[MLflow] Local log saved: {run_id}")
        return run_id

    with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d')}") as run:
        # Log parameters
        for key, value in params.items():
            try:
                mlflow.log_param(key, value)
            except Exception:
                mlflow.log_param(key, str(value)[:250])

        # Log metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                mlflow.log_metric(key, value)

        # Log tags
        all_tags = {
            "model_type": model_name,
            "framework": _get_framework(model_name),
            "training_date": datetime.now().isoformat(),
        }
        if tags:
            all_tags.update(tags)
        mlflow.set_tags(all_tags)

        # Log artifacts
        if artifacts_dir and os.path.exists(artifacts_dir):
            mlflow.log_artifacts(artifacts_dir)

        run_id = run.info.run_id
        logger.info(f"[MLflow] Run logged: {run_id}")
        return run_id


def register_model(run_id: str, model_name: str, model_path: str):
    """Register a trained model in MLflow Model Registry."""
    mlflow = setup_mlflow()
    if mlflow is None:
        logger.info(f"[MLflow] Local mode: model {model_name} registered as {run_id}")
        return

    try:
        model_uri = f"runs:/{run_id}/{model_path}"

        # Register model
        result = mlflow.register_model(model_uri, model_name)
        logger.info(f"[MLflow] Registered model: {model_name} v{result.version}")

        # Transition to staging
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=result.version,
            stage="Staging",
        )
        logger.info(f"[MLflow] Model {model_name} v{result.version} -> Staging")
        return result.version

    except Exception as e:
        logger.warning(f"[MLflow] Registration failed: {e}")
        return None


def promote_to_production(model_name: str, version: str):
    """Promote a model version from Staging to Production."""
    mlflow = setup_mlflow()
    if mlflow is None:
        logger.info(f"[MLflow] Local mode: {model_name} v{version} promoted")
        return

    try:
        client = mlflow.tracking.MlflowClient()

        # Archive current production version
        for mv in client.search_model_versions(f"name='{model_name}'"):
            if mv.current_stage == "Production":
                client.transition_model_version_stage(
                    name=model_name, version=mv.version, stage="Archived"
                )

        # Promote new version
        client.transition_model_version_stage(
            name=model_name, version=version, stage="Production"
        )
        logger.info(f"[MLflow] {model_name} v{version} -> Production")

    except Exception as e:
        logger.warning(f"[MLflow] Promotion failed: {e}")


def get_production_model(model_name: str):
    """Load the current production model from MLflow."""
    mlflow = setup_mlflow()
    if mlflow is None:
        return None

    try:
        model_uri = f"models:/{model_name}/Production"
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info(f"[MLflow] Loaded production model: {model_name}")
        return model
    except Exception as e:
        logger.warning(f"[MLflow] Failed to load production model: {e}")
        return None


def log_ensemble_run(xgb_metrics: dict, lgb_metrics: dict,
                     lstm_metrics: dict, ensemble_metrics: dict,
                     fairness_results: dict = None) -> str:
    """Log a complete ensemble training run."""
    all_params = {
        "xgb_cv_auc": xgb_metrics.get("cv_auc_mean", 0),
        "lgb_cv_auc": lgb_metrics.get("cv_auc_mean", 0),
        "lstm_val_auc": lstm_metrics.get("best_val_auc", 0),
        "ensemble_test_auc": ensemble_metrics.get("ensemble_auc", 0),
        "num_models": 3,
    }

    all_metrics = {
        "xgb_train_auc": xgb_metrics.get("train_auc", 0),
        "xgb_cv_auc_mean": xgb_metrics.get("cv_auc_mean", 0),
        "xgb_test_auc": xgb_metrics.get("val_auc", 0),
        "lgb_train_auc": lgb_metrics.get("train_auc", 0),
        "lgb_cv_auc_mean": lgb_metrics.get("cv_auc_mean", 0),
        "lgb_test_auc": lgb_metrics.get("val_auc", 0),
        "lstm_train_auc": lstm_metrics.get("train_auc", 0),
        "lstm_best_val_auc": lstm_metrics.get("best_val_auc", 0),
        "ensemble_auc": ensemble_metrics.get("ensemble_auc", 0),
    }

    tags = {"ensemble_type": "xgboost+lightgbm+lstm"}
    if fairness_results:
        tags["fairness_verdict"] = fairness_results.get("verdict", "N/A")

    return log_training_run("ensemble", all_params, all_metrics, tags=tags)


def _get_framework(model_name: str) -> str:
    """Get ML framework for a model name."""
    frameworks = {
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
        "lstm": "PyTorch",
        "ensemble": "XGBoost+LightGBM+PyTorch",
    }
    return frameworks.get(model_name, "unknown")
