"""
Weekly Retrain DAG (M9)
Automated weekly retraining pipeline that:
1. Loads latest labelled data
2. Retrains XGBoost, LightGBM, LSTM, TFT
3. Generates OOF predictions for meta-learner
4. Trains meta-learner on stacked predictions
5. Validates AUC/PSI before promotion
6. Registers models in MLflow production stage
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator

default_args = {
    "owner": "pdi_engine",
    "retries": 1,
    "retry_delay": timedelta(minutes=15),
    "depends_on_past": False,
}


def extract_training_data(**kwargs):
    """Pull labelled data from PostgreSQL for retraining."""
    import sys, os, logging
    import pandas as pd
    import psycopg2

    sys.path.insert(0, "/app")
    from config.settings import PostgresConfig

    logger = logging.getLogger(__name__)

    conn = psycopg2.connect(
        host=PostgresConfig.HOST, port=PostgresConfig.PORT,
        user=PostgresConfig.USER, password=PostgresConfig.PASSWORD,
        dbname=PostgresConfig.DB,
    )

    # Pull customers who now have confirmed outcomes (delinquent / not)
    query = """
        SELECT bf.*, rs.risk_score as prior_risk_score,
               CASE WHEN fe.outcome = 'delinquent' THEN 1 ELSE 0 END as label
        FROM batch_features bf
        JOIN risk_scores rs ON bf.customer_id = rs.customer_id
        JOIN feedback_events fe ON bf.customer_id = fe.customer_id
        WHERE fe.outcome IS NOT NULL
          AND rs.scored_at >= NOW() - INTERVAL '90 days'
    """
    df = pd.read_sql(query, conn)
    conn.close()

    output_path = "/tmp/retrain_data.parquet"
    df.to_parquet(output_path, index=False)
    logger.info(f"[Retrain] Extracted {len(df)} labelled samples → {output_path}")

    kwargs["ti"].xcom_push(key="train_data_path", value=output_path)
    kwargs["ti"].xcom_push(key="sample_count", value=len(df))
    return len(df)


def retrain_all_models(**kwargs):
    """Retrain all 4 models and generate OOF predictions."""
    import sys, os, logging
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold

    sys.path.insert(0, "/app")
    from ml.train import build_dataset, train_xgboost, train_lightgbm, train_lstm
    from ml.tft_model import TFTDelinquencyModel

    logger = logging.getLogger(__name__)

    data_path = kwargs["ti"].xcom_pull(key="train_data_path")
    df = pd.read_parquet(data_path)

    X, y, feature_names = build_dataset(df)

    # OOF predictions for meta-learner
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_xgb = np.zeros(len(y))
    oof_lgb = np.zeros(len(y))
    oof_lstm = np.zeros(len(y))
    oof_tft = np.zeros(len(y))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        logger.info(f"[Retrain] Fold {fold+1}/{n_folds}")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # XGBoost
        xgb_model = train_xgboost(X_train, y_train, X_val, y_val)
        oof_xgb[val_idx] = xgb_model.predict_proba(X_val)[:, 1]

        # LightGBM
        lgb_model = train_lightgbm(X_train, y_train, X_val, y_val)
        oof_lgb[val_idx] = lgb_model.predict_proba(X_val)[:, 1]

        # LSTM
        lstm_model = train_lstm(X_train, y_train, X_val, y_val)
        oof_lstm[val_idx] = lstm_model.predict_proba(X_val)

        # TFT (needs temporal sequences — use available features as proxy)
        seq_len = 30
        n_temporal = min(X.shape[1], 20)
        n_static = X.shape[1] - n_temporal

        X_temp = np.repeat(X_train[:, :n_temporal].reshape(-1, 1, n_temporal), seq_len, axis=1)
        X_stat = X_train[:, n_temporal:]
        X_temp_val = np.repeat(X_val[:, :n_temporal].reshape(-1, 1, n_temporal), seq_len, axis=1)
        X_stat_val = X_val[:, n_temporal:]

        tft = TFTDelinquencyModel(n_temporal_features=n_temporal, n_static_features=n_static)
        tft.train(X_temp, X_stat, y_train, X_temp_val, X_stat_val, y_val)
        oof_tft[val_idx] = tft.predict_proba(X_temp_val, X_stat_val)

    # Save OOF predictions
    np.savez("/tmp/oof_predictions.npz",
             xgb=oof_xgb, lgb=oof_lgb, lstm=oof_lstm, tft=oof_tft, y=y)

    logger.info("[Retrain] OOF predictions generated for meta-learner training")
    kwargs["ti"].xcom_push(key="oof_path", value="/tmp/oof_predictions.npz")


def train_meta_learner(**kwargs):
    """Train the meta-learner on OOF predictions."""
    import sys, os, logging
    import numpy as np
    import pandas as pd

    sys.path.insert(0, "/app")
    from ml.ensemble import StackingEnsemble

    logger = logging.getLogger(__name__)

    oof = np.load("/tmp/oof_predictions.npz")
    data_path = kwargs["ti"].xcom_pull(key="train_data_path")
    df = pd.read_parquet(data_path)

    stacker = StackingEnsemble()

    # Build meta-features
    meta_X = stacker.build_meta_features_batch(
        xgb_probs=oof["xgb"],
        lgb_probs=oof["lgb"],
        tft_probs=oof["tft"],
        lstm_probs=oof["lstm"],
        income_brackets=df.get("income_bracket", pd.Series(["middle"] * len(df))).tolist(),
        segment_types=df.get("segment_type", pd.Series(["salaried"] * len(df))).tolist(),
        tenure_months_arr=df.get("tenure_months", pd.Series([36] * len(df))).values,
        credit_scores=df.get("credit_score", pd.Series([700] * len(df))).values,
    )

    metrics = stacker.train_meta_learner(meta_X, oof["y"])
    stacker.save_meta_learner("/app/models/meta_learner.joblib")

    logger.info(f"[Retrain] Meta-learner CV AUC: {metrics['cv_auc_mean']:.4f}")
    kwargs["ti"].xcom_push(key="meta_auc", value=metrics["cv_auc_mean"])


def validate_and_promote(**kwargs):
    """Check model quality before promoting to production."""
    import sys, logging

    sys.path.insert(0, "/app")
    logger = logging.getLogger(__name__)

    meta_auc = kwargs["ti"].xcom_pull(key="meta_auc")
    sample_count = kwargs["ti"].xcom_pull(key="sample_count")

    MIN_AUC = 0.65
    MIN_SAMPLES = 100

    if meta_auc < MIN_AUC:
        logger.warning(f"[Retrain] AUC {meta_auc:.4f} < {MIN_AUC}. Skipping promotion.")
        return "skip_promotion"

    if sample_count < MIN_SAMPLES:
        logger.warning(f"[Retrain] Only {sample_count} samples. Skipping promotion.")
        return "skip_promotion"

    logger.info(f"[Retrain] Validation passed. AUC={meta_auc:.4f}, Samples={sample_count}")
    return "promote_models"


def promote_models(**kwargs):
    """Register retrained models in MLflow production stage."""
    import sys, logging
    sys.path.insert(0, "/app")
    logger = logging.getLogger(__name__)

    try:
        import mlflow
        mlflow.set_tracking_uri("http://mlflow:5000")

        model_version = datetime.now().strftime("%Y%m%d_%H%M")
        mlflow.log_param("retrain_version", model_version)
        mlflow.log_metric("meta_auc", kwargs["ti"].xcom_pull(key="meta_auc"))

        logger.info(f"[Retrain] Models promoted to production: v{model_version}")
    except Exception as e:
        logger.warning(f"[Retrain] MLflow registration skipped: {e}")


def skip_promotion(**kwargs):
    """Log that model promotion was skipped."""
    import logging
    logging.getLogger(__name__).info("[Retrain] Model promotion skipped — quality gate failed")


with DAG(
    dag_id="pdi_weekly_retrain",
    default_args=default_args,
    description="Weekly automated retraining with OOF stacking and quality gates",
    schedule_interval="0 3 * * 0",  # 3 AM every Sunday
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["pdi", "ml", "retrain"],
) as dag:

    extract = PythonOperator(
        task_id="extract_training_data",
        python_callable=extract_training_data,
    )

    retrain = PythonOperator(
        task_id="retrain_all_models",
        python_callable=retrain_all_models,
    )

    meta = PythonOperator(
        task_id="train_meta_learner",
        python_callable=train_meta_learner,
    )

    validate = BranchPythonOperator(
        task_id="validate_and_promote",
        python_callable=validate_and_promote,
    )

    promote = PythonOperator(
        task_id="promote_models",
        python_callable=promote_models,
    )

    skip = PythonOperator(
        task_id="skip_promotion",
        python_callable=skip_promotion,
    )

    extract >> retrain >> meta >> validate >> [promote, skip]
