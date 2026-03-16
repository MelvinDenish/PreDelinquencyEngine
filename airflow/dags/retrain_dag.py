"""
Airflow DAG: Model Retraining Pipeline
Bi-weekly (or drift-triggered) retraining with champion-challenger testing.
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator

default_args = {
    "owner": "pdi-engine",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

dag = DAG(
    "model_retraining",
    default_args=default_args,
    description="Bi-weekly model retraining with drift detection and fairness checks",
    schedule_interval="0 3 1,15 * *",  # 3 AM on 1st and 15th
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["pdi", "ml", "retraining"],
)


def check_drift(**kwargs):
    """Check for data drift. Branch to retrain or skip."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from feedback.drift_detector import check_and_alert
    drift_detected = check_and_alert()
    # Always retrain on schedule, but log drift status
    return "retrain_models"


def retrain_models(**kwargs):
    """Run full retraining pipeline."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from ml.train import train_pipeline
    results = train_pipeline()
    return results


def run_fairness_audit(**kwargs):
    """Post-training fairness audit."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import numpy as np
    import pandas as pd
    from sqlalchemy import create_engine
    from config.settings import PostgresConfig
    from ml.xgboost_model import XGBoostDelinquencyModel
    from ml.dataset_builder import build_training_dataset
    from ml.fairness import run_bias_audit

    X, y, features, cids = build_training_dataset()
    if X is None:
        return {"status": "no_data"}

    model = XGBoostDelinquencyModel()
    model.load()

    engine = create_engine(PostgresConfig.get_url())
    demo_df = pd.read_sql(
        "SELECT customer_id, age, gender, region, income_bracket FROM customers",
        engine,
    )

    results = run_bias_audit(model, X, y, demo_df.head(len(X)))
    return results


drift_check = BranchPythonOperator(
    task_id="check_drift",
    python_callable=check_drift,
    dag=dag,
)

retrain_task = PythonOperator(
    task_id="retrain_models",
    python_callable=retrain_models,
    dag=dag,
)

fairness_task = PythonOperator(
    task_id="run_fairness_audit",
    python_callable=run_fairness_audit,
    dag=dag,
)

drift_check >> retrain_task >> fairness_task
