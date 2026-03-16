"""
Airflow DAG: Batch Feature Computation
Runs daily to compute batch features using Apache Spark.
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "pdi-engine",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "batch_feature_computation",
    default_args=default_args,
    description="Daily batch feature computation with Apache Spark",
    schedule_interval="0 2 * * *",  # 2 AM daily
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["pdi", "features", "spark"],
)


def run_spark_batch(**kwargs):
    """Run Spark batch feature computation."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from batch_processing.spark_jobs import run_batch_pipeline
    run_batch_pipeline()


def run_feast_materialization(**kwargs):
    """Run Feast materialization to push features to online store."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from feature_store.materialize import export_features_to_parquet, run_materialization
    export_features_to_parquet()
    run_materialization()


spark_task = PythonOperator(
    task_id="run_spark_batch_features",
    python_callable=run_spark_batch,
    dag=dag,
)

feast_task = PythonOperator(
    task_id="materialize_to_feast",
    python_callable=run_feast_materialization,
    dag=dag,
)

spark_task >> feast_task
