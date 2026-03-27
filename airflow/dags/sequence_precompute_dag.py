# pyre-ignore-all-errors
"""
Sequence Precompute DAG (M7)
Daily Airflow DAG that precomputes 29-day sequences for TFT/LSTM
and caches them in Redis for fast real-time inference.
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "pdi_engine",
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
    "depends_on_past": False,
}


def precompute_sequences(**kwargs):
    """Precompute TFT/LSTM sequence windows for all active customers."""
    import sys, os, logging
    import numpy as np
    import psycopg2

    sys.path.insert(0, "/app")
    from config.settings import PostgresConfig
    from scoring_service.sequence_cache import SequenceCache

    logger = logging.getLogger(__name__)
    cache = SequenceCache()

    conn = psycopg2.connect(
        host=PostgresConfig.HOST, port=PostgresConfig.PORT,
        user=PostgresConfig.USER, password=PostgresConfig.PASSWORD,
        dbname=PostgresConfig.DB,
    )
    cursor = conn.cursor()

    # Get all active customers
    cursor.execute("SELECT customer_id FROM customers WHERE status = 'active'")
    customer_ids = [row[0] for row in cursor.fetchall()]

    logger.info(f"[SeqPrecompute] Processing {len(customer_ids)} customers")

    seq_len = 30
    cached_count = 0

    for cid in customer_ids:
        # Fetch last 30 days of streaming features
        cursor.execute("""
            SELECT feature_date, feature_vector
            FROM daily_features
            WHERE customer_id = %s
            ORDER BY feature_date DESC
            LIMIT %s
        """, (cid, seq_len))
        rows = cursor.fetchall()

        if len(rows) < 7:
            continue  # Not enough history

        # Build sequence (pad if < 30 days)
        feature_vectors = [row[1] for row in reversed(rows)]
        n_features = len(feature_vectors[0]) if feature_vectors else 10

        sequence = np.zeros((seq_len, n_features))
        for i, fv in enumerate(feature_vectors):
            offset = seq_len - len(feature_vectors) + i
            if isinstance(fv, list):
                sequence[offset] = np.array(fv[:n_features])

        # Fetch static features
        cursor.execute("""
            SELECT age, credit_score, tenure_months, product_count,
                   CASE WHEN has_credit_card THEN 1 ELSE 0 END,
                   CASE WHEN has_personal_loan THEN 1 ELSE 0 END,
                   CASE WHEN has_mortgage THEN 1 ELSE 0 END
            FROM batch_features WHERE customer_id = %s
        """, (cid,))
        static_row = cursor.fetchone()
        static_features = np.array(static_row or [30, 700, 36, 2, 0, 0, 0], dtype=np.float32)

        cache.cache_tft_sequence(cid, sequence, static_features)
        cached_count += 1

    cursor.close()
    conn.close()

    logger.info(f"[SeqPrecompute] Cached sequences for {cached_count}/{len(customer_ids)} customers")
    return {"cached_count": cached_count}


with DAG(
    dag_id="pdi_sequence_precompute",
    default_args=default_args,
    description="Daily precompute of TFT/LSTM sequences → Redis cache",
    schedule_interval="0 2 * * *",  # 2 AM daily
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["pdi", "ml", "cache"],
) as dag:

    precompute_task = PythonOperator(
        task_id="precompute_all_sequences",
        python_callable=precompute_sequences,
    )
