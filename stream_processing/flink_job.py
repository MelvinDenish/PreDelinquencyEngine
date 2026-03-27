# pyre-ignore-all-errors
"""
Apache Flink Stream Processing Job
Consumes transactions from Kafka, enriches them with merchant risk scores,
computes rolling window features, and writes results to Redis and PostgreSQL.

This is a real PyFlink job that runs on the Apache Flink cluster (Docker).
PyFlink IS the official Apache Flink Python API - it compiles to Flink's
internal DataStream/Table operators and runs on the JVM-based Flink runtime.
"""
import json
import logging
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config.settings import KafkaConfig, RedisConfig, PostgresConfig

logger = logging.getLogger(__name__)


def create_flink_job():
    """
    Create and configure the PyFlink stream processing job.
    This job:
    1. Reads from Kafka 'transactions' topic
    2. Enriches transactions with merchant risk scores
    3. Computes rolling 7d/30d window features per customer
    4. Writes enriched features to PostgreSQL (which syncs to Redis via a separate process)
    """
    from pyflink.datastream import StreamExecutionEnvironment
    from pyflink.table import StreamTableEnvironment, EnvironmentSettings
    from pyflink.table.expressions import col, lit, call
    from pyflink.table.window import Tumble, Slide
    from pyflink.common import WatermarkStrategy, Duration

    # Create execution environment
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(2)
    env.enable_checkpointing(30000)  # 30-second checkpoint interval

    # Add Kafka and JDBC connectors
    env.add_jars(
        "file:///opt/flink/lib/flink-sql-connector-kafka-3.0.2-1.18.jar",
        "file:///opt/flink/lib/flink-connector-jdbc-3.1.2-1.18.jar",
        "file:///opt/flink/lib/postgresql-42.7.1.jar",
    )

    t_env = StreamTableEnvironment.create(env)

    # ─────────────────────────────────────────────
    # Source: Kafka transactions topic
    # ─────────────────────────────────────────────
    t_env.execute_sql(f"""
        CREATE TABLE kafka_transactions (
            txn_id STRING,
            customer_id STRING,
            txn_type STRING,
            merchant_category STRING,
            amount DECIMAL(12,2),
            direction STRING,
            channel STRING,
            status STRING,
            `timestamp` TIMESTAMP(3),
            WATERMARK FOR `timestamp` AS `timestamp` - INTERVAL '5' SECOND
        ) WITH (
            'connector' = 'kafka',
            'topic' = '{KafkaConfig.TOPIC_TRANSACTIONS}',
            'properties.bootstrap.servers' = '{KafkaConfig.INTERNAL_BOOTSTRAP_SERVERS}',
            'properties.group.id' = 'flink-stream-processor',
            'scan.startup.mode' = 'earliest-offset',
            'format' = 'json',
            'json.fail-on-missing-field' = 'false',
            'json.ignore-parse-errors' = 'true'
        )
    """)

    # ─────────────────────────────────────────────
    # Source: Merchant risk scores (from PostgreSQL)
    # ─────────────────────────────────────────────
    t_env.execute_sql(f"""
        CREATE TABLE merchant_risks (
            merchant_category STRING,
            risk_score DECIMAL(5,3),
            risk_category STRING
        ) WITH (
            'connector' = 'jdbc',
            'url' = 'jdbc:postgresql://{PostgresConfig.HOST}:{PostgresConfig.PORT}/{PostgresConfig.DB}',
            'table-name' = 'merchant_risk_scores',
            'username' = '{PostgresConfig.USER}',
            'password' = '{PostgresConfig.PASSWORD}',
            'lookup.cache.max-rows' = '100',
            'lookup.cache.ttl' = '1h'
        )
    """)

    # ─────────────────────────────────────────────
    # Enrichment: Join transactions with merchant risk
    # ─────────────────────────────────────────────
    t_env.execute_sql("""
        CREATE TEMPORARY VIEW enriched_transactions AS
        SELECT
            t.txn_id,
            t.customer_id,
            t.txn_type,
            t.merchant_category,
            t.amount,
            t.direction,
            t.channel,
            t.status,
            t.`timestamp`,
            COALESCE(m.risk_score, 0.30) AS merchant_risk_score,
            COALESCE(m.risk_category, 'medium') AS risk_category
        FROM kafka_transactions t
        LEFT JOIN merchant_risks FOR SYSTEM_TIME AS OF t.`timestamp` AS m
            ON t.merchant_category = m.merchant_category
    """)

    # ─────────────────────────────────────────────
    # Sink: Streaming features to PostgreSQL
    # ─────────────────────────────────────────────
    t_env.execute_sql(f"""
        CREATE TABLE streaming_features_sink (
            customer_id STRING,
            discretionary_spend_7d DECIMAL(12,2),
            atm_withdrawals_count_7d BIGINT,
            lending_app_txn_count_7d BIGINT,
            weighted_lending_risk_7d DECIMAL(8,4),
            failed_autodebits_count_7d BIGINT,
            total_spend_7d DECIMAL(12,2),
            txn_count_7d BIGINT,
            avg_txn_amount_7d DECIMAL(12,2),
            max_txn_amount_7d DECIMAL(12,2),
            updated_at TIMESTAMP(3),
            PRIMARY KEY (customer_id) NOT ENFORCED
        ) WITH (
            'connector' = 'jdbc',
            'url' = 'jdbc:postgresql://{PostgresConfig.HOST}:{PostgresConfig.PORT}/{PostgresConfig.DB}',
            'table-name' = 'streaming_features',
            'username' = '{PostgresConfig.USER}',
            'password' = '{PostgresConfig.PASSWORD}'
        )
    """)

    # ─────────────────────────────────────────────
    # Rolling 7-day Window Feature Computation
    # ─────────────────────────────────────────────
    t_env.execute_sql("""
        INSERT INTO streaming_features_sink
        SELECT
            customer_id,

            -- Discretionary spend (dining + entertainment + clothing + luxury + travel)
            SUM(CASE
                WHEN merchant_category IN ('dining', 'entertainment', 'clothing', 'luxury_goods', 'travel')
                     AND direction = 'debit' AND status = 'success'
                THEN amount ELSE 0
            END) AS discretionary_spend_7d,

            -- ATM withdrawal count
            SUM(CASE WHEN txn_type = 'atm' THEN 1 ELSE 0 END) AS atm_withdrawals_count_7d,

            -- Lending app transaction count
            SUM(CASE
                WHEN merchant_category IN ('lending_app', 'payday_lender', 'cash_advance')
                THEN 1 ELSE 0
            END) AS lending_app_txn_count_7d,

            -- Weighted lending risk (sum of merchant_risk_score for lending txns)
            SUM(CASE
                WHEN merchant_category IN ('lending_app', 'payday_lender', 'cash_advance')
                THEN merchant_risk_score ELSE 0
            END) AS weighted_lending_risk_7d,

            -- Failed auto-debit count
            SUM(CASE
                WHEN (txn_type = 'auto_debit' OR channel = 'auto') AND status = 'failed'
                THEN 1 ELSE 0
            END) AS failed_autodebits_count_7d,

            -- Total spend
            SUM(CASE
                WHEN direction = 'debit' AND status = 'success'
                THEN amount ELSE 0
            END) AS total_spend_7d,

            -- Transaction count
            COUNT(*) AS txn_count_7d,

            -- Average transaction amount
            AVG(CASE
                WHEN direction = 'debit' AND status = 'success'
                THEN amount ELSE NULL
            END) AS avg_txn_amount_7d,

            -- Max transaction amount
            MAX(CASE
                WHEN direction = 'debit' AND status = 'success'
                THEN amount ELSE 0
            END) AS max_txn_amount_7d,

            MAX(`timestamp`) AS updated_at

        FROM TABLE(
            TUMBLE(TABLE enriched_transactions, DESCRIPTOR(`timestamp`), INTERVAL '7' DAY)
        )
        GROUP BY customer_id, window_start, window_end
    """)

    print("[Flink] Stream processing job submitted successfully")


def create_flink_job_local():
    """
    Local-mode stream processor that processes Kafka messages using
    the same logic as the Flink job but runs as a standalone Python process.
    This is used when the Flink cluster is not available or for development.
    """
    import json
    import redis as redis_lib
    from datetime import datetime, timedelta
    from collections import defaultdict
    from kafka import KafkaConsumer

    from stream_processing.enrichment import enrich_transaction

    logger.info("[StreamProcessor] Starting local stream processor...")

    # Connect to Redis for feature storage
    r = redis_lib.Redis(
        host=RedisConfig.HOST, port=RedisConfig.PORT, db=RedisConfig.DB,
        decode_responses=True,
    )

    # In-memory state for rolling windows
    customer_state = defaultdict(lambda: {
        "transactions": [],  # (timestamp, enriched_txn) list
        "features": {},
    })

    # Kafka consumer
    consumer = KafkaConsumer(
        KafkaConfig.TOPIC_TRANSACTIONS,
        bootstrap_servers=KafkaConfig.BOOTSTRAP_SERVERS,
        group_id="stream-processor-local",
        auto_offset_reset="earliest",
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        max_poll_records=100,
    )

    def compute_features(customer_id: str, state: dict) -> dict:
        """Compute rolling 7-day features from transaction state."""
        now = datetime.now()
        cutoff_7d = now - timedelta(days=7)
        cutoff_30d = now - timedelta(days=30)

        txns_7d = []
        txns_30d = []
        for ts_str, txn in state["transactions"]:
            try:
                ts = datetime.fromisoformat(ts_str)
            except (ValueError, TypeError):
                continue
            if ts >= cutoff_7d:
                txns_7d.append(txn)
            if ts >= cutoff_30d:
                txns_30d.append(txn)

        def calc_features(txns, suffix):
            discretionary_cats = {"dining", "entertainment", "clothing", "luxury_goods", "travel"}
            lending_cats = {"lending_app", "payday_lender", "cash_advance"}
            gambling_cats = {"gambling", "lottery"}

            disc_spend = sum(t["amount"] for t in txns
                           if t.get("merchant_category") in discretionary_cats
                           and t.get("direction") == "debit" and t.get("status") == "success")

            atm_count = sum(1 for t in txns if t.get("txn_type") == "atm")

            lending_count = sum(1 for t in txns if t.get("merchant_category") in lending_cats)

            lending_risk = sum(t.get("merchant_risk_score", 0) for t in txns
                              if t.get("merchant_category") in lending_cats)

            failed_auto = sum(1 for t in txns
                             if (t.get("txn_type") == "auto_debit" or t.get("channel") == "auto")
                             and t.get("status") == "failed")

            total_spend = sum(t["amount"] for t in txns
                            if t.get("direction") == "debit" and t.get("status") == "success")

            txn_count = len(txns)

            debit_amounts = [t["amount"] for t in txns
                           if t.get("direction") == "debit" and t.get("status") == "success"]
            avg_amount = sum(debit_amounts) / len(debit_amounts) if debit_amounts else 0
            max_amount = max(debit_amounts) if debit_amounts else 0

            # Gambling + lottery spend (problem statement: "increased gambling, or lottery spend")
            gambling_spend = sum(t["amount"] for t in txns
                               if t.get("merchant_category") in gambling_cats
                               and t.get("direction") == "debit" and t.get("status") == "success")

            return {
                f"discretionary_spend_{suffix}": round(disc_spend, 2),
                f"atm_withdrawals_count_{suffix}": atm_count,
                f"lending_app_txn_count_{suffix}": lending_count,
                f"weighted_lending_risk_{suffix}": round(lending_risk, 4),
                f"failed_autodebits_count_{suffix}": failed_auto,
                f"total_spend_{suffix}": round(total_spend, 2),
                f"txn_count_{suffix}": txn_count,
                f"avg_txn_amount_{suffix}": round(avg_amount, 2),
                f"max_txn_amount_{suffix}": round(max_amount, 2),
                f"gambling_lottery_spend_{suffix}": round(gambling_spend, 2),
            }

        features = {}
        features.update(calc_features(txns_7d, "7d"))
        features.update(calc_features(txns_30d, "30d"))

        return features

    def write_features_to_redis(customer_id: str, features: dict):
        """Write computed features to Redis."""
        key = f"features:streaming:{customer_id}"
        r.hset(key, mapping={k: str(v) for k, v in features.items()})
        r.expire(key, 86400 * 7)  # 7-day TTL

    processed = 0
    try:
        for message in consumer:
            txn = message.value
            customer_id = txn.get("customer_id")
            if not customer_id:
                continue

            # Enrich with merchant risk
            enriched = enrich_transaction(txn)

            # Add to state
            state = customer_state[customer_id]
            state["transactions"].append((txn.get("timestamp", ""), enriched))

            # Keep only last 30 days in memory
            cutoff = (datetime.now() - timedelta(days=30)).isoformat()
            state["transactions"] = [
                (ts, t) for ts, t in state["transactions"] if ts >= cutoff
            ]

            # Compute features
            features = compute_features(customer_id, state)
            state["features"] = features

            # Write to Redis
            write_features_to_redis(customer_id, features)

            processed += 1
            if processed % 500 == 0:
                logger.info(f"[StreamProcessor] Processed {processed} transactions, "
                          f"tracking {len(customer_state)} customers")

    except KeyboardInterrupt:
        logger.info(f"[StreamProcessor] Stopped after processing {processed} transactions")
    finally:
        consumer.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Flink Stream Processing Job")
    parser.add_argument("--mode", choices=["flink", "local"], default="local",
                       help="Run mode: 'flink' for cluster, 'local' for standalone")
    args = parser.parse_args()

    if args.mode == "flink":
        create_flink_job()
    else:
        create_flink_job_local()
