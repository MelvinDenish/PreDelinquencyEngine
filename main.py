# pyre-ignore-all-errors
"""
Pre-Delinquency Intervention Engine - Main Orchestrator
Central entry point to run all components of the PDI engine.
"""
import argparse
import os
import sys
import time
import subprocess
import logging

sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("PDI-Engine")


def cmd_infra_up():
    """Start Docker infrastructure."""
    print("Starting Docker infrastructure...")
    subprocess.run(["docker-compose", "up", "-d"], check=True)
    print("Waiting for services to be healthy...")
    time.sleep(15)
    subprocess.run(["docker-compose", "ps"], check=True)
    print("\n[OK] Infrastructure is up!")


def cmd_infra_down():
    """Stop Docker infrastructure."""
    subprocess.run(["docker-compose", "down"], check=True)
    print("[OK] Infrastructure stopped.")


def cmd_generate_data():
    """Generate synthetic data and load into PostgreSQL + Kafka."""
    from data_generator.run_generator import run_full_generation
    run_full_generation()


def cmd_stream_process():
    """Start the stream processor."""
    from stream_processing.flink_job import create_flink_job_local
    create_flink_job_local()


def cmd_batch_process():
    """Run batch feature computation with Spark."""
    from batch_processing.spark_jobs import run_batch_pipeline
    run_batch_pipeline()


def cmd_feast_setup():
    """Set up and materialize Feast feature store."""
    from feature_store.materialize import (
        export_features_to_parquet, run_feast_apply, run_materialization,
    )
    export_features_to_parquet()
    run_feast_apply()
    run_materialization()


def cmd_train():
    """Train ML models."""
    from ml.train import train_pipeline
    train_pipeline()


def cmd_score_all():
    """Score all customers."""
    import pandas as pd
    import httpx
    from sqlalchemy import create_engine
    from config.settings import PostgresConfig

    engine = create_engine(PostgresConfig.get_url())
    customers = pd.read_sql("SELECT customer_id FROM customers", engine)

    print(f"Scoring {len(customers)} customers...")
    scored = 0
    for _, row in customers.iterrows():
        try:
            resp = httpx.post(
                "http://localhost:8000/score",
                json={"customer_id": row["customer_id"]},
                timeout=10,
            )
            if resp.status_code == 200:
                scored += 1
                if scored % 100 == 0:
                    print(f"  Scored {scored}/{len(customers)}...")
        except Exception as e:
            logger.warning(f"Failed to score {row['customer_id']}: {e}")

    print(f"[OK] Scored {scored}/{len(customers)} customers")


def cmd_scoring_service():
    """Start the FastAPI scoring service."""
    import uvicorn
    from config.settings import ScoringConfig
    uvicorn.run(
        "scoring_service.app:app",
        host=ScoringConfig.HOST,
        port=ScoringConfig.PORT,
        reload=True,
    )


def cmd_dashboard():
    """Start the Plotly Dash dashboard."""
    from dashboard.app import app, DashboardConfig
    app.run(host=DashboardConfig.HOST, port=DashboardConfig.PORT, debug=DashboardConfig.DEBUG)


def cmd_celery_worker():
    """Start Celery worker for interventions."""
    subprocess.run([
        "celery", "-A", "intervention.celery_app", "worker",
        "--loglevel=info", "--queues=interventions,scoring",
    ])


def cmd_feedback_consumer():
    """Start the feedback event consumer."""
    from feedback.feedback_consumer import start_feedback_consumer
    start_feedback_consumer()


def cmd_full_pipeline():
    """Run the complete pipeline end-to-end (except infra which should be up)."""
    print("=" * 70)
    print("PRE-DELINQUENCY INTERVENTION ENGINE - FULL PIPELINE")
    print("=" * 70)

    print("\n[1/5] Generating data...")
    cmd_generate_data()

    print("\n[2/5] Running stream processing...")
    # Stream processing runs in background for real-time
    # For initial load, we process existing data
    from stream_processing.flink_job import create_flink_job_local
    # We'll run it briefly to process existing messages
    import threading
    stream_thread = threading.Thread(target=create_flink_job_local, daemon=True)
    stream_thread.start()
    time.sleep(10)  # Let it process some messages

    print("\n[3/5] Running batch features (if Spark cluster available)...")
    try:
        cmd_batch_process()
    except Exception as e:
        print(f"  Spark batch skipped ({e}). Computing features locally...")
        _compute_features_locally()

    print("\n[4/5] Training ML models...")
    cmd_train()

    print("\n[5/5] Pipeline complete!")
    print("=" * 70)
    print("Next steps:")
    print("  1. Start scoring service:  python main.py scoring-service")
    print("  2. Score all customers:    python main.py score-all  (requires scoring service)")
    print("  3. Start dashboard:        python main.py dashboard")
    print("=" * 70)


def _compute_features_locally():
    """Fallback: compute batch features locally without Spark."""
    import numpy as np
    import pandas as pd
    from sqlalchemy import create_engine
    from config.settings import PostgresConfig
    from datetime import datetime, timedelta

    engine = create_engine(PostgresConfig.get_url())
    print("  Computing batch features locally with pandas...")

    customers = pd.read_sql("SELECT * FROM customers", engine)
    transactions = pd.read_sql("SELECT * FROM transactions", engine)

    if transactions.empty:
        print("  No transactions found!")
        return

    transactions["timestamp"] = pd.to_datetime(transactions["timestamp"])
    now = datetime.now()

    batch_features = []
    for _, customer in customers.iterrows():
        cid = customer["customer_id"]
        cust_txns = transactions[transactions["customer_id"] == cid]

        # Salary delay
        salary_txns = cust_txns[cust_txns["txn_type"] == "salary_credit"]
        if not salary_txns.empty:
            recent_salary = salary_txns[salary_txns["timestamp"] > now - timedelta(days=90)]
            if not recent_salary.empty:
                delays = recent_salary["timestamp"].dt.day - customer["salary_credit_day"]
                salary_delay = max(int(delays.mean()), 0)
            else:
                salary_delay = 0
        else:
            salary_delay = 0

        # Utility delay
        utility_txns = cust_txns[cust_txns["merchant_category"] == "utility"]
        if not utility_txns.empty:
            delays = utility_txns["timestamp"].dt.day.apply(lambda d: max(d - 5, 0))
            utility_delay = round(delays.mean(), 2)
        else:
            utility_delay = 0

        # Spend trend
        recent_disc = cust_txns[
            (cust_txns["merchant_category"].isin(["dining", "entertainment", "clothing", "travel"])) &
            (cust_txns["direction"] == "debit") &
            (cust_txns["timestamp"] > now - timedelta(days=7))
        ]["amount"].sum()

        prev_disc = cust_txns[
            (cust_txns["merchant_category"].isin(["dining", "entertainment", "clothing", "travel"])) &
            (cust_txns["direction"] == "debit") &
            (cust_txns["timestamp"].between(now - timedelta(days=37), now - timedelta(days=30)))
        ]["amount"].sum()

        spend_trend = round(recent_disc / max(prev_disc, 1), 4)

        # Monthly spend stats
        recent_txns = cust_txns[
            (cust_txns["direction"] == "debit") &
            (cust_txns["status"] == "success") &
            (cust_txns["timestamp"] > now - timedelta(days=90))
        ]
        monthly_spend = recent_txns.groupby(recent_txns["timestamp"].dt.to_period("M"))["amount"].sum()
        avg_monthly = round(monthly_spend.mean(), 2) if not monthly_spend.empty else 0
        vol = round(monthly_spend.std() / max(monthly_spend.mean(), 1), 4) if len(monthly_spend) > 1 else 0

        products = customer.get("product_holdings", [])
        if isinstance(products, str):
            products = products.strip("{}").split(",")

        batch_features.append({
            "customer_id": cid,
            "salary_delay_days": salary_delay,
            "utility_payment_delay_avg": utility_delay,
            "discretionary_spend_trend": spend_trend,
            "credit_score": customer.get("credit_score", 650),
            "age": customer.get("age", 30),
            "tenure_months": customer.get("tenure_months", 12),
            "income_bracket": customer.get("income_bracket", "middle"),
            "region": customer.get("region", "South"),
            "gender": customer.get("gender", "M"),
            "product_count": len(products) if products else 1,
            "has_credit_card": "credit_card" in str(products),
            "has_personal_loan": "personal_loan" in str(products),
            "has_mortgage": "home_loan" in str(products),
            "avg_monthly_spend_3m": avg_monthly,
            "spend_volatility_3m": vol,
            "updated_at": datetime.now(),
        })

    batch_df = pd.DataFrame(batch_features)
    batch_df.to_sql("batch_features", engine, if_exists="replace", index=False)
    print(f"  [OK] Computed batch features for {len(batch_df)} customers")

    # Also compute streaming features locally
    streaming_features = []
    for _, customer in customers.iterrows():
        cid = customer["customer_id"]
        cust_txns = transactions[transactions["customer_id"] == cid]

        for suffix, days in [("7d", 7), ("30d", 30)]:
            recent = cust_txns[cust_txns["timestamp"] > now - timedelta(days=days)]
            debits = recent[(recent["direction"] == "debit") & (recent["status"] == "success")]

            disc_cats = {"dining", "entertainment", "clothing", "luxury_goods", "travel"}
            lending_cats = {"lending_app", "payday_lender", "cash_advance"}

            disc_spend = debits[debits["merchant_category"].isin(disc_cats)]["amount"].sum()
            atm_count = len(recent[recent["txn_type"] == "atm"])
            lending_count = len(recent[recent["merchant_category"].isin(lending_cats)])
            failed_auto = len(recent[(recent["channel"] == "auto") & (recent["status"] == "failed")])
            total_spend = debits["amount"].sum()
            txn_count = len(recent)
            avg_amount = debits["amount"].mean() if not debits.empty else 0
            max_amount = debits["amount"].max() if not debits.empty else 0

            if suffix == "7d":
                feature_row = {"customer_id": cid}
            feature_row[f"discretionary_spend_{suffix}"] = round(float(disc_spend), 2)
            feature_row[f"atm_withdrawals_count_{suffix}"] = int(atm_count)
            feature_row[f"lending_app_txn_count_{suffix}"] = int(lending_count)
            feature_row[f"weighted_lending_risk_{suffix}"] = round(lending_count * 0.85, 4)
            feature_row[f"failed_autodebits_count_{suffix}"] = int(failed_auto)
            feature_row[f"total_spend_{suffix}"] = round(float(total_spend), 2)
            feature_row[f"txn_count_{suffix}"] = int(txn_count)
            feature_row[f"avg_txn_amount_{suffix}"] = round(float(avg_amount), 2)
            feature_row[f"max_txn_amount_{suffix}"] = round(float(max_amount), 2)

        feature_row["savings_balance_pct_change_7d"] = round(float(np.random.uniform(-0.3, 0.1)), 4)
        feature_row["updated_at"] = datetime.now()
        streaming_features.append(feature_row)


    stream_df = pd.DataFrame(streaming_features)
    stream_df.to_sql("streaming_features", engine, if_exists="replace", index=False)
    print(f"  [OK] Computed streaming features for {len(stream_df)} customers")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-Delinquency Intervention Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  infra-up           Start Docker infrastructure
  infra-down         Stop Docker infrastructure
  generate-data      Generate synthetic data
  stream-process     Start Flink stream processor
  batch-process      Run Spark batch processing
  feast-setup        Set up Feast feature store
  train              Train ML models
  scoring-service    Start FastAPI scoring service
  score-all          Score all customers
  dashboard          Start Plotly Dash dashboard
  celery-worker      Start Celery worker
  feedback-consumer  Start feedback consumer
  full-pipeline      Run complete pipeline end-to-end
        """
    )
    parser.add_argument("command", help="Command to run")
    args = parser.parse_args()

    commands = {
        "infra-up": cmd_infra_up,
        "infra-down": cmd_infra_down,
        "generate-data": cmd_generate_data,
        "stream-process": cmd_stream_process,
        "batch-process": cmd_batch_process,
        "feast-setup": cmd_feast_setup,
        "train": cmd_train,
        "scoring-service": cmd_scoring_service,
        "score-all": cmd_score_all,
        "dashboard": cmd_dashboard,
        "celery-worker": cmd_celery_worker,
        "feedback-consumer": cmd_feedback_consumer,
        "full-pipeline": cmd_full_pipeline,
    }

    if args.command in commands:
        commands[args.command]()
    else:
        print(f"Unknown command: {args.command}")
        parser.print_help()


if __name__ == "__main__":
    main()

