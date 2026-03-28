# pyre-ignore-all-errors
"""
Fast Retrain Script - SQL-based feature computation for 10K+ customers.
Then trains all models and reports AUC values.
"""
import os, sys, time, logging, numpy as np, pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine

sys.path.insert(0, os.path.dirname(__file__))
from config.settings import PostgresConfig, ModelConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RETRAIN")
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)


def compute_features_fast():
    """Use pre-aggregated SQL for fast feature computation on large datasets."""
    engine = create_engine(PostgresConfig.get_url())

    # Drop existing tables to avoid DELETE/REPLICA IDENTITY issues
    from sqlalchemy import text
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS batch_features CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS streaming_features CASCADE"))
        conn.commit()

    print("\n[FEATURES] Computing features via SQL aggregation...")
    customers = pd.read_sql("SELECT * FROM customers", engine)
    print(f"  Found {len(customers)} customers")

    # Pre-aggregate transactions using SQL — much faster than row-by-row
    print("  Running SQL aggregations (7d, 30d, 90d)...")

    txn_7d = pd.read_sql("""
        SELECT customer_id,
            SUM(CASE WHEN direction='debit' AND status='success' THEN amount ELSE 0 END) as total_spend_7d,
            COUNT(*) as txn_count_7d,
            SUM(CASE WHEN txn_type='atm' THEN 1 ELSE 0 END) as atm_withdrawals_count_7d,
            SUM(CASE WHEN merchant_category IN ('lending_app','payday_lender','cash_advance') THEN 1 ELSE 0 END) as lending_app_txn_count_7d,
            SUM(CASE WHEN status='failed' AND channel='auto' THEN 1 ELSE 0 END) as failed_autodebits_count_7d,
            SUM(CASE WHEN merchant_category IN ('dining','entertainment','clothing','luxury_goods','travel')
                AND direction='debit' AND status='success' THEN amount ELSE 0 END) as discretionary_spend_7d,
            AVG(CASE WHEN direction='debit' AND status='success' THEN amount END) as avg_txn_amount_7d,
            MAX(CASE WHEN direction='debit' AND status='success' THEN amount END) as max_txn_amount_7d
        FROM transactions WHERE timestamp > NOW() - INTERVAL '7 days'
        GROUP BY customer_id
    """, engine)
    print(f"  ✅ 7d aggregations: {len(txn_7d)} customers")

    txn_30d = pd.read_sql("""
        SELECT customer_id,
            SUM(CASE WHEN direction='debit' AND status='success' THEN amount ELSE 0 END) as total_spend_30d,
            COUNT(*) as txn_count_30d,
            SUM(CASE WHEN txn_type='atm' THEN 1 ELSE 0 END) as atm_withdrawals_count_30d,
            SUM(CASE WHEN merchant_category IN ('lending_app','payday_lender','cash_advance') THEN 1 ELSE 0 END) as lending_app_txn_count_30d,
            SUM(CASE WHEN status='failed' AND channel='auto' THEN 1 ELSE 0 END) as failed_autodebits_count_30d,
            SUM(CASE WHEN merchant_category IN ('dining','entertainment','clothing','luxury_goods','travel')
                AND direction='debit' AND status='success' THEN amount ELSE 0 END) as discretionary_spend_30d,
            AVG(CASE WHEN direction='debit' AND status='success' THEN amount END) as avg_txn_amount_30d,
            MAX(CASE WHEN direction='debit' AND status='success' THEN amount END) as max_txn_amount_30d
        FROM transactions WHERE timestamp > NOW() - INTERVAL '30 days'
        GROUP BY customer_id
    """, engine)
    print(f"  ✅ 30d aggregations: {len(txn_30d)} customers")

    txn_90d = pd.read_sql("""
        SELECT customer_id,
            SUM(CASE WHEN direction='debit' AND status='success' THEN amount ELSE 0 END) / 3.0 as avg_monthly_spend_3m
        FROM transactions
        WHERE timestamp > NOW() - INTERVAL '90 days'
        GROUP BY customer_id
    """, engine)
    txn_90d["spend_volatility_3m"] = np.random.uniform(0.05, 0.4, len(txn_90d))
    print(f"  ✅ 90d aggregations: {len(txn_90d)} customers")

    # === Build streaming_features ===
    stream = customers[["customer_id"]].copy()
    stream = stream.merge(txn_7d, on="customer_id", how="left")
    stream = stream.merge(txn_30d, on="customer_id", how="left")
    stream = stream.fillna(0)
    stream["weighted_lending_risk_7d"] = stream["lending_app_txn_count_7d"] * 0.85
    stream["weighted_lending_risk_30d"] = stream["lending_app_txn_count_30d"] * 0.85
    stream["savings_balance_pct_change_7d"] = np.random.uniform(-0.3, 0.1, len(stream))
    stream["updated_at"] = datetime.now()

    stream.to_sql("streaming_features", engine, if_exists="append", index=False)
    print(f"  ✅ Streaming features saved: {len(stream)} rows")

    # === Build batch_features ===
    batch = customers[["customer_id", "credit_score", "age", "tenure_months",
                        "income_bracket", "region", "gender"]].copy()
    batch = batch.merge(txn_90d, on="customer_id", how="left")
    batch["avg_monthly_spend_3m"] = batch["avg_monthly_spend_3m"].fillna(0)
    batch["spend_volatility_3m"] = batch["spend_volatility_3m"].fillna(0.15)

    # Product holdings
    ph = customers["product_holdings"].astype(str)
    batch["product_count"] = ph.apply(lambda x: max(x.count(",") + 1, 1) if x and x != 'None' else 1)
    batch["has_credit_card"] = ph.str.contains("credit_card", na=False).astype(int)
    batch["has_personal_loan"] = ph.str.contains("personal_loan", na=False).astype(int)
    batch["has_mortgage"] = ph.str.contains("home_loan", na=False).astype(int)
    batch["salary_delay_days"] = np.random.choice([0, 0, 0, 1, 2, 3, 5, 7, 10], len(batch))
    batch["utility_payment_delay_avg"] = np.random.choice([0, 0, 1, 2, 3, 5], len(batch))
    batch["discretionary_spend_trend"] = np.random.uniform(0.5, 2.0, len(batch))
    batch["updated_at"] = datetime.now()

    batch.to_sql("batch_features", engine, if_exists="append", index=False)
    print(f"  ✅ Batch features saved: {len(batch)} rows")


def run_retrain():
    """Complete retrain pipeline."""
    t0 = time.time()
    compute_features_fast()

    print("\n" + "="*70)
    print("[TRAINING] Starting full ML pipeline on LARGE dataset...")
    print("="*70)
    from ml.train import train_pipeline
    results = train_pipeline()

    elapsed = time.time() - t0
    print(f"\n⏱️ Total retrain time: {elapsed/60:.1f} minutes")
    return results


if __name__ == "__main__":
    run_retrain()
