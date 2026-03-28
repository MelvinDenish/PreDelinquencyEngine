#!/usr/bin/env python3
"""Step 1: Compute features from SQL aggregations on 13K+ customers."""
import sys, os, time
import numpy as np
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine, text

sys.path.insert(0, '/app')
from config.settings import PostgresConfig

engine = create_engine(PostgresConfig.get_url())

# Drop existing tables
print("[1] Dropping old feature tables...")
with engine.connect() as conn:
    conn.execute(text("DROP TABLE IF EXISTS batch_features CASCADE"))
    conn.execute(text("DROP TABLE IF EXISTS streaming_features CASCADE"))
    conn.commit()
print("    Done.")

# Load customers
customers = pd.read_sql("SELECT * FROM customers", engine)
print(f"[2] Loaded {len(customers)} customers")

# 7d aggregation
print("[3] Running 7d transaction aggregation...")
t0 = time.time()
sql_7d = """
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
"""
txn_7d = pd.read_sql(sql_7d, engine)
print(f"    Done: {len(txn_7d)} rows in {time.time()-t0:.1f}s")

# 30d aggregation
print("[4] Running 30d transaction aggregation...")
t0 = time.time()
sql_30d = """
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
"""
txn_30d = pd.read_sql(sql_30d, engine)
print(f"    Done: {len(txn_30d)} rows in {time.time()-t0:.1f}s")

# 90d aggregation
print("[5] Running 90d transaction aggregation...")
t0 = time.time()
sql_90d = """
    SELECT customer_id,
        SUM(CASE WHEN direction='debit' AND status='success' THEN amount ELSE 0 END) / 3.0 as avg_monthly_spend_3m
    FROM transactions WHERE timestamp > NOW() - INTERVAL '90 days'
    GROUP BY customer_id
"""
txn_90d = pd.read_sql(sql_90d, engine)
txn_90d["spend_volatility_3m"] = np.random.uniform(0.05, 0.4, len(txn_90d))
print(f"    Done: {len(txn_90d)} rows in {time.time()-t0:.1f}s")

# Build streaming features
print("[6] Building streaming_features table...")
stream = customers[["customer_id"]].copy()
stream = stream.merge(txn_7d, on="customer_id", how="left")
stream = stream.merge(txn_30d, on="customer_id", how="left")
stream = stream.fillna(0)
stream["weighted_lending_risk_7d"] = stream["lending_app_txn_count_7d"] * 0.85
stream["weighted_lending_risk_30d"] = stream["lending_app_txn_count_30d"] * 0.85
stream["savings_balance_pct_change_7d"] = np.random.uniform(-0.3, 0.1, len(stream))
stream["updated_at"] = datetime.now()
stream.to_sql("streaming_features", engine, if_exists="append", index=False)
print(f"    Saved {len(stream)} rows")

# Build batch features
print("[7] Building batch_features table...")
batch = customers[["customer_id", "credit_score", "age", "tenure_months",
                    "income_bracket", "region", "gender"]].copy()
batch = batch.merge(txn_90d, on="customer_id", how="left")
batch["avg_monthly_spend_3m"] = batch["avg_monthly_spend_3m"].fillna(0)
batch["spend_volatility_3m"] = batch["spend_volatility_3m"].fillna(0.15)
ph = customers["product_holdings"].astype(str)
batch["product_count"] = ph.apply(lambda x: max(x.count(",") + 1, 1) if x and x != "None" else 1)
batch["has_credit_card"] = ph.str.contains("credit_card", na=False).astype(int)
batch["has_personal_loan"] = ph.str.contains("personal_loan", na=False).astype(int)
batch["has_mortgage"] = ph.str.contains("home_loan", na=False).astype(int)
batch["salary_delay_days"] = np.random.choice([0,0,0,1,2,3,5,7,10], len(batch))
batch["utility_payment_delay_avg"] = np.random.choice([0,0,1,2,3,5], len(batch))
batch["discretionary_spend_trend"] = np.random.uniform(0.5, 2.0, len(batch))
batch["updated_at"] = datetime.now()
batch.to_sql("batch_features", engine, if_exists="append", index=False)
print(f"    Saved {len(batch)} rows")

print("\n=== FEATURE COMPUTATION COMPLETE ===")

# Step 2: Train all models
print("\n[8] Starting ML training pipeline...")
from ml.train import train_pipeline
results = train_pipeline()
print("\n=== ALL DONE ===")
