# pyre-ignore-all-errors
"""
Dataset Builder
Builds training datasets from PostgreSQL by merging streaming features,
batch features, and outcome labels.
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config.settings import PostgresConfig, ModelConfig


def build_training_dataset() -> tuple:
    """
    Build training dataset from PostgreSQL.
    Returns (X, y, feature_names, customer_ids) tuple.

    Labels are generated based on actual behavioral signals:
    - Customers with high lending_app activity, failed auto-debits,
      salary delays, and spending increases are labeled as at-risk.
    """
    engine = create_engine(PostgresConfig.get_url())

    # Load streaming features
    streaming_df = pd.read_sql("SELECT * FROM streaming_features", engine)

    # Load batch features
    batch_df = pd.read_sql("SELECT * FROM batch_features", engine)

    # Merge on customer_id
    merged = streaming_df.merge(batch_df, on="customer_id", how="inner",
                                suffixes=("_stream", "_batch"))

    if merged.empty:
        print("[DatasetBuilder] WARNING: No feature data found. Run data generation and feature computation first.")
        return None, None, None, None

    # Generate labels from actual behavioral signals
    # This creates a composite risk signal based on real feature patterns
    merged["risk_signal"] = (
        # Lending app activity (strong signal)
        (merged.get("lending_app_txn_count_7d", 0).fillna(0).astype(float) * 0.15) +
        # Failed auto-debits (strong signal)
        (merged.get("failed_autodebits_count_7d", 0).fillna(0).astype(float) * 0.20) +
        # Salary delay (moderate signal)
        (merged.get("salary_delay_days", 0).fillna(0).astype(float).clip(0, 30) / 30 * 0.15) +
        # Savings drawdown (moderate signal)
        ((-merged.get("savings_balance_pct_change_7d", 0).fillna(0).astype(float)).clip(0, 1) * 0.15) +
        # High spending relative to income (moderate signal)
        (merged.get("discretionary_spend_trend", 1.0).fillna(1.0).astype(float).clip(0, 3) / 3 * 0.10) +
        # Weighted lending risk (strong signal)
        (merged.get("weighted_lending_risk_7d", 0).fillna(0).astype(float).clip(0, 5) / 5 * 0.15) +
        # Utility delay (moderate signal)
        (merged.get("utility_payment_delay_avg", 0).fillna(0).astype(float).clip(0, 30) / 30 * 0.10)
    )

    # Normalize risk_signal to [0, 1]
    min_signal = merged["risk_signal"].min()
    max_signal = merged["risk_signal"].max()
    if max_signal > min_signal:
        merged["risk_signal_norm"] = (merged["risk_signal"] - min_signal) / (max_signal - min_signal)
    else:
        merged["risk_signal_norm"] = 0.0

    # Add noise for realistic variance
    noise = np.random.normal(0, 0.05, len(merged))
    merged["risk_signal_norm"] = (merged["risk_signal_norm"] + noise).clip(0, 1)

    # Binary label: 1 = delinquent risk, 0 = stable
    threshold = merged["risk_signal_norm"].quantile(0.75)  # Top 25% as at-risk
    merged["label"] = (merged["risk_signal_norm"] >= threshold).astype(int)

    # Select feature columns
    feature_cols = []
    for col in ModelConfig.FEATURE_COLUMNS:
        if col in merged.columns:
            feature_cols.append(col)
        elif f"{col}_stream" in merged.columns:
            feature_cols.append(f"{col}_stream")
        elif f"{col}_batch" in merged.columns:
            feature_cols.append(f"{col}_batch")

    # Convert booleans to int
    for col in feature_cols:
        if merged[col].dtype == bool:
            merged[col] = merged[col].astype(int)

    X = merged[feature_cols].fillna(0).values.astype(np.float32)
    y = merged["label"].values.astype(np.float32)
    customer_ids = merged["customer_id"].values

    print(f"[DatasetBuilder] Built dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  -> Positive (at-risk): {int(y.sum())} ({y.mean()*100:.1f}%)")
    print(f"  -> Negative (stable):  {int(len(y) - y.sum())} ({(1-y.mean())*100:.1f}%)")

    return X, y, feature_cols, customer_ids


def build_temporal_dataset(sequence_length: int = 30) -> tuple:
    """
    Build temporal dataset for LSTM training.
    Creates sequences of daily feature snapshots per customer.
    Returns (X_sequences, y, customer_ids).
    """
    engine = create_engine(PostgresConfig.get_url())

    # Get daily snapshots from transactions
    query = """
    SELECT
        t.customer_id,
        DATE(t.timestamp) as txn_date,
        SUM(CASE WHEN t.direction = 'debit' AND t.status = 'success' THEN t.amount ELSE 0 END) as daily_spend,
        COUNT(*) as daily_txn_count,
        SUM(CASE WHEN t.txn_type = 'atm' THEN 1 ELSE 0 END) as daily_atm_count,
        SUM(CASE WHEN t.merchant_category IN ('lending_app','payday_lender','cash_advance') THEN 1 ELSE 0 END) as daily_lending_count,
        SUM(CASE WHEN t.status = 'failed' THEN 1 ELSE 0 END) as daily_failed_count,
        SUM(CASE WHEN t.merchant_category IN ('dining','entertainment','clothing','luxury_goods','travel')
                AND t.direction = 'debit' AND t.status = 'success' THEN t.amount ELSE 0 END) as daily_discretionary,
        AVG(CASE WHEN t.direction = 'debit' AND t.status = 'success' THEN t.amount ELSE NULL END) as daily_avg_amount
    FROM transactions t
    GROUP BY t.customer_id, DATE(t.timestamp)
    ORDER BY t.customer_id, txn_date
    """

    df = pd.read_sql(query, engine)
    if df.empty:
        return None, None, None

    # Build sequences
    feature_cols = ["daily_spend", "daily_txn_count", "daily_atm_count",
                    "daily_lending_count", "daily_failed_count",
                    "daily_discretionary", "daily_avg_amount"]

    sequences = []
    labels = []
    cust_ids = []

    for customer_id, group in df.groupby("customer_id"):
        group = group.sort_values("txn_date").fillna(0)
        values = group[feature_cols].values.astype(np.float32)

        if len(values) < sequence_length:
            # Pad with zeros
            padding = np.zeros((sequence_length - len(values), len(feature_cols)), dtype=np.float32)
            values = np.vstack([padding, values])

        # Take last `sequence_length` days
        seq = values[-sequence_length:]

        # Label: based on last week's behavior trends
        last_week = values[-7:] if len(values) >= 7 else values
        first_week = values[:7] if len(values) >= 14 else values[:len(values)//2]

        # Risk increases if spending, lending, and failures are trending up
        lending_trend = last_week[:, 3].mean() - first_week[:, 3].mean()
        failure_trend = last_week[:, 4].mean() - first_week[:, 4].mean()
        spend_trend = (last_week[:, 0].mean() / max(first_week[:, 0].mean(), 1)) - 1

        risk_score = lending_trend * 0.4 + failure_trend * 0.3 + max(spend_trend, 0) * 0.3
        label = 1 if risk_score > 0.1 else 0

        sequences.append(seq)
        labels.append(label)
        cust_ids.append(customer_id)

    X = np.array(sequences, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)

    print(f"[DatasetBuilder] Built temporal dataset: {X.shape[0]} sequences, "
          f"length={X.shape[1]}, features={X.shape[2]}")
    print(f"  -> Positive: {int(y.sum())} ({y.mean()*100:.1f}%)")

    return X, y, np.array(cust_ids)


if __name__ == "__main__":
    print("Building tabular dataset...")
    X, y, features, cids = build_training_dataset()
    if X is not None:
        print(f"\nFeatures used: {features}")

    print("\nBuilding temporal dataset...")
    X_seq, y_seq, cids_seq = build_temporal_dataset()

