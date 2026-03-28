# pyre-ignore-all-errors
"""
Dataset Builder
Builds training datasets from PostgreSQL by merging streaming features,
batch features, and outcome-based labels from payment_events.

Labels are derived from ACTUAL missed payment events (insufficient balance
for EMI on due date), NOT from a formula over input features. This eliminates
label leakage — features come from behavioral history, labels come from
future payment outcomes.
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
    Build training dataset from PostgreSQL with outcome-based labels.
    Returns (X, y, feature_names, customer_ids) tuple.

    Features: streaming_features + batch_features (behavioral history)
    Labels: payment_events table (actual missed EMI/auto-debit events)

    No label leakage — features and labels are derived from different
    data sources. Features capture behavioral patterns; labels capture
    actual payment outcomes from the financial simulation.
    """
    engine = create_engine(PostgresConfig.get_url())

    # Load streaming features (computed from transaction history)
    streaming_df = pd.read_sql("SELECT * FROM streaming_features", engine)

    # Load batch features (computed from periodic aggregations)
    batch_df = pd.read_sql("SELECT * FROM batch_features", engine)

    # Merge on customer_id
    merged = streaming_df.merge(batch_df, on="customer_id", how="inner",
                                suffixes=("_stream", "_batch"))

    if merged.empty:
        print("[DatasetBuilder] WARNING: No feature data found. Run data generation and feature computation first.")
        return None, None, None, None

    # ─── Outcome-based labels from payment_events ───────────────
    # Label = 1 if customer had any missed payment (insufficient balance for EMI)
    # These events are generated during the financial simulation when
    # a customer's running balance cannot cover their EMI on the due date.
    outcome_df = pd.read_sql("""
        SELECT customer_id,
               COUNT(*) as missed_payment_count
        FROM payment_events
        GROUP BY customer_id
    """, engine)

    if outcome_df.empty:
        print("[DatasetBuilder] WARNING: No payment events found. Labels will be all zeros.")
        print("  -> Run data generation first to populate payment_events table.")
        merged["label"] = 0
    else:
        outcome_df["label"] = 1
        merged = merged.merge(
            outcome_df[["customer_id", "label"]],
            on="customer_id", how="left"
        )
        merged["label"] = merged["label"].fillna(0).astype(int)

    delinquent_count = int(merged["label"].sum())
    total = len(merged)
    print(f"[DatasetBuilder] Labels from payment_events: "
          f"{delinquent_count}/{total} delinquent ({delinquent_count/max(total,1)*100:.1f}%)")

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

    # ── Feature Engineering — Ratio & Interaction Features ──
    eps = 1e-6
    total_spend_7d = merged.get("total_spend_7d", pd.Series(0, index=merged.index)).fillna(0).astype(float)
    total_spend_30d = merged.get("total_spend_30d", pd.Series(0, index=merged.index)).fillna(0).astype(float)
    avg_monthly_3m = merged.get("avg_monthly_spend_3m", pd.Series(0, index=merged.index)).fillna(0).astype(float)
    txn_count_7d = merged.get("txn_count_7d", pd.Series(1, index=merged.index)).fillna(1).astype(float)
    lending_7d_raw = merged.get("lending_app_txn_count_7d", pd.Series(0, index=merged.index)).fillna(0).astype(float)
    failed_7d_raw = merged.get("failed_autodebits_count_7d", pd.Series(0, index=merged.index)).fillna(0).astype(float)
    savings_pct = merged.get("savings_balance_pct_change_7d", pd.Series(0, index=merged.index)).fillna(0).astype(float)
    w_lending_raw = merged.get("weighted_lending_risk_7d", pd.Series(0, index=merged.index)).fillna(0).astype(float)
    disc_trend_raw = merged.get("discretionary_spend_trend", pd.Series(1, index=merged.index)).fillna(1).astype(float)

    merged["spend_to_income_ratio"] = total_spend_30d / (avg_monthly_3m + eps)
    merged["lending_to_txn_ratio"] = lending_7d_raw / (txn_count_7d + 1)
    merged["failed_debit_rate"] = failed_7d_raw / (txn_count_7d + 1)
    merged["week_vs_month_spend"] = total_spend_7d / (total_spend_30d / 4.3 + eps)
    merged["savings_x_lending"] = (-savings_pct).clip(0, 1) * lending_7d_raw
    merged["risk_acceleration"] = disc_trend_raw * w_lending_raw
    merged["lending_failed_interaction"] = np.log1p(lending_7d_raw) * np.log1p(failed_7d_raw)

    engineered_features = [
        "spend_to_income_ratio", "lending_to_txn_ratio", "failed_debit_rate",
        "week_vs_month_spend", "savings_x_lending", "risk_acceleration",
        "lending_failed_interaction",
    ]
    for ef in engineered_features:
        if ef in merged.columns:
            feature_cols.append(ef)

    X = merged[feature_cols].fillna(0).values.astype(np.float32)
    y = merged["label"].values.astype(np.float32)
    customer_ids = merged["customer_id"].values

    print(f"[DatasetBuilder] Built dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  -> Positive (delinquent): {int(y.sum())} ({y.mean()*100:.1f}%)")
    print(f"  -> Negative (stable):     {int(len(y) - y.sum())} ({(1-y.mean())*100:.1f}%)")
    print(f"  -> Engineered features: {len(engineered_features)}")

    return X, y, feature_cols, customer_ids


def build_temporal_dataset(sequence_length: int = 30) -> tuple:
    """
    Build temporal dataset for TFT training.
    Creates sequences of daily feature snapshots per customer.
    Returns (X_sequences, y, customer_ids).

    Labels use outcome-based payment_events (same as tabular dataset).
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

    # Get outcome-based labels from payment_events
    outcome_df = pd.read_sql("""
        SELECT customer_id, 1 as label
        FROM payment_events
        GROUP BY customer_id
    """, engine)
    label_map = dict(zip(outcome_df["customer_id"], outcome_df["label"])) if not outcome_df.empty else {}

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

        # Label from payment_events (outcome-based, not trend-based)
        label = label_map.get(customer_id, 0)

        sequences.append(seq)
        labels.append(label)
        cust_ids.append(customer_id)

    X = np.array(sequences, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)

    print(f"[DatasetBuilder] Built temporal dataset: {X.shape[0]} sequences, "
          f"length={X.shape[1]}, features={X.shape[2]}")
    print(f"  -> Positive (delinquent): {int(y.sum())} ({y.mean()*100:.1f}%)")

    return X, y, np.array(cust_ids)


if __name__ == "__main__":
    print("Building tabular dataset...")
    X, y, features, cids = build_training_dataset()
    if X is not None:
        print(f"\nFeatures used: {features}")

    print("\nBuilding temporal dataset...")
    X_seq, y_seq, cids_seq = build_temporal_dataset()
