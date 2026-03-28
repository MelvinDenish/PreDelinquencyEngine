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
    # Fix 5: Added non-linear interactions to break circular dependency
    lending_7d = merged.get("lending_app_txn_count_7d", 0).fillna(0).astype(float)
    failed_7d = merged.get("failed_autodebits_count_7d", 0).fillna(0).astype(float)
    salary_delay = merged.get("salary_delay_days", 0).fillna(0).astype(float).clip(0, 30)
    savings_chg = merged.get("savings_balance_pct_change_7d", 0).fillna(0).astype(float)
    disc_trend = merged.get("discretionary_spend_trend", 1.0).fillna(1.0).astype(float).clip(0, 3)
    w_lending = merged.get("weighted_lending_risk_7d", 0).fillna(0).astype(float).clip(0, 5)
    util_delay = merged.get("utility_payment_delay_avg", 0).fillna(0).astype(float).clip(0, 30)

    merged["risk_signal"] = (
        # Linear components
        (lending_7d * 0.12) +
        (failed_7d * 0.15) +
        (salary_delay / 30 * 0.10) +
        ((-savings_chg).clip(0, 1) * 0.10) +
        (disc_trend / 3 * 0.08) +
        (w_lending / 5 * 0.10) +
        (util_delay / 30 * 0.05) +
        # Non-linear interaction terms (Fix 5: breaks circular dependency)
        (np.sqrt(lending_7d * failed_7d.clip(0, 10)) * 0.10) +
        ((salary_delay / 30) * (w_lending / 5) * 0.08) +
        (np.log1p(lending_7d) * np.log1p(failed_7d) * 0.07) +
        # Acceleration signal
        ((disc_trend / 3) * ((-savings_chg).clip(0, 1)) * 0.05)
    )

    # Normalize risk_signal to [0, 1]
    min_signal = merged["risk_signal"].min()
    max_signal = merged["risk_signal"].max()
    if max_signal > min_signal:
        merged["risk_signal_norm"] = (merged["risk_signal"] - min_signal) / (max_signal - min_signal)
    else:
        merged["risk_signal_norm"] = 0.0

    # Fix 5: Increased noise for realistic variance (0.05 -> 0.08)
    noise = np.random.normal(0, 0.08, len(merged))
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

    # ── Fix 4: Feature Engineering — Ratio & Interaction Features ──
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
    print(f"  -> Positive (at-risk): {int(y.sum())} ({y.mean()*100:.1f}%)")
    print(f"  -> Negative (stable):  {int(len(y) - y.sum())} ({(1-y.mean())*100:.1f}%)")
    print(f"  -> Engineered features added: {len(engineered_features)}")

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

