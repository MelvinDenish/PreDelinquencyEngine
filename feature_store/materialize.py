"""
Feast Materialization Script
Exports features from PostgreSQL to Parquet and materializes to Redis online store.
"""
import os
import sys
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config.settings import PostgresConfig, FeastConfig

from sqlalchemy import create_engine


def export_features_to_parquet():
    """Export streaming and batch features from PostgreSQL to Parquet for Feast."""
    engine = create_engine(PostgresConfig.get_url())

    data_dir = os.path.join(FeastConfig.REPO_PATH, "data")
    os.makedirs(data_dir, exist_ok=True)

    # Export streaming features
    streaming_df = pd.read_sql("SELECT * FROM streaming_features", engine)
    if "updated_at" not in streaming_df.columns:
        streaming_df["updated_at"] = datetime.now()
    streaming_df["updated_at"] = pd.to_datetime(streaming_df["updated_at"])
    streaming_path = os.path.join(data_dir, "streaming_features.parquet")
    streaming_df.to_parquet(streaming_path, index=False)
    print(f"[Feast] Exported {len(streaming_df)} streaming feature rows to {streaming_path}")

    # Export batch features
    batch_df = pd.read_sql("SELECT * FROM batch_features", engine)
    if "updated_at" not in batch_df.columns:
        batch_df["updated_at"] = datetime.now()
    batch_df["updated_at"] = pd.to_datetime(batch_df["updated_at"])
    batch_path = os.path.join(data_dir, "batch_features.parquet")
    batch_df.to_parquet(batch_path, index=False)
    print(f"[Feast] Exported {len(batch_df)} batch feature rows to {batch_path}")

    return streaming_path, batch_path


def run_feast_apply():
    """Apply Feast feature definitions (register features)."""
    from feast import FeatureStore
    store = FeatureStore(repo_path=FeastConfig.REPO_PATH)
    store.apply([])  # This registers all entities and feature views from the repo
    print("[Feast] Feature definitions applied successfully")


def run_materialization():
    """Materialize features from offline store to online store (Redis)."""
    from feast import FeatureStore
    store = FeatureStore(repo_path=FeastConfig.REPO_PATH)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)

    store.materialize(start_date=start_date, end_date=end_date)
    print(f"[Feast] Materialization complete: {start_date} to {end_date}")


def get_online_features(customer_ids: list) -> pd.DataFrame:
    """Retrieve online features for given customer IDs."""
    from feast import FeatureStore
    store = FeatureStore(repo_path=FeastConfig.REPO_PATH)

    entity_rows = [{"customer_id": cid} for cid in customer_ids]

    feature_refs = [
        "streaming_features:discretionary_spend_7d",
        "streaming_features:discretionary_spend_30d",
        "streaming_features:atm_withdrawals_count_7d",
        "streaming_features:atm_withdrawals_count_30d",
        "streaming_features:lending_app_txn_count_7d",
        "streaming_features:lending_app_txn_count_30d",
        "streaming_features:weighted_lending_risk_7d",
        "streaming_features:weighted_lending_risk_30d",
        "streaming_features:savings_balance_pct_change_7d",
        "streaming_features:failed_autodebits_count_7d",
        "streaming_features:failed_autodebits_count_30d",
        "streaming_features:total_spend_7d",
        "streaming_features:total_spend_30d",
        "streaming_features:txn_count_7d",
        "streaming_features:txn_count_30d",
        "streaming_features:avg_txn_amount_7d",
        "streaming_features:max_txn_amount_7d",
        "batch_features:salary_delay_days",
        "batch_features:utility_payment_delay_avg",
        "batch_features:discretionary_spend_trend",
        "batch_features:credit_score",
        "batch_features:age",
        "batch_features:tenure_months",
        "batch_features:product_count",
        "batch_features:has_credit_card",
        "batch_features:has_personal_loan",
        "batch_features:has_mortgage",
        "batch_features:avg_monthly_spend_3m",
        "batch_features:spend_volatility_3m",
    ]

    features = store.get_online_features(
        features=feature_refs,
        entity_rows=entity_rows,
    ).to_df()

    return features


if __name__ == "__main__":
    print("Step 1: Exporting features to Parquet...")
    export_features_to_parquet()
    print("\nStep 2: Applying Feast definitions...")
    run_feast_apply()
    print("\nStep 3: Materializing to online store...")
    run_materialization()
    print("\nDone!")
