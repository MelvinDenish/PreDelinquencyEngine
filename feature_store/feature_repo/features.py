"""
Feast Feature Definitions
Defines entities, data sources, and feature views for streaming and batch features.
"""
from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, Float64, Int32, Int64, String, Bool


# ─────────────────────────────────────────────
# Entity Definition
# ─────────────────────────────────────────────
customer = Entity(
    name="customer_id",
    value_type=ValueType.STRING,
    description="Unique customer identifier",
)


# ─────────────────────────────────────────────
# Data Sources
# ─────────────────────────────────────────────
streaming_features_source = FileSource(
    path="data/streaming_features.parquet",
    timestamp_field="updated_at",
    created_timestamp_column="updated_at",
)

batch_features_source = FileSource(
    path="data/batch_features.parquet",
    timestamp_field="updated_at",
    created_timestamp_column="updated_at",
)


# ─────────────────────────────────────────────
# Feature Views
# ─────────────────────────────────────────────
streaming_feature_view = FeatureView(
    name="streaming_features",
    entities=[customer],
    ttl=timedelta(days=7),
    schema=[
        Field(name="discretionary_spend_7d", dtype=Float64),
        Field(name="discretionary_spend_30d", dtype=Float64),
        Field(name="atm_withdrawals_count_7d", dtype=Int32),
        Field(name="atm_withdrawals_count_30d", dtype=Int32),
        Field(name="lending_app_txn_count_7d", dtype=Int32),
        Field(name="lending_app_txn_count_30d", dtype=Int32),
        Field(name="weighted_lending_risk_7d", dtype=Float64),
        Field(name="weighted_lending_risk_30d", dtype=Float64),
        Field(name="savings_balance_pct_change_7d", dtype=Float64),
        Field(name="failed_autodebits_count_7d", dtype=Int32),
        Field(name="failed_autodebits_count_30d", dtype=Int32),
        Field(name="total_spend_7d", dtype=Float64),
        Field(name="total_spend_30d", dtype=Float64),
        Field(name="txn_count_7d", dtype=Int32),
        Field(name="txn_count_30d", dtype=Int32),
        Field(name="avg_txn_amount_7d", dtype=Float64),
        Field(name="max_txn_amount_7d", dtype=Float64),
    ],
    source=streaming_features_source,
    online=True,
)

batch_feature_view = FeatureView(
    name="batch_features",
    entities=[customer],
    ttl=timedelta(days=30),
    schema=[
        Field(name="salary_delay_days", dtype=Int32),
        Field(name="utility_payment_delay_avg", dtype=Float64),
        Field(name="discretionary_spend_trend", dtype=Float64),
        Field(name="credit_score", dtype=Int32),
        Field(name="age", dtype=Int32),
        Field(name="tenure_months", dtype=Int32),
        Field(name="product_count", dtype=Int32),
        Field(name="has_credit_card", dtype=Bool),
        Field(name="has_personal_loan", dtype=Bool),
        Field(name="has_mortgage", dtype=Bool),
        Field(name="avg_monthly_spend_3m", dtype=Float64),
        Field(name="spend_volatility_3m", dtype=Float64),
    ],
    source=batch_features_source,
    online=True,
)
