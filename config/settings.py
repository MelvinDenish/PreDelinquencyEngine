# pyre-ignore-all-errors
"""
Pre-Delinquency Intervention Engine - Centralized Configuration
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")


class PostgresConfig:
    HOST = os.getenv("POSTGRES_HOST", "localhost")
    PORT = int(os.getenv("POSTGRES_PORT", 5432))
    USER = os.getenv("POSTGRES_USER", "pdi_user")
    PASSWORD = os.getenv("POSTGRES_PASSWORD", "pdi_password")
    DB = os.getenv("POSTGRES_DB", "pdi_db")

    @classmethod
    def get_url(cls):
        return f"postgresql://{cls.USER}:{cls.PASSWORD}@{cls.HOST}:{cls.PORT}/{cls.DB}"

    @classmethod
    def get_async_url(cls):
        return f"postgresql+asyncpg://{cls.USER}:{cls.PASSWORD}@{cls.HOST}:{cls.PORT}/{cls.DB}"


class KafkaConfig:
    BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    INTERNAL_BOOTSTRAP_SERVERS = os.getenv("KAFKA_INTERNAL_BOOTSTRAP_SERVERS", "kafka:29092")

    # Topic names
    TOPIC_TRANSACTIONS = "transactions"
    TOPIC_ACCOUNT_UPDATES = "account_updates"
    TOPIC_RISK_SCORES = "risk_scores"
    TOPIC_INTERVENTIONS = "interventions"
    TOPIC_FEEDBACK = "feedback_events"
    TOPIC_ENRICHED_TRANSACTIONS = "enriched_transactions"
    TOPIC_FEATURES = "computed_features"


class RedisConfig:
    HOST = os.getenv("REDIS_HOST", "localhost")
    PORT = int(os.getenv("REDIS_PORT", 6379))
    DB = int(os.getenv("REDIS_DB", 0))

    @classmethod
    def get_url(cls):
        return f"redis://{cls.HOST}:{cls.PORT}/{cls.DB}"


class FlinkConfig:
    JOBMANAGER_HOST = os.getenv("FLINK_JOBMANAGER_HOST", "localhost")
    JOBMANAGER_PORT = int(os.getenv("FLINK_JOBMANAGER_PORT", 8081))

    @classmethod
    def get_rest_url(cls):
        return f"http://{cls.JOBMANAGER_HOST}:{cls.JOBMANAGER_PORT}"


class SparkConfig:
    MASTER_URL = os.getenv("SPARK_MASTER_URL", "spark://localhost:7077")
    MASTER_WEBUI = os.getenv("SPARK_MASTER_WEBUI", "http://localhost:8082")


class FeastConfig:
    REPO_PATH = os.getenv("FEAST_REPO_PATH", str(PROJECT_ROOT / "feature_store" / "feature_repo"))


class MLflowConfig:
    TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "pdi_delinquency_prediction")


class DebeziumConfig:
    CONNECT_URL = os.getenv("DEBEZIUM_CONNECT_URL", "http://localhost:8083")


class CeleryConfig:
    BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/1")
    RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/2")


class ScoringConfig:
    HOST = os.getenv("SCORING_SERVICE_HOST", "0.0.0.0")
    PORT = int(os.getenv("SCORING_SERVICE_PORT", 8000))


class DashboardConfig:
    HOST = os.getenv("DASHBOARD_HOST", "0.0.0.0")
    PORT = int(os.getenv("DASHBOARD_PORT", 8050))
    DEBUG = os.getenv("DASHBOARD_DEBUG", "True").lower() == "true"


class ModelConfig:
    # 4-model ensemble weights (XGBoost + LightGBM + LSTM + TFT)
    ENSEMBLE_XGB_WEIGHT = float(os.getenv("ENSEMBLE_XGB_WEIGHT", 0.30))
    ENSEMBLE_LGB_WEIGHT = float(os.getenv("ENSEMBLE_LGB_WEIGHT", 0.20))
    ENSEMBLE_LSTM_WEIGHT = float(os.getenv("ENSEMBLE_LSTM_WEIGHT", 0.15))
    ENSEMBLE_TFT_WEIGHT = float(os.getenv("ENSEMBLE_TFT_WEIGHT", 0.35))
    USE_META_LEARNER = os.getenv("USE_META_LEARNER", "True").lower() == "true"
    RISK_CRITICAL_THRESHOLD = float(os.getenv("RISK_CRITICAL_THRESHOLD", 0.7))
    RISK_WATCH_THRESHOLD = float(os.getenv("RISK_WATCH_THRESHOLD", 0.5))
    COOLDOWN_DAYS = int(os.getenv("COOLDOWN_DAYS", 7))

    # Feature list used by models (must be in exact order)
    FEATURE_COLUMNS = [
        "discretionary_spend_7d",
        "discretionary_spend_30d",
        "atm_withdrawals_count_7d",
        "atm_withdrawals_count_30d",
        "lending_app_txn_count_7d",
        "lending_app_txn_count_30d",
        "weighted_lending_risk_7d",
        "weighted_lending_risk_30d",
        "savings_balance_pct_change_7d",
        "failed_autodebits_count_7d",
        "failed_autodebits_count_30d",
        "total_spend_7d",
        "total_spend_30d",
        "txn_count_7d",
        "txn_count_30d",
        "avg_txn_amount_7d",
        "max_txn_amount_7d",
        "salary_delay_days",
        "utility_payment_delay_avg",
        "discretionary_spend_trend",
        "credit_score",
        "age",
        "tenure_months",
        "product_count",
        "has_credit_card",
        "has_personal_loan",
        "has_mortgage",
        "avg_monthly_spend_3m",
        "spend_volatility_3m",
        # M2: Asset-side features
        "fd_closed_count_90d",
        "fd_closure_amount_90d",
        "sip_stopped_flag",
        "sip_gaps_3m",
        "insurance_lapse_flag",
        "insurance_missed_payments_3m",
        # M3: Employer health
        "employer_health_score",
    ]


class DataGenConfig:
    NUM_CUSTOMERS = int(os.getenv("NUM_CUSTOMERS", 1000))
    TRANSACTION_MONTHS = int(os.getenv("TRANSACTION_MONTHS", 6))
    STRESS_CUSTOMER_PCT = float(os.getenv("STRESS_CUSTOMER_PCT", 0.20))
