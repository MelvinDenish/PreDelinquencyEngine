"""
Apache Superset Configuration
Connects to PDI PostgreSQL database for advanced SQL-based analytics.
"""
import os

# ─────────────────────────────────────────────
# Superset core config
# ─────────────────────────────────────────────
ROW_LIMIT = 5000
SUPERSET_WEBSERVER_PORT = 8088

# Secret key - generate a real one for production
SECRET_KEY = os.getenv("SUPERSET_SECRET_KEY", "pdi-superset-secret-key-change-in-production")

# Database URI for Superset's own metadata
SQLALCHEMY_DATABASE_URI = os.getenv(
    "SUPERSET_METADATA_DB",
    "sqlite:////app/superset_home/superset.db"
)

# ─────────────────────────────────────────────
# PDI database connection (auto-registered)
# ─────────────────────────────────────────────
SQLALCHEMY_EXAMPLES_URI = None  # Disable examples

# Additional databases to register on startup
EXTRA_DATABASES = [
    {
        "database_name": "PDI PostgreSQL",
        "sqlalchemy_uri": os.getenv(
            "PDI_DB_URI",
            "postgresql://pdi_user:pdi_password@pdi-postgres:5432/pdi_db"
        ),
        "expose_in_sqllab": True,
        "allow_ctas": True,
        "allow_cvas": True,
        "allow_dml": False,
    }
]

# ─────────────────────────────────────────────
# Feature flags
# ─────────────────────────────────────────────
FEATURE_FLAGS = {
    "DASHBOARD_NATIVE_FILTERS": True,
    "DASHBOARD_CROSS_FILTERS": True,
    "DASHBOARD_NATIVE_FILTERS_SET": True,
    "ENABLE_TEMPLATE_PROCESSING": True,
    "ALERT_REPORTS": True,
}

# ─────────────────────────────────────────────
# Cache config
# ─────────────────────────────────────────────
CACHE_CONFIG = {
    "CACHE_TYPE": "RedisCache",
    "CACHE_DEFAULT_TIMEOUT": 300,
    "CACHE_KEY_PREFIX": "superset_",
    "CACHE_REDIS_HOST": os.getenv("REDIS_HOST", "pdi-redis"),
    "CACHE_REDIS_PORT": int(os.getenv("REDIS_PORT", 6379)),
    "CACHE_REDIS_DB": 2,
}

DATA_CACHE_CONFIG = {
    "CACHE_TYPE": "RedisCache",
    "CACHE_DEFAULT_TIMEOUT": 600,
    "CACHE_KEY_PREFIX": "superset_data_",
    "CACHE_REDIS_HOST": os.getenv("REDIS_HOST", "pdi-redis"),
    "CACHE_REDIS_PORT": int(os.getenv("REDIS_PORT", 6379)),
    "CACHE_REDIS_DB": 3,
}

# ─────────────────────────────────────────────
# Alerts & Reports
# ─────────────────────────────────────────────
ALERT_REPORTS_NOTIFICATION_DRY_RUN = True
WEBDRIVER_TYPE = "chrome"

# ─────────────────────────────────────────────
# SQL Lab
# ─────────────────────────────────────────────
SQLLAB_TIMEOUT = 300
SUPERSET_SQLLAB_TIMEOUT = 300

# ─────────────────────────────────────────────
# Pre-configured SQL queries for PDI
# ─────────────────────────────────────────────
PDI_PRESET_QUERIES = {
    "risk_overview": """
        SELECT
            risk_tier,
            COUNT(*) as customer_count,
            AVG(risk_score)::numeric(5,3) as avg_risk_score,
            AVG(credit_score_mapped) as avg_credit_score
        FROM risk_scores
        GROUP BY risk_tier
        ORDER BY avg_risk_score DESC
    """,
    "high_risk_customers": """
        SELECT
            rs.customer_id,
            c.first_name || ' ' || c.last_name as customer_name,
            c.city, c.region, c.income_bracket,
            rs.risk_score, rs.risk_tier, rs.credit_score_mapped,
            rs.scored_at
        FROM risk_scores rs
        JOIN customers c ON rs.customer_id = c.customer_id
        WHERE rs.risk_tier = 'critical'
        ORDER BY rs.risk_score DESC
        LIMIT 100
    """,
    "intervention_effectiveness": """
        SELECT
            i.intervention_type,
            COUNT(*) as total_sent,
            COUNT(CASE WHEN fe.outcome = 'positive' THEN 1 END) as positive_outcomes,
            ROUND(
                100.0 * COUNT(CASE WHEN fe.outcome = 'positive' THEN 1 END) / NULLIF(COUNT(*), 0),
                2
            ) as success_rate_pct
        FROM interventions i
        LEFT JOIN feedback_events fe ON i.customer_id = fe.customer_id
        GROUP BY i.intervention_type
        ORDER BY success_rate_pct DESC NULLS LAST
    """,
    "regional_risk_heatmap": """
        SELECT
            c.region, c.city,
            COUNT(DISTINCT c.customer_id) as total_customers,
            COUNT(DISTINCT CASE WHEN rs.risk_tier = 'critical' THEN rs.customer_id END) as critical_count,
            AVG(rs.risk_score)::numeric(5,3) as avg_risk
        FROM customers c
        LEFT JOIN risk_scores rs ON c.customer_id = rs.customer_id
        GROUP BY c.region, c.city
        ORDER BY avg_risk DESC NULLS LAST
    """,
}
