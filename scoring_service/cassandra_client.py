"""
Cassandra Risk Score Storage Client
Persists delinquency risk scores to Apache Cassandra for high-throughput,
time-series risk score history as specified in the PreDelinquency architecture.
"""
import logging
import os
from datetime import datetime
from typing import Optional, List, Dict
import uuid

logger = logging.getLogger(__name__)

# Cassandra host from environment (Docker service name or localhost)
CASSANDRA_HOST = os.getenv("CASSANDRA_HOST", "localhost")
CASSANDRA_PORT = int(os.getenv("CASSANDRA_PORT", "9042"))
CASSANDRA_KEYSPACE = "pdi"

_session = None
_cluster = None


def get_session():
    """Return a singleton Cassandra session."""
    global _session, _cluster
    if _session is not None:
        return _session
    try:
        from cassandra.cluster import Cluster
        from cassandra.policies import DCAwareRoundRobinPolicy

        _cluster = Cluster(
            contact_points=[CASSANDRA_HOST],
            port=CASSANDRA_PORT,
            load_balancing_policy=DCAwareRoundRobinPolicy(local_dc="datacenter1"),
            connect_timeout=10,
        )
        _session = _cluster.connect()
        _ensure_schema(_session)
        logger.info(f"Connected to Cassandra at {CASSANDRA_HOST}:{CASSANDRA_PORT}")
    except Exception as e:
        logger.warning(f"Cassandra unavailable — risk scores will not be written to Cassandra: {e}")
        _session = None
    return _session


def _ensure_schema(session):
    """Create keyspace and tables if they don't exist."""
    session.execute(f"""
        CREATE KEYSPACE IF NOT EXISTS {CASSANDRA_KEYSPACE}
        WITH replication = {{
            'class': 'SimpleStrategy',
            'replication_factor': 1
        }}
    """)

    session.set_keyspace(CASSANDRA_KEYSPACE)

    # Main risk scores table — partitioned by customer, clustered by time (newest first)
    session.execute("""
        CREATE TABLE IF NOT EXISTS risk_scores (
            customer_id     TEXT,
            scored_at       TIMESTAMP,
            score_id        UUID,
            risk_score      DOUBLE,
            risk_tier       TEXT,
            credit_score    INT,
            xgboost_score   DOUBLE,
            lightgbm_score  DOUBLE,
            lstm_score      DOUBLE,
            ensemble_score  DOUBLE,
            top_features    TEXT,
            model_version   TEXT,
            PRIMARY KEY (customer_id, scored_at, score_id)
        ) WITH CLUSTERING ORDER BY (scored_at DESC, score_id DESC)
          AND default_time_to_live = 7776000
    """)

    # Summary table — latest score per customer (for fast lookups)
    session.execute("""
        CREATE TABLE IF NOT EXISTS risk_scores_latest (
            customer_id     TEXT PRIMARY KEY,
            scored_at       TIMESTAMP,
            risk_score      DOUBLE,
            risk_tier       TEXT,
            credit_score    INT,
            ensemble_score  DOUBLE,
            top_features    TEXT,
            model_version   TEXT
        )
    """)

    # Tier summary table — all customers in a given risk tier
    session.execute("""
        CREATE TABLE IF NOT EXISTS risk_scores_by_tier (
            risk_tier       TEXT,
            scored_at       TIMESTAMP,
            customer_id     TEXT,
            risk_score      DOUBLE,
            PRIMARY KEY (risk_tier, scored_at, customer_id)
        ) WITH CLUSTERING ORDER BY (scored_at DESC, customer_id ASC)
          AND default_time_to_live = 2592000
    """)

    logger.info(f"Cassandra schema ready in keyspace '{CASSANDRA_KEYSPACE}'")


def write_risk_score(
    customer_id: str,
    risk_score: float,
    risk_tier: str,
    credit_score: int,
    xgboost_score: Optional[float] = None,
    lightgbm_score: Optional[float] = None,
    lstm_score: Optional[float] = None,
    ensemble_score: Optional[float] = None,
    top_features: Optional[str] = None,
    model_version: str = "v2.0",
) -> bool:
    """
    Write a risk score to Cassandra (risk_scores + risk_scores_latest + risk_scores_by_tier).
    Returns True on success, False on failure (non-blocking).
    """
    session = get_session()
    if session is None:
        return False

    try:
        import json
        scored_at = datetime.utcnow()
        score_id = uuid.uuid4()

        # Main history table
        session.execute(
            f"""
            INSERT INTO {CASSANDRA_KEYSPACE}.risk_scores
              (customer_id, scored_at, score_id, risk_score, risk_tier, credit_score,
               xgboost_score, lightgbm_score, lstm_score, ensemble_score,
               top_features, model_version)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                customer_id, scored_at, score_id,
                risk_score, risk_tier, credit_score,
                xgboost_score, lightgbm_score, lstm_score,
                ensemble_score or risk_score,
                json.dumps(top_features) if top_features and not isinstance(top_features, str) else top_features,
                model_version,
            ),
        )

        # Latest lookup
        session.execute(
            f"""
            INSERT INTO {CASSANDRA_KEYSPACE}.risk_scores_latest
              (customer_id, scored_at, risk_score, risk_tier, credit_score,
               ensemble_score, top_features, model_version)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                customer_id, scored_at, risk_score, risk_tier, credit_score,
                ensemble_score or risk_score, top_features, model_version,
            ),
        )

        # Tier index
        session.execute(
            f"""
            INSERT INTO {CASSANDRA_KEYSPACE}.risk_scores_by_tier
              (risk_tier, scored_at, customer_id, risk_score)
            VALUES (%s, %s, %s, %s)
            """,
            (risk_tier, scored_at, customer_id, risk_score),
        )

        logger.debug(f"Risk score written to Cassandra for customer {customer_id}")
        return True

    except Exception as e:
        logger.warning(f"Cassandra write failed for {customer_id}: {e}")
        return False


def get_latest_score(customer_id: str) -> Optional[Dict]:
    """Fetch the latest risk score for a customer from Cassandra."""
    session = get_session()
    if session is None:
        return None
    try:
        row = session.execute(
            f"SELECT * FROM {CASSANDRA_KEYSPACE}.risk_scores_latest WHERE customer_id = %s",
            (customer_id,),
        ).one()
        if row:
            return {
                "customer_id": row.customer_id,
                "risk_score": row.risk_score,
                "risk_tier": row.risk_tier,
                "credit_score": row.credit_score,
                "ensemble_score": row.ensemble_score,
                "scored_at": row.scored_at.isoformat() if row.scored_at else None,
            }
    except Exception as e:
        logger.warning(f"Cassandra read failed: {e}")
    return None


def get_score_history(customer_id: str, limit: int = 30) -> List[Dict]:
    """Fetch risk score history for a customer from Cassandra."""
    session = get_session()
    if session is None:
        return []
    try:
        rows = session.execute(
            f"""SELECT customer_id, scored_at, risk_score, risk_tier, credit_score,
                       ensemble_score, top_features
                FROM {CASSANDRA_KEYSPACE}.risk_scores
                WHERE customer_id = %s
                LIMIT %s""",
            (customer_id, limit),
        )
        return [
            {
                "customer_id": r.customer_id,
                "risk_score": r.risk_score,
                "risk_tier": r.risk_tier,
                "credit_score": r.credit_score,
                "ensemble_score": r.ensemble_score,
                "top_features": r.top_features,
                "scored_at": r.scored_at.isoformat() if r.scored_at else None,
            }
            for r in rows
        ]
    except Exception as e:
        logger.warning(f"Cassandra history read failed: {e}")
        return []


def get_customers_by_tier(risk_tier: str, limit: int = 100) -> List[Dict]:
    """Fetch customers in a given risk tier from Cassandra."""
    session = get_session()
    if session is None:
        return []
    try:
        rows = session.execute(
            f"""SELECT customer_id, risk_score, scored_at
                FROM {CASSANDRA_KEYSPACE}.risk_scores_by_tier
                WHERE risk_tier = %s
                LIMIT %s""",
            (risk_tier, limit),
        )
        return [
            {
                "customer_id": r.customer_id,
                "risk_score": r.risk_score,
                "scored_at": r.scored_at.isoformat() if r.scored_at else None,
            }
            for r in rows
        ]
    except Exception as e:
        logger.warning(f"Cassandra tier query failed: {e}")
        return []


def close():
    """Close the Cassandra cluster connection."""
    global _session, _cluster
    if _cluster:
        _cluster.shutdown()
        _cluster = None
        _session = None
        logger.info("Cassandra connection closed")
