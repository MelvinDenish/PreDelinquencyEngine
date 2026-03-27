"""
Outcome Resolution DAG (M11)
Daily Airflow DAG that tracks intervention outcomes:
1. Checks if intervened customers subsequently defaulted or recovered
2. Updates intervention records with outcome data
3. Computes intervention effectiveness metrics
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "pdi_engine",
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
    "depends_on_past": False,
}


def resolve_outcomes(**kwargs):
    """Check intervention outcomes and update records."""
    import sys, os, logging
    import psycopg2

    sys.path.insert(0, "/app")
    from config.settings import PostgresConfig

    logger = logging.getLogger(__name__)

    conn = psycopg2.connect(
        host=PostgresConfig.HOST, port=PostgresConfig.PORT,
        user=PostgresConfig.USER, password=PostgresConfig.PASSWORD,
        dbname=PostgresConfig.DB,
    )
    cursor = conn.cursor()

    # Find interventions sent 30+ days ago without outcome resolution
    cursor.execute("""
        SELECT i.id, i.customer_id, i.intervention_type, i.channel, i.sent_at
        FROM interventions i
        WHERE i.sent_at < NOW() - INTERVAL '30 days'
          AND i.outcome IS NULL
        LIMIT 5000
    """)
    pending = cursor.fetchall()
    logger.info(f"[OutcomeResolution] Processing {len(pending)} pending interventions")

    resolved = 0
    for int_id, cid, int_type, channel, sent_at in pending:
        # Check if customer defaulted after intervention
        cursor.execute("""
            SELECT COUNT(*) FROM feedback_events
            WHERE customer_id = %s
              AND event_type = 'default'
              AND event_timestamp > %s
        """, (cid, sent_at))
        default_count = cursor.fetchone()[0]

        # Check if customer's risk improved
        cursor.execute("""
            SELECT risk_score FROM risk_scores
            WHERE customer_id = %s
            ORDER BY scored_at DESC LIMIT 1
        """, (cid,))
        latest = cursor.fetchone()
        current_score = latest[0] if latest else 0.5

        cursor.execute("""
            SELECT risk_score FROM risk_scores
            WHERE customer_id = %s AND scored_at <= %s
            ORDER BY scored_at DESC LIMIT 1
        """, (cid, sent_at))
        baseline = cursor.fetchone()
        baseline_score = baseline[0] if baseline else 0.5

        # Determine outcome
        if default_count > 0:
            outcome = "defaulted"
        elif float(current_score) < float(baseline_score) * 0.85:
            outcome = "recovered"
        elif float(current_score) < float(baseline_score):
            outcome = "improved"
        else:
            outcome = "no_change"

        cursor.execute("""
            UPDATE interventions
            SET outcome = %s,
                outcome_resolved_at = NOW(),
                risk_score_before = %s,
                risk_score_after = %s
            WHERE id = %s
        """, (outcome, float(baseline_score), float(current_score), int_id))
        resolved += 1

    conn.commit()
    cursor.close()
    conn.close()

    logger.info(f"[OutcomeResolution] Resolved {resolved} interventions")
    return {"resolved": resolved}


def compute_effectiveness_metrics(**kwargs):
    """Compute channel-level intervention effectiveness."""
    import sys, os, logging
    import psycopg2

    sys.path.insert(0, "/app")
    from config.settings import PostgresConfig

    logger = logging.getLogger(__name__)

    conn = psycopg2.connect(
        host=PostgresConfig.HOST, port=PostgresConfig.PORT,
        user=PostgresConfig.USER, password=PostgresConfig.PASSWORD,
        dbname=PostgresConfig.DB,
    )
    cursor = conn.cursor()

    cursor.execute("""
        SELECT channel,
               COUNT(*) as total,
               SUM(CASE WHEN outcome = 'recovered' THEN 1 ELSE 0 END) as recovered,
               SUM(CASE WHEN outcome = 'improved' THEN 1 ELSE 0 END) as improved,
               SUM(CASE WHEN outcome = 'defaulted' THEN 1 ELSE 0 END) as defaulted,
               AVG(risk_score_before - risk_score_after) as avg_risk_reduction
        FROM interventions
        WHERE outcome IS NOT NULL
          AND sent_at >= NOW() - INTERVAL '90 days'
        GROUP BY channel
    """)

    metrics = {}
    for channel, total, recovered, improved, defaulted, avg_reduction in cursor.fetchall():
        effectiveness = (recovered + improved) / max(total, 1) * 100
        metrics[channel] = {
            "total": total,
            "recovered": recovered,
            "improved": improved,
            "defaulted": defaulted,
            "effectiveness_pct": round(effectiveness, 1),
            "avg_risk_reduction": round(float(avg_reduction or 0), 4),
        }
        logger.info(
            f"[OutcomeResolution] {channel}: {effectiveness:.1f}% effective "
            f"({recovered + improved}/{total})"
        )

    cursor.close()
    conn.close()
    return metrics


with DAG(
    dag_id="pdi_outcome_resolution",
    default_args=default_args,
    description="Daily outcome tracking for intervention effectiveness",
    schedule_interval="0 6 * * *",  # 6 AM daily
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["pdi", "intervention", "outcomes"],
) as dag:

    resolve = PythonOperator(
        task_id="resolve_outcomes",
        python_callable=resolve_outcomes,
    )

    effectiveness = PythonOperator(
        task_id="compute_effectiveness_metrics",
        python_callable=compute_effectiveness_metrics,
    )

    resolve >> effectiveness
