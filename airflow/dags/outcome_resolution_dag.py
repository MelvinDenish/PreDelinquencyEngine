# pyre-ignore-all-errors
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

def compute_ab_lift(**kwargs):
    """P4: Compute A/B holdout lift with statistical significance."""
    import sys, os, logging
    import psycopg2

    sys.path.insert(0, "/app")
    from config.settings import PostgresConfig
    from ml.ab_holdout import ABHoldout

    logger = logging.getLogger(__name__)
    holdout = ABHoldout()

    conn = psycopg2.connect(
        host=PostgresConfig.HOST, port=PostgresConfig.PORT,
        user=PostgresConfig.USER, password=PostgresConfig.PASSWORD,
        dbname=PostgresConfig.DB,
    )
    cursor = conn.cursor()

    # Get treatment outcomes (intervened customers)
    cursor.execute("""
        SELECT i.customer_id,
               CASE WHEN i.outcome IN ('recovered','improved') THEN 1 ELSE 0 END AS success
        FROM interventions i
        JOIN ab_holdout_assignments a ON i.customer_id = a.customer_id
        WHERE a.holdout_group = 'treatment'
          AND i.outcome IS NOT NULL
          AND i.sent_at >= NOW() - INTERVAL '90 days'
    """)
    treatment = cursor.fetchall()

    # Get control outcomes (no intervention)
    cursor.execute("""
        SELECT a.customer_id,
               CASE WHEN rs.risk_score < 0.50 THEN 1 ELSE 0 END AS success
        FROM ab_holdout_assignments a
        JOIN risk_scores rs ON a.customer_id = rs.customer_id
        WHERE a.holdout_group = 'control'
          AND rs.scored_at >= NOW() - INTERVAL '90 days'
    """)
    control = cursor.fetchall()

    cursor.close()
    conn.close()

    treatment_rates = [r[1] for r in treatment]
    control_rates = [r[1] for r in control]

    lift = holdout.compute_lift(treatment_rates, control_rates)
    logger.info(f"[A/B Lift] Treatment: {lift.get('treatment_rate', 0):.3f}, "
                f"Control: {lift.get('control_rate', 0):.3f}, "
                f"Lift: {lift.get('lift_pct', 0):.1f}%, p={lift.get('p_value', 1):.4f}")
    return lift


def tag_self_cures(**kwargs):
    """P7: Identify customers who recovered without any intervention."""
    import sys, os, logging
    import psycopg2

    sys.path.insert(0, "/app")
    from config.settings import PostgresConfig
    from ml.self_cure_detector import SelfCureDetector

    logger = logging.getLogger(__name__)
    detector = SelfCureDetector()

    conn = psycopg2.connect(
        host=PostgresConfig.HOST, port=PostgresConfig.PORT,
        user=PostgresConfig.USER, password=PostgresConfig.PASSWORD,
        dbname=PostgresConfig.DB,
    )
    cursor = conn.cursor()

    # Find customers whose risk dropped significantly without intervention
    cursor.execute("""
        SELECT DISTINCT rs1.customer_id, rs1.risk_score AS old_score, rs2.risk_score AS new_score
        FROM risk_scores rs1
        JOIN risk_scores rs2 ON rs1.customer_id = rs2.customer_id
        LEFT JOIN interventions i ON rs1.customer_id = i.customer_id
            AND i.sent_at BETWEEN rs1.scored_at AND rs2.scored_at
        WHERE rs1.risk_score >= 0.50
          AND rs2.risk_score < 0.35
          AND rs2.scored_at > rs1.scored_at
          AND rs2.scored_at >= NOW() - INTERVAL '60 days'
          AND i.id IS NULL
        LIMIT 5000
    """)
    candidates = cursor.fetchall()

    tagged = 0
    for cid, old_score, new_score in candidates:
        try:
            detector.tag_self_cure(cursor, cid, float(old_score), float(new_score))
            tagged += 1
        except Exception:
            pass

    conn.commit()
    cursor.close()
    conn.close()

    logger.info(f"[SelfCure] Tagged {tagged} self-cure events")
    return {"self_cures_tagged": tagged}


def update_bandit_rewards(**kwargs):
    """P14: Feed intervention outcomes back to LinUCB channel bandit."""
    import sys, os, logging
    import psycopg2

    sys.path.insert(0, "/app")
    from config.settings import PostgresConfig

    logger = logging.getLogger(__name__)

    try:
        from ml.channel_bandit import LinUCBChannelBandit
    except ImportError:
        logger.warning("[Bandit] channel_bandit module not available")
        return {"updated": 0}

    bandit_path = "/app/models/channel_bandit.joblib"
    bandit = LinUCBChannelBandit()
    if os.path.exists(bandit_path):
        bandit.load(bandit_path)

    conn = psycopg2.connect(
        host=PostgresConfig.HOST, port=PostgresConfig.PORT,
        user=PostgresConfig.USER, password=PostgresConfig.PASSWORD,
        dbname=PostgresConfig.DB,
    )
    cursor = conn.cursor()

    # Get recently resolved interventions with known outcomes
    cursor.execute("""
        SELECT i.customer_id, i.channel, i.outcome,
               c.age, c.income_bracket, c.region, c.credit_score
        FROM interventions i
        JOIN customers c ON i.customer_id = c.customer_id
        WHERE i.outcome IS NOT NULL
          AND i.outcome_resolved_at >= NOW() - INTERVAL '7 days'
        LIMIT 5000
    """)
    rows = cursor.fetchall()

    updated = 0
    for cid, channel, outcome, age, income, region, credit in rows:
        reward = 1.0 if outcome in ("recovered", "improved") else 0.0
        ctx = {
            "age": age or 35, "income_bracket": income or "mid",
            "risk_score": 0.5, "segment_type": "salaried",
            "tenure_months": 24, "num_dependents": 1,
            "region": region or "metro", "credit_score": credit or 700,
        }
        try:
            bandit.update(ctx, channel, reward)
            updated += 1
        except Exception:
            pass

    cursor.close()
    conn.close()

    # Persist updated bandit weights
    try:
        bandit.save(bandit_path)
    except Exception:
        pass

    logger.info(f"[Bandit] Updated {updated} reward observations")
    return {"bandit_updates": updated}


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

    ab_lift = PythonOperator(
        task_id="compute_ab_lift",
        python_callable=compute_ab_lift,
    )

    self_cure = PythonOperator(
        task_id="tag_self_cures",
        python_callable=tag_self_cures,
    )

    bandit_update = PythonOperator(
        task_id="update_bandit_rewards",
        python_callable=update_bandit_rewards,
    )

    resolve >> [effectiveness, ab_lift, self_cure, bandit_update]
