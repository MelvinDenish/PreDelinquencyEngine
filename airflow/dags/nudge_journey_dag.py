# pyre-ignore-all-errors
"""
Airflow DAG — Nudge Journey Daily Executor — P2
Executes pending nudge journey steps and escalates/cancels based on latest risk scores.
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import logging
import sys
import os

sys.path.insert(0, "/opt/airflow/pdi")
logger = logging.getLogger(__name__)

default_args = {
    "owner": "pdi_engine",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
    "email_on_failure": False,
}

dag = DAG(
    "nudge_journey_executor",
    default_args=default_args,
    description="Execute pending nudge journey steps and manage journey lifecycle",
    schedule_interval="0 9 * * *",   # 09:00 IST daily (03:30 UTC)
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["intervention", "nudge_journey", "pdi"],
)


def execute_pending_steps(**context):
    """
    Find and execute all nudge journey steps scheduled for today.
    Dispatches via existing notification_dispatcher.
    """
    from sqlalchemy import create_engine
    from config.settings import PostgresConfig
    from intervention.notification_dispatcher import NotificationDispatcher
    import pandas as pd

    engine = create_engine(PostgresConfig.get_url())
    dispatcher = NotificationDispatcher()

    # Fetch steps due today (±1 hour window for safety)
    due_steps = pd.read_sql("""
        SELECT nj.*, c.first_name, c.last_name, c.phone, c.email,
               c.employment_type, c.monthly_salary,
               r.risk_score, r.risk_tier
        FROM nudge_journeys nj
        JOIN customers c ON c.customer_id = nj.customer_id
        LEFT JOIN LATERAL (
            SELECT risk_score, risk_tier FROM risk_scores rs
            WHERE rs.customer_id = nj.customer_id
            ORDER BY scored_at DESC LIMIT 1
        ) r ON true
        WHERE nj.status = 'pending'
          AND nj.scheduled_at <= NOW() + INTERVAL '1 hour'
          AND nj.scheduled_at >= NOW() - INTERVAL '1 hour'
        ORDER BY nj.customer_id, nj.step_number
    """, engine)

    logger.info(f"[NudgeJourney] {len(due_steps)} steps to execute")
    executed = 0

    for _, step in due_steps.iterrows():
        try:
            # Check if customer has already paid/recovered since journey start
            latest_tier = step.get("risk_tier", "watch")
            if latest_tier == "stable":
                # Cancel remaining journey steps
                from intervention.nudge_journey import NudgeJourneyOrchestrator
                NudgeJourneyOrchestrator().cancel_journey(
                    step["customer_id"], reason="resolved"
                )
                continue

            # Dispatch the step
            customer = step.to_dict()
            result = dispatcher.dispatch(
                customer_id=step["customer_id"],
                customer=customer,
                channel=step["channel"],
                message_type=step["message_type"],
                risk_tier=latest_tier,
                risk_score=float(step.get("risk_score", 0.5) or 0.5),
                journey_id=step["journey_id"],
                journey_step=int(step["step_number"]),
            )

            # Mark step as sent
            with engine.connect() as conn:
                conn.execute("""
                    UPDATE nudge_journeys
                    SET status = 'sent', sent_at = NOW(),
                        dispatch_result = %s
                    WHERE journey_id = %s AND step_number = %s
                """, (str(result), step["journey_id"], step["step_number"]))

            executed += 1

        except Exception as e:
            logger.error(f"[NudgeJourney] Step failed for {step['customer_id']}: {e}")

    logger.info(f"[NudgeJourney] Executed {executed}/{len(due_steps)} steps")
    return {"executed": executed, "total_due": len(due_steps)}


def expire_stale_journeys(**context):
    """Mark journeys as expired if customer hasn't been scored in 45 days."""
    from sqlalchemy import create_engine
    from config.settings import PostgresConfig

    engine = create_engine(PostgresConfig.get_url())
    with engine.connect() as conn:
        result = conn.execute("""
            UPDATE nudge_journeys
            SET status = 'expired'
            WHERE status = 'pending'
              AND created_at < NOW() - INTERVAL '45 days'
        """)
    logger.info(f"[NudgeJourney] Expired {result.rowcount} stale journey steps")


def escalate_critical_journeys(**context):
    """
    If a customer escalated to critical mid-journey, replace pending watch-tier
    steps with the critical-tier template.
    """
    from sqlalchemy import create_engine
    from config.settings import PostgresConfig
    from intervention.nudge_journey import NudgeJourneyOrchestrator
    import pandas as pd

    engine = create_engine(PostgresConfig.get_url())
    orchestrator = NudgeJourneyOrchestrator()

    # Find customers in watch-journey who are now critical
    escalated = pd.read_sql("""
        SELECT DISTINCT nj.customer_id, r.risk_score
        FROM nudge_journeys nj
        JOIN LATERAL (
            SELECT risk_score, risk_tier FROM risk_scores rs
            WHERE rs.customer_id = nj.customer_id
            ORDER BY scored_at DESC LIMIT 1
        ) r ON true
        WHERE nj.status = 'pending'
          AND nj.risk_tier_at_entry = 'watch'
          AND r.risk_tier = 'critical'
    """, engine)

    for _, row in escalated.iterrows():
        cid = row["customer_id"]
        orchestrator.cancel_journey(cid, reason="escalated_to_critical")
        orchestrator.create_journey(
            customer_id=cid,
            risk_tier="critical",
            risk_score=float(row["risk_score"]),
        )
        logger.info(f"[NudgeJourney] Escalated journey for {cid}")


with dag:
    t1 = PythonOperator(
        task_id="execute_pending_steps",
        python_callable=execute_pending_steps,
    )
    t2 = PythonOperator(
        task_id="escalate_critical_journeys",
        python_callable=escalate_critical_journeys,
    )
    t3 = PythonOperator(
        task_id="expire_stale_journeys",
        python_callable=expire_stale_journeys,
    )

    t1 >> t2 >> t3
