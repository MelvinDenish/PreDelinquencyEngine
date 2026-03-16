"""
Feedback Consumer
Consumes feedback events from Kafka and updates outcome labels
in the offline store for model retraining.
"""
import os
import sys
import json
import logging
from datetime import datetime

import psycopg2
from kafka import KafkaConsumer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config.settings import KafkaConfig, PostgresConfig

logger = logging.getLogger(__name__)


def process_feedback_event(event: dict):
    """
    Process a single feedback event.
    Updates interventions table and feedback_events table.
    """
    customer_id = event.get("customer_id")
    intervention_id = event.get("intervention_id")
    outcome = event.get("outcome")  # paid, restructured, defaulted, no_response

    if not customer_id or not outcome:
        logger.warning(f"Invalid feedback event: {event}")
        return

    conn = psycopg2.connect(
        host=PostgresConfig.HOST, port=PostgresConfig.PORT,
        user=PostgresConfig.USER, password=PostgresConfig.PASSWORD,
        dbname=PostgresConfig.DB,
    )
    cursor = conn.cursor()

    # Update intervention outcome
    if intervention_id:
        cursor.execute(
            """UPDATE interventions
               SET outcome = %s, responded_at = NOW()
               WHERE id = %s""",
            (outcome, intervention_id)
        )

    # Map outcome to label for ML training
    label = 1 if outcome in ("defaulted",) else 0

    # Insert feedback event
    cursor.execute(
        """INSERT INTO feedback_events
           (customer_id, intervention_id, outcome, label, event_timestamp)
           VALUES (%s, %s, %s, %s, NOW())""",
        (customer_id, intervention_id, outcome, label)
    )

    conn.commit()
    cursor.close()
    conn.close()

    logger.info(f"[Feedback] Processed: {customer_id} -> {outcome} (label={label})")


def start_feedback_consumer():
    """Start consuming feedback events from Kafka."""
    consumer = KafkaConsumer(
        KafkaConfig.TOPIC_FEEDBACK,
        bootstrap_servers=KafkaConfig.BOOTSTRAP_SERVERS,
        group_id="feedback-consumer",
        auto_offset_reset="earliest",
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    )

    logger.info("[Feedback] Consumer started, listening for feedback events...")

    try:
        for message in consumer:
            process_feedback_event(message.value)
    except KeyboardInterrupt:
        logger.info("[Feedback] Consumer stopped")
    finally:
        consumer.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    start_feedback_consumer()

