"""
Kafka Producer - Publishes real-time events to Kafka topics.
Supports transaction events, account updates, risk scores, and feedback events.
"""
import json
import logging
from typing import Dict, Optional

from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config.settings import KafkaConfig

logger = logging.getLogger(__name__)


class PDIKafkaProducer:
    """Enterprise-grade Kafka producer for the PDI engine."""

    def __init__(self, bootstrap_servers: str = None):
        self.bootstrap_servers = bootstrap_servers or KafkaConfig.BOOTSTRAP_SERVERS
        self._producer = None

    def _get_producer(self) -> KafkaProducer:
        if self._producer is None:
            self._producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if k else None,
                acks="all",
                retries=3,
                max_in_flight_requests_per_connection=1,
                linger_ms=10,
                batch_size=16384,
                compression_type="gzip",
            )
        return self._producer

    def publish_transaction(self, transaction: Dict):
        """Publish a transaction event."""
        producer = self._get_producer()
        producer.send(
            KafkaConfig.TOPIC_TRANSACTIONS,
            key=transaction.get("customer_id"),
            value=transaction,
        )
        logger.debug(f"Published transaction {transaction.get('txn_id')} to Kafka")

    def publish_account_update(self, update: Dict):
        """Publish an account balance update."""
        producer = self._get_producer()
        producer.send(
            KafkaConfig.TOPIC_ACCOUNT_UPDATES,
            key=update.get("customer_id"),
            value=update,
        )

    def publish_risk_score(self, score_event: Dict):
        """Publish a risk score event."""
        producer = self._get_producer()
        producer.send(
            KafkaConfig.TOPIC_RISK_SCORES,
            key=score_event.get("customer_id"),
            value=score_event,
        )

    def publish_intervention(self, intervention: Dict):
        """Publish an intervention event."""
        producer = self._get_producer()
        producer.send(
            KafkaConfig.TOPIC_INTERVENTIONS,
            key=intervention.get("customer_id"),
            value=intervention,
        )

    def publish_feedback(self, feedback: Dict):
        """Publish a feedback event."""
        producer = self._get_producer()
        producer.send(
            KafkaConfig.TOPIC_FEEDBACK,
            key=feedback.get("customer_id"),
            value=feedback,
        )

    def flush(self):
        if self._producer:
            self._producer.flush()

    def close(self):
        if self._producer:
            self._producer.flush()
            self._producer.close()
            self._producer = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
