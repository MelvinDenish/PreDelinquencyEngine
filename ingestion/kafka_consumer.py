# pyre-ignore-all-errors
"""
Kafka Consumer - Base consumer utilities for consuming from Kafka topics.
"""
import json
import logging
from typing import Callable, Dict, List, Optional

from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config.settings import KafkaConfig

logger = logging.getLogger(__name__)


class PDIKafkaConsumer:
    """Enterprise-grade Kafka consumer for the PDI engine."""

    def __init__(
        self,
        topics: List[str],
        group_id: str,
        bootstrap_servers: str = None,
        auto_offset_reset: str = "earliest",
    ):
        self.topics = topics
        self.group_id = group_id
        self.bootstrap_servers = bootstrap_servers or KafkaConfig.BOOTSTRAP_SERVERS
        self.auto_offset_reset = auto_offset_reset
        self._consumer = None

    def _get_consumer(self) -> KafkaConsumer:
        if self._consumer is None:
            self._consumer = KafkaConsumer(
                *self.topics,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                auto_offset_reset=self.auto_offset_reset,
                enable_auto_commit=True,
                auto_commit_interval_ms=5000,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                key_deserializer=lambda k: k.decode("utf-8") if k else None,
                max_poll_records=500,
                session_timeout_ms=30000,
            )
        return self._consumer

    def consume(self, handler: Callable[[Dict], None], max_messages: Optional[int] = None):
        """Consume messages and pass each to the handler function."""
        consumer = self._get_consumer()
        count = 0

        logger.info(f"Starting consumer for topics: {self.topics}, group: {self.group_id}")

        try:
            for message in consumer:
                try:
                    handler(message.value)
                    count += 1

                    if count % 1000 == 0:
                        logger.info(f"Processed {count} messages from {self.topics}")

                    if max_messages and count >= max_messages:
                        logger.info(f"Reached max messages ({max_messages}), stopping")
                        break
                except Exception as e:
                    logger.error(f"Error processing message: {e}", exc_info=True)
        except KeyboardInterrupt:
            logger.info("Consumer interrupted by user")
        finally:
            self.close()

    def consume_batch(self, handler: Callable[[List[Dict]], None], batch_size: int = 100):
        """Consume messages in batches."""
        consumer = self._get_consumer()
        batch = []

        try:
            for message in consumer:
                batch.append(message.value)
                if len(batch) >= batch_size:
                    handler(batch)
                    batch = []

            if batch:  # Process remaining
                handler(batch)
        except KeyboardInterrupt:
            if batch:
                handler(batch)
        finally:
            self.close()

    def close(self):
        if self._consumer:
            self._consumer.close()
            self._consumer = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
