# pyre-ignore-all-errors
"""
Debezium CDC Connector Registration
Registers a PostgreSQL CDC (Change Data Capture) connector with Debezium Connect.
Captures INSERT/UPDATE/DELETE events from PostgreSQL tables and publishes to Kafka topics.
"""
import os
import sys
import json
import time
import logging
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config.settings import PostgresConfig, KafkaConfig

logger = logging.getLogger(__name__)

DEBEZIUM_CONNECT_URL = os.getenv("DEBEZIUM_CONNECT_URL", "http://localhost:8083")


def get_connector_config() -> dict:
    """Build Debezium PostgreSQL connector configuration."""
    return {
        "name": "pdi-postgres-cdc",
        "config": {
            # Connector class
            "connector.class": "io.debezium.connector.postgresql.PostgresConnector",

            # Database connection
            "database.hostname": PostgresConfig.HOST,
            "database.port": str(PostgresConfig.PORT),
            "database.user": PostgresConfig.USER,
            "database.password": PostgresConfig.PASSWORD,
            "database.dbname": PostgresConfig.DB,
            "database.server.name": "pdi",

            # Topic prefix
            "topic.prefix": "pdi.cdc",

            # Tables to monitor
            "table.include.list": ",".join([
                "public.customers",
                "public.transactions",
                "public.account_balances",
                "public.risk_scores",
                "public.interventions",
                "public.feedback_events",
            ]),

            # PostgreSQL logical decoding
            "plugin.name": "pgoutput",
            "slot.name": "pdi_debezium_slot",
            "publication.name": "pdi_publication",

            # Schema handling
            "schema.history.internal.kafka.bootstrap.servers": KafkaConfig.BOOTSTRAP_SERVERS,
            "schema.history.internal.kafka.topic": "pdi.schema-changes",

            # Transforms: flatten and route
            "transforms": "unwrap,route",
            "transforms.unwrap.type": "io.debezium.transforms.ExtractNewRecordState",
            "transforms.unwrap.drop.tombstones": "true",
            "transforms.unwrap.delete.handling.mode": "rewrite",
            "transforms.unwrap.add.fields": "op,table,source.ts_ms",
            "transforms.route.type": "org.apache.kafka.connect.transforms.RegexRouter",
            "transforms.route.regex": "pdi\\.cdc\\.public\\.(.*)",
            "transforms.route.replacement": "pdi.cdc.$1",

            # Serialization
            "key.converter": "org.apache.kafka.connect.json.JsonConverter",
            "key.converter.schemas.enable": "false",
            "value.converter": "org.apache.kafka.connect.json.JsonConverter",
            "value.converter.schemas.enable": "false",

            # Snapshot mode
            "snapshot.mode": "initial",

            # Heartbeat
            "heartbeat.interval.ms": "10000",

            # Error handling
            "errors.tolerance": "all",
            "errors.log.enable": "true",
            "errors.log.include.messages": "true",
        }
    }


def register_connector(max_retries: int = 10, retry_delay: int = 5) -> bool:
    """Register the CDC connector with Debezium Connect."""
    connector_config = get_connector_config()

    for attempt in range(max_retries):
        try:
            # Check if Connect is ready
            resp = requests.get(f"{DEBEZIUM_CONNECT_URL}/connectors", timeout=5)
            if resp.status_code != 200:
                logger.info(f"[Debezium] Connect not ready (attempt {attempt+1}/{max_retries})")
                time.sleep(retry_delay)
                continue

            existing = resp.json()

            # Check if connector already exists
            if connector_config["name"] in existing:
                logger.info(f"[Debezium] Connector '{connector_config['name']}' already exists. Updating...")
                resp = requests.put(
                    f"{DEBEZIUM_CONNECT_URL}/connectors/{connector_config['name']}/config",
                    json=connector_config["config"],
                    headers={"Content-Type": "application/json"},
                    timeout=10,
                )
            else:
                logger.info(f"[Debezium] Registering connector '{connector_config['name']}'...")
                resp = requests.post(
                    f"{DEBEZIUM_CONNECT_URL}/connectors",
                    json=connector_config,
                    headers={"Content-Type": "application/json"},
                    timeout=10,
                )

            if resp.status_code in (200, 201):
                logger.info(f"[Debezium] Connector registered successfully!")
                return True
            else:
                logger.warning(f"[Debezium] Registration failed: {resp.status_code} - {resp.text}")

        except requests.ConnectionError:
            logger.info(f"[Debezium] Connect not reachable (attempt {attempt+1}/{max_retries})")
            time.sleep(retry_delay)
        except Exception as e:
            logger.error(f"[Debezium] Error: {e}")
            time.sleep(retry_delay)

    logger.error("[Debezium] Failed to register connector after all retries")
    return False


def get_connector_status() -> dict:
    """Get the current status of the CDC connector."""
    try:
        resp = requests.get(
            f"{DEBEZIUM_CONNECT_URL}/connectors/pdi-postgres-cdc/status",
            timeout=5,
        )
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


def delete_connector() -> bool:
    """Delete the CDC connector."""
    try:
        resp = requests.delete(
            f"{DEBEZIUM_CONNECT_URL}/connectors/pdi-postgres-cdc",
            timeout=5,
        )
        return resp.status_code == 204
    except Exception as e:
        logger.error(f"[Debezium] Delete failed: {e}")
        return False


def list_topics() -> list:
    """List Kafka topics created by Debezium."""
    try:
        resp = requests.get(
            f"{DEBEZIUM_CONNECT_URL}/connectors/pdi-postgres-cdc/topics",
            timeout=5,
        )
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Registering Debezium CDC connector...")
    success = register_connector()
    if success:
        print("Connector registered. Checking status...")
        status = get_connector_status()
        print(json.dumps(status, indent=2))
    else:
        print("Failed to register connector.")
