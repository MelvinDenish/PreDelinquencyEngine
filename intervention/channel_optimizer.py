"""
Channel Optimizer
Selects the best outreach channel for each customer based on
their historical engagement data and preferences.
"""
import os
import sys
import logging
from typing import Dict

import psycopg2
import redis as redis_lib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config.settings import PostgresConfig, RedisConfig

logger = logging.getLogger(__name__)

# Channel priority for escalation
CHANNEL_ESCALATION_ORDER = ["app", "sms", "email", "rm_call"]


def get_customer_channel_preference(customer_id: str) -> str:
    """Get customer's preferred communication channel from profile."""
    conn = psycopg2.connect(
        host=PostgresConfig.HOST, port=PostgresConfig.PORT,
        user=PostgresConfig.USER, password=PostgresConfig.PASSWORD,
        dbname=PostgresConfig.DB,
    )
    cursor = conn.cursor()
    cursor.execute(
        "SELECT preferred_channel FROM customers WHERE customer_id = %s",
        (customer_id,)
    )
    row = cursor.fetchone()
    cursor.close()
    conn.close()

    return row[0] if row else "sms"


def get_channel_response_history(customer_id: str) -> Dict[str, float]:
    """
    Get historical response rates by channel for this customer.
    Returns dict mapping channel to response rate (0.0 - 1.0).
    """
    conn = psycopg2.connect(
        host=PostgresConfig.HOST, port=PostgresConfig.PORT,
        user=PostgresConfig.USER, password=PostgresConfig.PASSWORD,
        dbname=PostgresConfig.DB,
    )
    cursor = conn.cursor()
    cursor.execute(
        """SELECT channel,
                  COUNT(*) as total,
                  SUM(CASE WHEN outcome IS NOT NULL AND outcome != 'no_response' THEN 1 ELSE 0 END) as responded
           FROM interventions
           WHERE customer_id = %s
           GROUP BY channel""",
        (customer_id,)
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    response_rates = {}
    for channel, total, responded in rows:
        response_rates[channel] = responded / total if total > 0 else 0.5

    return response_rates


def select_optimal_channel(customer_id: str, risk_tier: str,
                           escalation_level: int = 0) -> str:
    """
    Select optimal outreach channel based on:
    1. Customer preference
    2. Historical response rates
    3. Risk tier (higher risk -> more direct channel)
    4. Escalation level
    """
    # Get preference and history
    preferred = get_customer_channel_preference(customer_id)
    response_history = get_channel_response_history(customer_id)

    # For critical tier or high escalation, use more direct channels
    if risk_tier == "critical" or escalation_level >= 2:
        return "rm_call"

    # If we have response history, use highest-engagement channel
    if response_history:
        best_channel = max(response_history, key=response_history.get)
        if response_history[best_channel] > 0.3:
            return best_channel

    # For escalation, move up the channel order
    if escalation_level > 0:
        pref_idx = CHANNEL_ESCALATION_ORDER.index(preferred) if preferred in CHANNEL_ESCALATION_ORDER else 0
        next_idx = min(pref_idx + escalation_level, len(CHANNEL_ESCALATION_ORDER) - 1)
        return CHANNEL_ESCALATION_ORDER[next_idx]

    return preferred

