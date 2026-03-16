"""
Cooldown Manager
Implements cooldown periods and escalation logic to prevent over-contacting
while ensuring critical risks are addressed.
"""
import os
import sys
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import psycopg2
import redis as redis_lib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config.settings import PostgresConfig, RedisConfig, ModelConfig

logger = logging.getLogger(__name__)


class CooldownManager:
    """Manages intervention cooldown periods and escalation."""

    def __init__(self, cooldown_days: int = None):
        self.cooldown_days = cooldown_days or ModelConfig.COOLDOWN_DAYS
        self.redis = redis_lib.Redis(
            host=RedisConfig.HOST, port=RedisConfig.PORT, db=RedisConfig.DB,
            decode_responses=True,
        )

    def is_in_cooldown(self, customer_id: str) -> Tuple[bool, Optional[datetime]]:
        """
        Check if customer is in cooldown period.
        Returns (is_cooldown, cooldown_until).
        """
        cooldown_key = f"cooldown:{customer_id}"
        cooldown_until_str = self.redis.get(cooldown_key)

        if cooldown_until_str:
            cooldown_until = datetime.fromisoformat(cooldown_until_str)
            if datetime.now() < cooldown_until:
                return True, cooldown_until
            else:
                # Cooldown expired
                self.redis.delete(cooldown_key)
                return False, None

        return False, None

    def set_cooldown(self, customer_id: str, days: int = None):
        """Set cooldown period for a customer after intervention."""
        days = days or self.cooldown_days
        cooldown_until = datetime.now() + timedelta(days=days)
        cooldown_key = f"cooldown:{customer_id}"
        self.redis.set(cooldown_key, cooldown_until.isoformat(), ex=days * 86400)
        logger.info(f"[Cooldown] Set {days}-day cooldown for {customer_id}")

    def get_escalation_level(self, customer_id: str) -> int:
        """
        Get escalation level based on:
        - Number of interventions without response
        - Time since first intervention
        Returns 0 (initial), 1 (nudge -> sms), 2 (sms -> rm_call)
        """
        conn = psycopg2.connect(
            host=PostgresConfig.HOST, port=PostgresConfig.PORT,
            user=PostgresConfig.USER, password=PostgresConfig.PASSWORD,
            dbname=PostgresConfig.DB,
        )
        cursor = conn.cursor()

        # Recent interventions without response
        cursor.execute(
            """SELECT COUNT(*) FROM interventions
               WHERE customer_id = %s
               AND outcome IS NULL OR outcome = 'no_response'
               AND sent_at > NOW() - INTERVAL '30 days'""",
            (customer_id,)
        )
        no_response_count = cursor.fetchone()[0]

        # Check if risk is worsening
        cursor.execute(
            """SELECT risk_score FROM risk_scores
               WHERE customer_id = %s
               ORDER BY scored_at DESC LIMIT 2""",
            (customer_id,)
        )
        scores = cursor.fetchall()
        cursor.close()
        conn.close()

        risk_worsening = False
        if len(scores) >= 2:
            risk_worsening = float(scores[0][0]) > float(scores[1][0])

        # Determine escalation
        if no_response_count >= 3 or (no_response_count >= 1 and risk_worsening):
            return 2  # Direct RM call
        elif no_response_count >= 1:
            return 1  # Escalate channel
        return 0  # Normal

    def should_intervene(self, customer_id: str, risk_tier: str) -> Tuple[bool, int, str]:
        """
        Determine whether to intervene.
        Returns (should_intervene, escalation_level, reason).
        """
        # Critical customers bypass cooldown
        if risk_tier == "critical":
            is_cooldown, _ = self.is_in_cooldown(customer_id)
            if is_cooldown:
                # Still intervene but escalate
                level = self.get_escalation_level(customer_id)
                return True, max(level, 1), "Critical risk - bypassing cooldown"
            level = self.get_escalation_level(customer_id)
            return True, level, "Critical risk tier"

        # Check cooldown for non-critical
        is_cooldown, cooldown_until = self.is_in_cooldown(customer_id)
        if is_cooldown:
            return False, 0, f"In cooldown until {cooldown_until}"

        level = self.get_escalation_level(customer_id)

        if risk_tier == "watch":
            return True, level, "Watch tier - proactive intervention"

        return False, 0, "Stable tier - no intervention needed"

