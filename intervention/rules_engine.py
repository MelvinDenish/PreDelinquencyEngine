"""
Rules Engine
Determines intervention type based on risk tier transitions
and SHAP-driven signal-aware routing.
"""
import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, Optional, Tuple

import psycopg2
import redis as redis_lib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config.settings import PostgresConfig, RedisConfig, ModelConfig

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Intervention type mapping based on SHAP drivers
# ─────────────────────────────────────────────
SHAP_INTERVENTION_MAP = {
    # Salary-related signals -> Payment holiday
    "salary_delay_days": "payment_holiday",

    # Savings drawdown -> EMI restructuring
    "savings_balance_pct_change_7d": "emi_restructuring",

    # Lending app usage spike -> Wellness check-in
    "lending_app_txn_count_7d": "wellness_checkin",
    "weighted_lending_risk_7d": "wellness_checkin",
    "lending_app_txn_count_30d": "wellness_checkin",

    # Failed auto-debits -> EMI restructuring
    "failed_autodebits_count_7d": "emi_restructuring",
    "failed_autodebits_count_30d": "emi_restructuring",

    # High discretionary spending -> Budget nudge
    "discretionary_spend_7d": "budget_nudge",
    "discretionary_spend_30d": "budget_nudge",
    "discretionary_spend_trend": "budget_nudge",

    # ATM patterns -> Wellness check-in
    "atm_withdrawals_count_7d": "wellness_checkin",

    # Gambling / lottery spend -> Wellness check-in
    "gambling_lottery_spend_7d": "wellness_checkin",
    "gambling_lottery_spend_30d": "wellness_checkin",

    # Default
    "utility_payment_delay_avg": "payment_reminder",
}

INTERVENTION_DESCRIPTIONS = {
    "payment_holiday": "Offer payment holiday to ease financial stress during salary disruption",
    "emi_restructuring": "Restructure EMI schedule to lower monthly burden",
    "wellness_checkin": "Financial wellness check-in to provide counseling and support",
    "budget_nudge": "Gentle spending awareness nudge with budgeting tips",
    "payment_reminder": "Friendly payment reminder before due date",
    "escalation_call": "Relationship manager direct outreach call",
}


def check_tier_transition(customer_id: str, new_tier: str) -> Optional[str]:
    """
    Check if there's a risk tier transition for this customer.
    Returns the previous tier if transition occurred, None otherwise.
    Alerts are only triggered on tier transitions (per project spec).
    """
    r = redis_lib.Redis(host=RedisConfig.HOST, port=RedisConfig.PORT, db=RedisConfig.DB,
                        decode_responses=True)

    key = f"risk_tier:{customer_id}"
    previous_tier = r.get(key)

    # Update stored tier
    r.set(key, new_tier)

    if previous_tier and previous_tier != new_tier:
        # Tier change detected
        tier_order = {"stable": 0, "watch": 1, "critical": 2}
        if tier_order.get(new_tier, 0) > tier_order.get(previous_tier, 0):
            # Worsening - trigger alert
            return previous_tier
    elif previous_tier is None:
        # First score - only alert if not stable
        if new_tier in ("watch", "critical"):
            return "new"

    return None


def determine_intervention(
    customer_id: str,
    risk_score: float,
    risk_tier: str,
    shap_drivers: list,
) -> Optional[Dict]:
    """
    Determine the appropriate intervention based on risk tier and SHAP drivers.
    Returns intervention dict or None if no intervention needed.
    """
    # Check for tier transition
    previous_tier = check_tier_transition(customer_id, risk_tier)
    if previous_tier is None and risk_tier == "stable":
        return None  # No intervention for stable customers without transition

    # Determine intervention type from top SHAP driver
    intervention_type = "wellness_checkin"  # Default
    trigger_reason = "Risk tier transition detected"

    if shap_drivers:
        top_driver = shap_drivers[0]
        feature_name = top_driver.get("feature", "")
        if feature_name in SHAP_INTERVENTION_MAP:
            intervention_type = SHAP_INTERVENTION_MAP[feature_name]
            trigger_reason = f"Top risk driver: {feature_name} (SHAP: {top_driver.get('shap_value', 0):.3f})"

    # Escalate for critical tier
    if risk_tier == "critical" and risk_score > 0.85:
        intervention_type = "escalation_call"
        trigger_reason = f"Critical risk score ({risk_score:.2f}) - escalation required"

    return {
        "customer_id": customer_id,
        "intervention_type": intervention_type,
        "trigger_reason": trigger_reason,
        "risk_score": risk_score,
        "risk_tier": risk_tier,
        "shap_drivers": shap_drivers,
        "previous_tier": previous_tier,
        "description": INTERVENTION_DESCRIPTIONS.get(intervention_type, ""),
    }


def save_intervention(intervention: Dict) -> int:
    """Save intervention to PostgreSQL. Returns intervention ID."""
    conn = psycopg2.connect(
        host=PostgresConfig.HOST, port=PostgresConfig.PORT,
        user=PostgresConfig.USER, password=PostgresConfig.PASSWORD,
        dbname=PostgresConfig.DB,
    )
    cursor = conn.cursor()

    cursor.execute(
        """INSERT INTO interventions
        (customer_id, intervention_type, channel, trigger_reason,
         shap_drivers, risk_score_at_trigger, risk_tier_at_trigger, status)
        VALUES (%s, %s, %s, %s, %s, %s, %s, 'pending')
        RETURNING id""",
        (
            intervention["customer_id"],
            intervention["intervention_type"],
            intervention.get("channel", "app"),
            intervention["trigger_reason"],
            json.dumps(intervention.get("shap_drivers", [])),
            intervention["risk_score"],
            intervention["risk_tier"],
        )
    )

    intervention_id = cursor.fetchone()[0]
    conn.commit()
    cursor.close()
    conn.close()

    return intervention_id

