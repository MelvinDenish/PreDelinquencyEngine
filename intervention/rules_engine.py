# pyre-ignore-all-errors
"""
Rules Engine — Segment-Aware
Determines intervention type based on risk tier transitions,
SHAP-driven signal-aware routing, customer segment thresholds,
cold-start caps, and product action proposals.
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

    # Asset-side features (M2)
    "fd_premature_closures_90d": "emi_restructuring",
    "sip_stoppages_90d": "savings_review",
    "insurance_lapse_flag": "wellness_checkin",

    # Employer health (M3)
    "employer_payroll_delay_avg": "payment_holiday",
    "employer_headcount_change_pct": "wellness_checkin",

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
    "savings_review": "Savings & investment portfolio health review",
}

# Segment-specific risk tier thresholds
SEGMENT_THRESHOLDS = {
    "salaried":      {"watch": 0.50, "critical": 0.70},
    "self_employed":  {"watch": 0.45, "critical": 0.65},
    "gig_worker":     {"watch": 0.40, "critical": 0.60},
    "retiree":        {"watch": 0.45, "critical": 0.70},
    "agricultural":   {"watch": 0.40, "critical": 0.60},
    "nri":            {"watch": 0.50, "critical": 0.70},
    "student":        {"watch": 0.55, "critical": 0.75},
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
    segment_type: str = "salaried",
    is_cold_start: bool = False,
    customer_features: dict = None,
) -> Optional[Dict]:
    """
    Determine the appropriate intervention based on risk tier, SHAP drivers,
    customer segment, and cold-start status.
    Returns intervention dict or None if no intervention needed.
    """
    # Cold-start cap: never route to critical interventions
    if is_cold_start and risk_tier == "critical":
        risk_tier = "watch"
        logger.info(f"[RulesEngine] Cold-start cap applied for {customer_id}")

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

    # Segment-specific escalation thresholds
    seg_thresholds = SEGMENT_THRESHOLDS.get(segment_type, SEGMENT_THRESHOLDS["salaried"])
    critical_threshold = seg_thresholds.get("critical", 0.70)

    # Escalate for critical tier
    if risk_tier == "critical" and risk_score > critical_threshold + 0.15:
        intervention_type = "escalation_call"
        trigger_reason = f"Critical risk score ({risk_score:.2f}) for {segment_type} — escalation required"

    # Generate product action proposals
    product_actions = []
    if customer_features and risk_tier in ("watch", "critical"):
        try:
            from intervention.product_actions import ProductActionEngine
            product_engine = ProductActionEngine()
            product_actions = product_engine.generate_proposals(
                customer_id, customer_features, risk_score, risk_tier
            )
        except Exception as e:
            logger.warning(f"[RulesEngine] Product action generation failed: {e}")

    # P9: Uplift gate — suppress interventions for "sure payers" (negative uplift)
    uplift_gate_passed = True
    try:
        import os
        import numpy as np
        from ml.uplift_model import UpliftModel
        uplift_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'uplift_model.joblib')
        if os.path.exists(uplift_path) and customer_features:
            uplift = UpliftModel()
            uplift.load(uplift_path)
            feature_vec = np.array([customer_features.get(c, 0.0) for c in ModelConfig.FEATURE_COLUMNS])
            u_score = uplift.predict_uplift_single(feature_vec)
            if u_score is not None and u_score <= 0:
                logger.info(f"[RulesEngine] Uplift gate: {customer_id} suppressed (uplift={u_score:.4f})")
                uplift_gate_passed = False
    except Exception as e:
        logger.debug(f"[RulesEngine] Uplift gate check skipped: {e}")

    if not uplift_gate_passed:
        return None  # Don't intervene on sure-payers


    return {
        "customer_id": customer_id,
        "intervention_type": intervention_type,
        "trigger_reason": trigger_reason,
        "risk_score": risk_score,
        "risk_tier": risk_tier,
        "segment_type": segment_type,
        "is_cold_start": is_cold_start,
        "shap_drivers": shap_drivers,
        "previous_tier": previous_tier,
        "description": INTERVENTION_DESCRIPTIONS.get(intervention_type, ""),
        "product_actions": [a["action_type"] for a in product_actions],
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

