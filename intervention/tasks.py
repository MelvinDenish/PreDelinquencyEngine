# pyre-ignore-all-errors
"""
Celery Tasks
Async tasks for intervention dispatch, scoring batches, and feedback processing.
"""
import os
import sys
import json
import logging
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from intervention.celery_app import celery_app
from intervention.rules_engine import determine_intervention, save_intervention
from intervention.channel_optimizer import select_optimal_channel
from intervention.cooldown_manager import CooldownManager
from intervention.outreach import send_outreach

logger = logging.getLogger(__name__)
cooldown_mgr = CooldownManager()


@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def dispatch_intervention(self, customer_id: str, risk_score: float,
                          risk_tier: str, shap_drivers: list):
    """
    Dispatch an intervention for a customer.
    Full pipeline: cooldown check -> intervention determination ->
    channel selection -> outreach dispatch.
    """
    try:
        # Step 1: Check cooldown and escalation
        should_act, escalation_level, reason = cooldown_mgr.should_intervene(
            customer_id, risk_tier
        )

        if not should_act:
            logger.info(f"[Intervention] Skipping {customer_id}: {reason}")
            return {"status": "skipped", "reason": reason}

        # Step 2: Determine intervention type
        intervention = determine_intervention(
            customer_id, risk_score, risk_tier, shap_drivers
        )

        if not intervention:
            return {"status": "no_intervention_needed"}

        # Step 3: Select optimal channel
        channel = select_optimal_channel(customer_id, risk_tier, escalation_level)
        intervention["channel"] = channel
        intervention["escalation_level"] = escalation_level

        # Step 4: Save intervention
        intervention_id = save_intervention(intervention)
        intervention["intervention_id"] = intervention_id

        # Step 5: Dispatch outreach
        outreach_result = send_outreach(intervention)

        # Step 6: Set cooldown
        cooldown_days = 7 if risk_tier != "critical" else 3
        cooldown_mgr.set_cooldown(customer_id, cooldown_days)

        logger.info(f"[Intervention] Dispatched {intervention['intervention_type']} "
                    f"via {channel} to {customer_id} (ID: {intervention_id})")

        return {
            "status": "dispatched",
            "intervention_id": intervention_id,
            "type": intervention["intervention_type"],
            "channel": channel,
            "escalation_level": escalation_level,
        }

    except Exception as exc:
        logger.error(f"[Intervention] Error for {customer_id}: {exc}")
        self.retry(exc=exc)


@celery_app.task
def process_scoring_batch(customer_ids: list):
    """Process a batch of customers for scoring and intervention."""
    import httpx

    results = []
    for customer_id in customer_ids:
        try:
            # Score via scoring service
            response = httpx.post(
                "http://localhost:8000/score",
                json={"customer_id": customer_id},
                timeout=10,
            )

            if response.status_code == 200:
                score_data = response.json()

                # Check if intervention needed
                if score_data["risk_tier"] in ("critical", "watch"):
                    dispatch_intervention.delay(
                        customer_id,
                        score_data["risk_score"],
                        score_data["risk_tier"],
                        score_data.get("top_shap_features", []),
                    )
                    results.append({"customer_id": customer_id, "status": "intervention_queued"})
                else:
                    results.append({"customer_id": customer_id, "status": "stable"})
            else:
                results.append({"customer_id": customer_id, "status": "scoring_failed"})

        except Exception as e:
            results.append({"customer_id": customer_id, "error": str(e)})

    return results

