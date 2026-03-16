"""
Outreach Module
Handles multi-channel intervention delivery using Apprise.
Supports SMS, email, app notifications, and RM call scheduling.
"""
import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict

import psycopg2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config.settings import PostgresConfig

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Message Templates
# ─────────────────────────────────────────────
MESSAGE_TEMPLATES = {
    "payment_holiday": {
        "title": "Payment Relief Available",
        "body": (
            "Dear {customer_name}, we understand you may be experiencing "
            "financial changes. We'd like to offer you a payment holiday option. "
            "Please reply YES to learn more, or call us at 1800-XXX-XXXX."
        ),
    },
    "emi_restructuring": {
        "title": "EMI Adjustment Offer",
        "body": (
            "Dear {customer_name}, we've noticed your payment patterns have shifted. "
            "We can restructure your EMI to a more comfortable amount. "
            "Reply RESTRUCTURE or visit our app to explore options."
        ),
    },
    "wellness_checkin": {
        "title": "Financial Wellness Check-in",
        "body": (
            "Hi {customer_name}, your financial wellness matters to us. "
            "Would you like to speak with a financial advisor? "
            "We're here to help. Reply TALK to schedule a call."
        ),
    },
    "budget_nudge": {
        "title": "Smart Spending Insights",
        "body": (
            "Hi {customer_name}, we've noticed increased spending recently. "
            "Here's a tip: Setting monthly spending alerts can help track your budget. "
            "Open the app to see your spending summary."
        ),
    },
    "payment_reminder": {
        "title": "Payment Reminder",
        "body": (
            "Hi {customer_name}, this is a friendly reminder about your upcoming "
            "payment. Set up auto-pay in our app to never miss a due date."
        ),
    },
    "escalation_call": {
        "title": "Priority Outreach",
        "body": (
            "Dear {customer_name}, our relationship manager would like to "
            "speak with you regarding your account. Please expect a call "
            "from us within 24 hours, or call us at 1800-XXX-XXXX."
        ),
    },
}


def _get_customer_name(customer_id: str) -> str:
    """Get customer name from database."""
    try:
        conn = psycopg2.connect(
            host=PostgresConfig.HOST, port=PostgresConfig.PORT,
            user=PostgresConfig.USER, password=PostgresConfig.PASSWORD,
            dbname=PostgresConfig.DB,
        )
        cursor = conn.cursor()
        cursor.execute(
            "SELECT first_name, last_name FROM customers WHERE customer_id = %s",
            (customer_id,)
        )
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        return f"{row[0]} {row[1]}" if row else "Valued Customer"
    except Exception:
        return "Valued Customer"


def send_outreach(intervention: Dict) -> Dict:
    """
    Send outreach through the selected channel.
    Uses Apprise for multi-channel delivery.
    """
    customer_id = intervention["customer_id"]
    channel = intervention.get("channel", "app")
    intervention_type = intervention["intervention_type"]

    # Get customer name
    customer_name = _get_customer_name(customer_id)

    # Get message template
    template = MESSAGE_TEMPLATES.get(intervention_type, MESSAGE_TEMPLATES["wellness_checkin"])
    title = template["title"]
    body = template["body"].format(customer_name=customer_name)

    result = {
        "channel": channel,
        "title": title,
        "body": body,
        "status": "sent",
        "sent_at": datetime.now().isoformat(),
    }

    try:
        if channel == "app":
            # In-app notification (stored in Redis for app retrieval)
            import redis as redis_lib
            from config.settings import RedisConfig
            r = redis_lib.Redis(host=RedisConfig.HOST, port=RedisConfig.PORT, db=RedisConfig.DB)
            notification = {
                "title": title,
                "body": body,
                "type": intervention_type,
                "timestamp": datetime.now().isoformat(),
                "intervention_id": intervention.get("intervention_id"),
            }
            r.lpush(f"notifications:{customer_id}", json.dumps(notification))
            r.ltrim(f"notifications:{customer_id}", 0, 49)  # Keep last 50
            result["delivery"] = "in_app_notification"

        elif channel == "sms":
            # SMS via Apprise (when configured with SMS service)
            try:
                import apprise
                apobj = apprise.Apprise()
                # In production, add SMS service URL: apobj.add('sns://...')
                # For now, log the message
                logger.info(f"[SMS] To {customer_id}: {title} - {body}")
                result["delivery"] = "sms_queued"
            except ImportError:
                logger.info(f"[SMS] Apprise not available. Message: {body}")
                result["delivery"] = "sms_logged"

        elif channel == "email":
            # Email via Apprise
            try:
                import apprise
                apobj = apprise.Apprise()
                # In production: apobj.add('mailto://user:pass@gmail.com')
                logger.info(f"[Email] To {customer_id}: {title} - {body}")
                result["delivery"] = "email_queued"
            except ImportError:
                logger.info(f"[Email] Apprise not available. Message: {body}")
                result["delivery"] = "email_logged"

        elif channel == "rm_call":
            # Schedule RM callback
            logger.info(f"[RM Call] Scheduled call for {customer_id}: {title}")
            result["delivery"] = "rm_call_scheduled"

        # Update intervention status in DB
        if intervention.get("intervention_id"):
            _update_intervention_status(
                intervention["intervention_id"],
                "sent",
                channel,
            )

    except Exception as e:
        logger.error(f"[Outreach] Failed for {customer_id}: {e}")
        result["status"] = "failed"
        result["error"] = str(e)

    return result


def _update_intervention_status(intervention_id: int, status: str, channel: str):
    """Update intervention status in PostgreSQL."""
    conn = psycopg2.connect(
        host=PostgresConfig.HOST, port=PostgresConfig.PORT,
        user=PostgresConfig.USER, password=PostgresConfig.PASSWORD,
        dbname=PostgresConfig.DB,
    )
    cursor = conn.cursor()
    cursor.execute(
        """UPDATE interventions
           SET status = %s, channel = %s, sent_at = NOW()
           WHERE id = %s""",
        (status, channel, intervention_id)
    )
    conn.commit()
    cursor.close()
    conn.close()
