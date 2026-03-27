# pyre-ignore-all-errors
"""
Production Notification Dispatcher
Actually sends notifications via multiple channels based on risk tier routing.

Channels:
  - Email:   SMTP (Gmail/Outlook/SendGrid)
  - SMS:     Twilio API
  - WhatsApp: Twilio WhatsApp API
  - Push:    Webhook to any push service
  - RM Call: Creates task in DB + optional webhook to CRM
  - Collector: Creates assignment in DB + optional webhook to collections system
"""
import os
import sys
import json
import logging
import smtplib
import uuid
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from typing import Dict, Optional, List

import psycopg2
import redis as redis_lib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config.settings import PostgresConfig, RedisConfig

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Configuration from environment
# ─────────────────────────────────────────────
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")  # App password for Gmail
SMTP_FROM_NAME = os.getenv("SMTP_FROM_NAME", "Barclays India - Financial Wellness")
SMTP_FROM_EMAIL = os.getenv("SMTP_FROM_EMAIL", SMTP_USER)

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER", "")
TWILIO_WHATSAPP_FROM = os.getenv("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")

WEBHOOK_PUSH_URL = os.getenv("WEBHOOK_PUSH_URL", "")
WEBHOOK_CRM_URL = os.getenv("WEBHOOK_CRM_URL", "")
WEBHOOK_COLLECTIONS_URL = os.getenv("WEBHOOK_COLLECTIONS_URL", "")

# ─────────────────────────────────────────────
# Risk-tier → Channel routing matrix
# ─────────────────────────────────────────────
CHANNEL_ROUTING = {
    # (min_score, max_score): [channels_in_priority_order]
    "low_watch":    {"range": (0.30, 0.50), "channels": ["app_push", "sms"],
                     "action": "budget_nudge", "urgency": "low"},
    "medium_watch": {"range": (0.50, 0.65), "channels": ["sms", "whatsapp", "app_push"],
                     "action": "wellness_tips", "urgency": "medium"},
    "high_watch":   {"range": (0.65, 0.75), "channels": ["email", "sms", "whatsapp"],
                     "action": "emi_restructuring_offer", "urgency": "high"},
    "critical":     {"range": (0.75, 0.85), "channels": ["rm_call", "email", "sms"],
                     "action": "proactive_outreach", "urgency": "critical"},
    "severe":       {"range": (0.85, 1.00), "channels": ["collector_assignment", "rm_call", "email"],
                     "action": "collections_escalation", "urgency": "severe"},
}


def _get_risk_routing(risk_score: float) -> Dict:
    """Determine channel routing based on risk score."""
    for tier_name, config in CHANNEL_ROUTING.items():
        low, high = config["range"]
        if low <= risk_score < high:
            return {"tier": tier_name, **config}
    if risk_score >= 0.85:
        return {"tier": "severe", **CHANNEL_ROUTING["severe"]}
    return {"tier": "stable", "channels": [], "action": "none", "urgency": "none"}


def _check_cooldown(customer_id: str, channel: str, cooldown_hours: int = 24) -> bool:
    """Check if customer was recently contacted on this channel. Returns True if OK to send."""
    try:
        r = redis_lib.Redis(host=RedisConfig.HOST, port=RedisConfig.PORT,
                            db=RedisConfig.DB, decode_responses=True)
        key = f"notif_cooldown:{customer_id}:{channel}"
        if r.exists(key):
            return False  # Still in cooldown
        r.setex(key, cooldown_hours * 3600, "1")
        return True
    except Exception as e:
        logger.warning(f"[Cooldown] Redis error: {e}, proceeding anyway")
        return True


# ─────────────────────────────────────────────
# Channel Dispatchers — Actually Send
# ─────────────────────────────────────────────
def send_email(customer: Dict, subject: str, html_body: str, text_body: str = "") -> Dict:
    """Send email via SMTP. Returns delivery result."""
    to_email = customer.get("email", "")
    if not to_email:
        return {"status": "failed", "error": "No email address", "channel": "email"}

    if not SMTP_USER or not SMTP_PASSWORD:
        logger.warning("[Email] SMTP not configured, logging instead")
        return _log_notification(customer, "email", text_body or html_body, "simulated")

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = f"{SMTP_FROM_NAME} <{SMTP_FROM_EMAIL}>"
        msg["To"] = to_email

        if text_body:
            msg.attach(MIMEText(text_body, "plain"))
        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_FROM_EMAIL, to_email, msg.as_string())

        logger.info(f"[Email] ✅ Sent to {to_email} for {customer.get('customer_id')}")
        return _log_notification(customer, "email", text_body[:500], "delivered")

    except Exception as e:
        logger.error(f"[Email] ❌ Failed for {customer.get('customer_id')}: {e}")
        return _log_notification(customer, "email", text_body[:500], "failed", str(e))


def send_sms(customer: Dict, message: str) -> Dict:
    """Send SMS via Twilio. Returns delivery result."""
    phone = customer.get("phone", "")
    if not phone:
        return {"status": "failed", "error": "No phone number", "channel": "sms"}

    if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
        logger.warning("[SMS] Twilio not configured, logging instead")
        return _log_notification(customer, "sms", message, "simulated")

    try:
        from twilio.rest import Client
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        tw_msg = client.messages.create(
            body=message[:1600],  # SMS limit
            from_=TWILIO_FROM_NUMBER,
            to=phone
        )
        logger.info(f"[SMS] ✅ Sent to {phone}, SID: {tw_msg.sid}")
        return _log_notification(customer, "sms", message, "delivered",
                                 metadata={"twilio_sid": tw_msg.sid})

    except ImportError:
        logger.warning("[SMS] twilio package not installed")
        return _log_notification(customer, "sms", message, "simulated")
    except Exception as e:
        logger.error(f"[SMS] ❌ Failed for {phone}: {e}")
        return _log_notification(customer, "sms", message, "failed", str(e))


def send_whatsapp(customer: Dict, message: str) -> Dict:
    """Send WhatsApp message via Twilio WhatsApp API."""
    phone = customer.get("phone", "")
    if not phone:
        return {"status": "failed", "error": "No phone", "channel": "whatsapp"}

    if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
        logger.warning("[WhatsApp] Twilio not configured, logging instead")
        return _log_notification(customer, "whatsapp", message, "simulated")

    try:
        from twilio.rest import Client
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        tw_msg = client.messages.create(
            body=message,
            from_=TWILIO_WHATSAPP_FROM,
            to=f"whatsapp:{phone}"
        )
        logger.info(f"[WhatsApp] ✅ Sent to {phone}, SID: {tw_msg.sid}")
        return _log_notification(customer, "whatsapp", message, "delivered",
                                 metadata={"twilio_sid": tw_msg.sid})
    except ImportError:
        return _log_notification(customer, "whatsapp", message, "simulated")
    except Exception as e:
        logger.error(f"[WhatsApp] ❌ Failed: {e}")
        return _log_notification(customer, "whatsapp", message, "failed", str(e))


def send_push_notification(customer: Dict, title: str, body: str) -> Dict:
    """Send push notification via webhook."""
    payload = {
        "customer_id": customer.get("customer_id"),
        "title": title,
        "body": body,
        "deep_link": f"barclays://wellness/{customer.get('customer_id')}",
        "timestamp": datetime.utcnow().isoformat(),
    }

    if not WEBHOOK_PUSH_URL:
        logger.warning("[Push] Webhook not configured, logging instead")
        return _log_notification(customer, "app_push", body, "simulated")

    try:
        import requests
        resp = requests.post(WEBHOOK_PUSH_URL, json=payload, timeout=5)
        status = "delivered" if resp.status_code < 300 else "failed"
        logger.info(f"[Push] {'✅' if status == 'delivered' else '❌'} {customer.get('customer_id')}")
        return _log_notification(customer, "app_push", body, status)
    except Exception as e:
        logger.error(f"[Push] ❌ Failed: {e}")
        return _log_notification(customer, "app_push", body, "failed", str(e))


def assign_rm_call(customer: Dict, risk_score: float, intervention: Dict,
                   call_script: str = "") -> Dict:
    """Create RM (Relationship Manager) callback task in database."""
    task_id = f"RM_{uuid.uuid4().hex[:10].upper()}"
    customer_id = customer.get("customer_id")

    try:
        conn = psycopg2.connect(
            host=PostgresConfig.HOST, port=PostgresConfig.PORT,
            user=PostgresConfig.USER, password=PostgresConfig.PASSWORD,
            dbname=PostgresConfig.DB,
        )
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO rm_tasks (task_id, customer_id, risk_score, risk_tier,
                                  priority, call_script, intervention_type,
                                  shap_drivers, status, created_at, due_by)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'pending', NOW(),
                    NOW() + INTERVAL '4 hours')
            ON CONFLICT (task_id) DO NOTHING
        """, (
            task_id, customer_id, risk_score,
            intervention.get("risk_tier", "watch"),
            "P1" if risk_score >= 0.8 else "P2",
            call_script[:2000] if call_script else "Contact customer for wellness check-in",
            intervention.get("intervention_type", "wellness_checkin"),
            json.dumps(intervention.get("shap_drivers", [])[:5]),
        ))
        conn.commit()
        cursor.close()
        conn.close()

        logger.info(f"[RM] ✅ Task {task_id} created for {customer_id}")

        # Optional CRM webhook
        if WEBHOOK_CRM_URL:
            try:
                import requests
                requests.post(WEBHOOK_CRM_URL, json={
                    "task_id": task_id, "customer_id": customer_id,
                    "risk_score": risk_score, "priority": "P1" if risk_score >= 0.8 else "P2",
                    "call_script": call_script[:500],
                }, timeout=5)
            except Exception:
                pass

        return _log_notification(customer, "rm_call", call_script[:500], "assigned",
                                 metadata={"task_id": task_id})

    except Exception as e:
        logger.error(f"[RM] ❌ Task creation failed: {e}")
        return _log_notification(customer, "rm_call", call_script[:500], "failed", str(e))


def assign_collector(customer: Dict, risk_score: float, intervention: Dict,
                     collector_brief: str = "") -> Dict:
    """Create collector assignment for severe risk cases."""
    assignment_id = f"COL_{uuid.uuid4().hex[:10].upper()}"
    customer_id = customer.get("customer_id")

    # Calculate restructuring offer based on DTI and income
    monthly_salary = customer.get("monthly_salary", 50000)
    dti = customer.get("dti_ratio", 0.5)
    restructuring_offer = {
        "max_emi_reduction_pct": min(int((dti - 0.4) * 100), 50) if dti > 0.4 else 10,
        "tenure_extension_months": 12 if risk_score > 0.9 else 6,
        "payment_holiday_months": 3 if risk_score > 0.9 else 1,
        "interest_rate_concession_bps": 50 if risk_score > 0.9 else 25,
        "settlement_offer_pct": round(max(70, 100 - (risk_score - 0.85) * 200), 1),
    }

    try:
        conn = psycopg2.connect(
            host=PostgresConfig.HOST, port=PostgresConfig.PORT,
            user=PostgresConfig.USER, password=PostgresConfig.PASSWORD,
            dbname=PostgresConfig.DB,
        )
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO collector_assignments
            (assignment_id, customer_id, risk_score, risk_tier,
             collector_brief, restructuring_offer, intervention_type,
             status, priority, created_at, due_by)
            VALUES (%s, %s, %s, %s, %s, %s, %s, 'assigned', %s, NOW(),
                    NOW() + INTERVAL '24 hours')
            ON CONFLICT (assignment_id) DO NOTHING
        """, (
            assignment_id, customer_id, risk_score,
            intervention.get("risk_tier", "critical"),
            collector_brief[:3000] if collector_brief else "Immediate contact required",
            json.dumps(restructuring_offer),
            intervention.get("intervention_type", "escalation_call"),
            "P0" if risk_score >= 0.9 else "P1",
        ))
        conn.commit()
        cursor.close()
        conn.close()

        logger.info(f"[Collector] ✅ Assignment {assignment_id} for {customer_id}, "
                     f"offer: {restructuring_offer['max_emi_reduction_pct']}% EMI reduction")

        # Optional collections webhook
        if WEBHOOK_COLLECTIONS_URL:
            try:
                import requests
                requests.post(WEBHOOK_COLLECTIONS_URL, json={
                    "assignment_id": assignment_id,
                    "customer_id": customer_id,
                    "customer_name": f"{customer.get('first_name')} {customer.get('last_name')}",
                    "phone": customer.get("phone"),
                    "risk_score": risk_score,
                    "restructuring_offer": restructuring_offer,
                    "priority": "P0" if risk_score >= 0.9 else "P1",
                }, timeout=5)
            except Exception:
                pass

        return _log_notification(customer, "collector_assignment",
                                 collector_brief[:500], "assigned",
                                 metadata={"assignment_id": assignment_id,
                                           "restructuring_offer": restructuring_offer})

    except Exception as e:
        logger.error(f"[Collector] ❌ Assignment failed: {e}")
        return _log_notification(customer, "collector_assignment",
                                 collector_brief[:500], "failed", str(e))


# ─────────────────────────────────────────────
# Notification Logger — All notifications tracked in DB
# ─────────────────────────────────────────────
def _log_notification(customer: Dict, channel: str, message: str,
                      status: str, error: str = None, metadata: dict = None) -> Dict:
    """Log every notification attempt to the notifications table."""
    notification_id = f"NOTIF_{uuid.uuid4().hex[:12].upper()}"
    customer_id = customer.get("customer_id", "UNKNOWN")

    try:
        conn = psycopg2.connect(
            host=PostgresConfig.HOST, port=PostgresConfig.PORT,
            user=PostgresConfig.USER, password=PostgresConfig.PASSWORD,
            dbname=PostgresConfig.DB,
        )
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO notifications
            (notification_id, customer_id, channel, message_preview,
             status, error_message, metadata, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
        """, (
            notification_id, customer_id, channel,
            message[:500] if message else "",
            status, error,
            json.dumps(metadata) if metadata else None,
        ))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        logger.error(f"[NotifLog] DB error: {e}")

    result = {
        "notification_id": notification_id,
        "customer_id": customer_id,
        "channel": channel,
        "status": status,
        "timestamp": datetime.utcnow().isoformat(),
    }
    if error:
        result["error"] = error
    if metadata:
        result.update(metadata)
    return result


# ─────────────────────────────────────────────
# Main Dispatcher — Routes to correct channels
# ─────────────────────────────────────────────
def dispatch_notification(
    customer: Dict,
    intervention: Dict,
    messages: Dict,
) -> List[Dict]:
    """
    Route and send notifications through appropriate channels based on risk score.

    Args:
        customer: Full customer profile with phone, email, etc.
        intervention: Intervention details from rules engine
        messages: Dict with keys 'sms', 'email_html', 'email_text', 'push_title',
                  'push_body', 'rm_call_script', 'collector_brief'

    Returns:
        List of delivery results for each channel attempted
    """
    risk_score = intervention.get("risk_score", 0)
    routing = _get_risk_routing(risk_score)

    if routing["tier"] == "stable":
        return []  # No notification for stable customers

    results = []
    channels_to_use = routing["channels"]
    customer_id = customer.get("customer_id", "UNKNOWN")

    # P14: Use LinUCB bandit for channel selection if available
    try:
        from ml.channel_bandit import LinUCBChannelBandit
        import os
        bandit_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'channel_bandit.joblib')
        if os.path.exists(bandit_path):
            bandit = LinUCBChannelBandit()
            bandit.load(bandit_path)
            bandit_ctx = {
                "age": customer.get("age", 35),
                "income_bracket": customer.get("income_bracket", "mid"),
                "risk_score": risk_score,
                "segment_type": customer.get("segment_type", "salaried"),
                "tenure_months": customer.get("tenure_months", 24),
                "num_dependents": customer.get("num_dependents", 1),
                "region": customer.get("region", "metro"),
                "credit_score": customer.get("credit_score", 700),
            }
            best_channel = bandit.select_channel(bandit_ctx)
            # Put bandit's choice first, keep others as fallback
            if best_channel in channels_to_use:
                channels_to_use = [best_channel] + [c for c in channels_to_use if c != best_channel]
            else:
                channels_to_use = [best_channel] + channels_to_use
            logger.info(f"[Dispatcher] Bandit selected: {best_channel} for {customer_id}")
    except Exception as e:
        logger.debug(f"[Dispatcher] Bandit not available: {e}")

    # P2: Accept journey metadata for nudge journey tracking
    journey_id = intervention.get("journey_id")
    journey_step = intervention.get("journey_step")

    logger.info(f"[Dispatcher] {customer_id} | Risk: {risk_score:.2f} | "
                f"Tier: {routing['tier']} | Channels: {channels_to_use}"
                f"{f' | Journey: {journey_id} Step {journey_step}' if journey_id else ''}")

    for channel in channels_to_use:
        # Check cooldown
        if not _check_cooldown(customer_id, channel):
            logger.info(f"[Dispatcher] {customer_id} | {channel} skipped (cooldown)")
            results.append({"channel": channel, "status": "skipped_cooldown",
                           "customer_id": customer_id})
            continue

        try:
            if channel == "sms":
                sms_msg = messages.get("sms", messages.get("fallback", ""))
                if sms_msg:
                    result = send_sms(customer, sms_msg)
                    results.append(result)

            elif channel == "email":
                email_html = messages.get("email_html", "")
                email_text = messages.get("email_text", messages.get("sms", ""))
                subject = messages.get("email_subject",
                    f"Important update from Barclays - {routing['action'].replace('_', ' ').title()}")
                if email_html or email_text:
                    result = send_email(customer, subject, email_html, email_text)
                    results.append(result)

            elif channel == "whatsapp":
                wa_msg = messages.get("whatsapp", messages.get("sms", ""))
                if wa_msg:
                    result = send_whatsapp(customer, wa_msg)
                    results.append(result)

            elif channel == "app_push":
                title = messages.get("push_title", "Financial wellness update")
                body = messages.get("push_body", messages.get("sms", ""))
                if body:
                    result = send_push_notification(customer, title, body)
                    results.append(result)

            elif channel == "rm_call":
                script = messages.get("rm_call_script", "")
                result = assign_rm_call(customer, risk_score, intervention, script)
                results.append(result)

            elif channel == "collector_assignment":
                brief = messages.get("collector_brief", "")
                result = assign_collector(customer, risk_score, intervention, brief)
                results.append(result)

        except Exception as e:
            logger.error(f"[Dispatcher] {channel} error for {customer_id}: {e}")
            results.append({"channel": channel, "status": "error",
                           "error": str(e), "customer_id": customer_id})

    return results


# ─────────────────────────────────────────────
# Convenience: Full pipeline in one call
# ─────────────────────────────────────────────
def process_and_notify(customer: Dict, intervention: Dict) -> List[Dict]:
    """
    Full notification pipeline:
    1. Generate messages via GenAI for each channel
    2. Route to appropriate channels
    3. Send/assign
    4. Log everything
    """
    from intervention.genai_messages import generate_multi_channel_messages

    messages = generate_multi_channel_messages(
        customer=customer,
        intervention=intervention,
    )

    return dispatch_notification(customer, intervention, messages)
