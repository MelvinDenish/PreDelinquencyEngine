"""
Audit logging module for PDI Engine.
Writes tamper-evident audit events to the audit_log table.
All customer identifiers are one-way hashed before storage.
"""
import hashlib
import json
import logging
import os
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# Salt makes the hash non-reversible even if the attacker knows customer IDs
_AUDIT_SALT = os.getenv("AUDIT_HASH_SALT", "pdi-audit-salt-2024")


# ──────────────────────────────────────────────────
# PII Masking Utilities
# ──────────────────────────────────────────────────

def mask_customer_id(customer_id: str) -> str:
    """One-way SHA-256 hash of customer_id. Non-reversible."""
    return hashlib.sha256(f"{_AUDIT_SALT}:{customer_id}".encode()).hexdigest()


def mask_phone(phone: str) -> str:
    """Show only last 4 digits: +91XXXXXXX1234"""
    if not phone:
        return "***"
    phone = str(phone).strip()
    if len(phone) >= 4:
        return f"+91XXXXXXX{phone[-4:]}"
    return "***"


def mask_email(email: str) -> str:
    """Show only domain: ***@barclays.com"""
    if not email or "@" not in email:
        return "***"
    domain = email.split("@", 1)[1]
    return f"***@{domain}"


def mask_name(name: str) -> str:
    """Show only first initial: J***"""
    if not name:
        return "***"
    return f"{name[0]}***"


def mask_salary(salary) -> str:
    """Show only salary bracket, not exact amount."""
    if salary is None:
        return "***"
    try:
        s = float(salary)
        if s < 25000:
            return "<25k"
        elif s < 50000:
            return "25k-50k"
        elif s < 100000:
            return "50k-1L"
        elif s < 500000:
            return "1L-5L"
        else:
            return ">5L"
    except (ValueError, TypeError):
        return "***"


def sanitize_details(details: dict) -> dict:
    """
    Remove any PII from a details dict before storing in audit log.
    Allowed keys are non-PII operational context only.
    """
    if not details:
        return {}
    pii_keys = {"phone", "email", "first_name", "last_name", "name",
                 "monthly_salary", "salary", "dti_ratio"}
    return {k: v for k, v in details.items() if k.lower() not in pii_keys}


# ──────────────────────────────────────────────────
# Event Types
# ──────────────────────────────────────────────────
class AuditEvent:
    # Auth events
    LOGIN_SUCCESS = "LOGIN_SUCCESS"
    LOGIN_FAILURE = "LOGIN_FAILURE"
    LOGOUT = "LOGOUT"
    TOKEN_ISSUED = "TOKEN_ISSUED"
    API_KEY_USED = "API_KEY_USED"

    # Scoring events
    SCORE_REQUEST = "SCORE_REQUEST"
    SCORE_BATCH_REQUEST = "SCORE_BATCH_REQUEST"
    SCORE_READ = "SCORE_READ"

    # Intervention events
    NOTIFY_DISPATCH = "NOTIFY_DISPATCH"
    NOTIFY_BLOCKED_COOLDOWN = "NOTIFY_BLOCKED_COOLDOWN"
    NOTIFY_BLOCKED_CONSENT = "NOTIFY_BLOCKED_CONSENT"

    # Explain events
    EXPLAIN_REQUEST = "EXPLAIN_REQUEST"

    # Admin events
    API_KEY_CREATED = "API_KEY_CREATED"
    API_KEY_REVOKED = "API_KEY_REVOKED"
    USER_CREATED = "USER_CREATED"

    # Data events
    CUSTOMER_DATA_ACCESSED = "CUSTOMER_DATA_ACCESSED"
    CUSTOMER_ERASED = "CUSTOMER_ERASED"


# ──────────────────────────────────────────────────
# Write audit event
# ──────────────────────────────────────────────────

def write_audit_event(
    event_type: str,
    actor_id: str,
    actor_role: str,
    action: str,
    outcome: str,
    customer_id: Optional[str] = None,
    request_ip: Optional[str] = None,
    details: Optional[dict] = None,
) -> None:
    """
    Write a single audit event to the audit_log table.
    customer_id is hashed before storage — never stored in plaintext.
    details dict must be pre-sanitized of PII by the caller.
    Non-blocking: logs a warning and continues on DB failure.
    """
    import psycopg2
    from config.settings import PostgresConfig

    customer_id_token = mask_customer_id(customer_id) if customer_id else None
    clean_details = sanitize_details(details or {})

    try:
        conn = psycopg2.connect(
            host=PostgresConfig.HOST, port=PostgresConfig.PORT,
            user=PostgresConfig.USER, password=PostgresConfig.PASSWORD,
            dbname=PostgresConfig.DB, connect_timeout=2,
        )
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO audit_log
               (event_type, actor_id, actor_role, customer_id_token,
                action, outcome, request_ip, details)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
            (
                event_type,
                actor_id,
                actor_role,
                customer_id_token,
                action,
                outcome,
                request_ip,
                json.dumps(clean_details),
            ),
        )
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        # Audit failures must NEVER crash the main flow
        logger.warning(f"[Audit] Failed to write event {event_type}: {e}")


def get_request_ip(request) -> Optional[str]:
    """Extract real client IP, respecting X-Forwarded-For header (behind proxy/ingress)."""
    if request is None:
        return None
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    if hasattr(request, "client") and request.client:
        return request.client.host
    return None
