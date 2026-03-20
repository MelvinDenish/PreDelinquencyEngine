"""
GenAI-Powered Intervention Message Generator
Uses Groq's LLM API to generate personalized, empathetic intervention messages
based on customer risk profiles and SHAP-driven stress signals.

This fulfills the problem statement's requirement for "GenAI-driven risk detection"
by producing tailored, context-aware outreach messaging.
"""
import os
import sys
import json
import logging
from typing import Dict, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ─────────────────────────────────────────────
# Fallback templates when Groq is unavailable
# ─────────────────────────────────────────────
FALLBACK_TEMPLATES = {
    "payment_holiday": (
        "Dear {name}, we noticed some changes in your financial activity recently. "
        "We'd like to offer you a payment holiday on your upcoming EMI. This gives you "
        "breathing room without any impact on your credit score. Reply YES to activate, "
        "or call us at 1800-XXX-XXXX."
    ),
    "emi_restructuring": (
        "Dear {name}, managing EMIs can be challenging during financial transitions. "
        "We have an EMI restructuring option that could lower your monthly payments by "
        "up to 30%. No fees, no credit impact. Would you like to explore this? "
        "Reply RESTRUCTURE or visit your nearest branch."
    ),
    "wellness_checkin": (
        "Dear {name}, at Barclays we care about your financial wellness. We've noticed "
        "some patterns that suggest you might benefit from a quick financial health check. "
        "Our certified advisors can help — it's free and confidential. Book a slot at "
        "barclays.in/wellness or call 1800-XXX-XXXX."
    ),
    "budget_nudge": (
        "Hi {name}, quick heads up — your spending this week is trending higher than "
        "usual. Here's a tip: setting a daily spending limit in the app can help you "
        "stay on track. Tap here to set it up → barclays.in/budget"
    ),
    "payment_reminder": (
        "Hi {name}, friendly reminder — your payment of ₹{amount} is due in 3 days. "
        "Pay now through the app to avoid any late fees. Need help? We're here for you."
    ),
    "escalation_call": (
        "Dear {name}, this is a priority message from Barclays. Your relationship "
        "manager {rm_name} would like to discuss some options to support your "
        "financial goals. Please expect a call today, or reach us at 1800-XXX-XXXX."
    ),
}

# ─────────────────────────────────────────────
# System prompt for the LLM
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are an empathetic financial communication specialist at Barclays Bank India.
Your job is to write short, warm, and supportive intervention messages for customers showing early signs of financial stress.

RULES:
1. Be warm, supportive, and non-judgmental — never accusatory
2. Focus on HELP, not collections
3. Keep messages under 3 sentences for SMS, 5 sentences for email
4. Include a clear call-to-action
5. Never mention "delinquency", "default", "risk score", or "financial stress" explicitly
6. Use Indian English conventions (₹ symbol, "EMI", "UPI")
7. Personalise using the customer's first name
8. Maintain Barclays' premium brand voice
"""


def _build_prompt(customer: Dict, intervention_type: str,
                  risk_score: float, shap_drivers: list,
                  channel: str = "sms") -> str:
    """Build the LLM prompt with customer context."""
    driver_text = ""
    if shap_drivers:
        top3 = shap_drivers[:3]
        driver_text = "\n".join([
            f"  - {d['feature']}: value={d.get('value', 'N/A')}, "
            f"impact={d.get('shap_value', 0):.3f}"
            for d in top3
        ])

    return f"""Generate a {channel.upper()} intervention message for this customer:

Customer: {customer.get('first_name', 'Customer')} {customer.get('last_name', '')}
City: {customer.get('city', 'Unknown')}
Region: {customer.get('region', 'Unknown')}
Income bracket: {customer.get('income_bracket', 'middle')}
Tenure: {customer.get('tenure_months', 0)} months

Intervention type: {intervention_type}
Top stress signals:
{driver_text}

Channel: {channel} (keep it {'very short, under 160 chars' if channel == 'sms' else 'concise but warm'})

Write ONLY the message text, nothing else."""


def generate_message_groq(
    customer: Dict,
    intervention_type: str,
    risk_score: float,
    shap_drivers: list,
    channel: str = "sms",
) -> str:
    """Generate a personalized intervention message using Groq's LLM API."""

    # Try Groq API
    if GROQ_API_KEY:
        try:
            from groq import Groq

            client = Groq(api_key=GROQ_API_KEY)

            prompt = _build_prompt(customer, intervention_type,
                                   risk_score, shap_drivers, channel)

            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=300,
                top_p=0.9,
            )

            message = response.choices[0].message.content.strip()

            # Remove any quotes the LLM wraps around the message
            if message.startswith('"') and message.endswith('"'):
                message = message[1:-1]
            if message.startswith("'") and message.endswith("'"):
                message = message[1:-1]

            logger.info(f"[GenAI] Generated {channel} message for "
                       f"{customer.get('customer_id', '?')} via Groq ({GROQ_MODEL})")
            return message

        except ImportError:
            logger.warning("[GenAI] groq package not installed. Using fallback template.")
        except Exception as e:
            logger.warning(f"[GenAI] Groq API error: {e}. Using fallback template.")

    # Fallback to template
    return _generate_fallback(customer, intervention_type)


def _generate_fallback(customer: Dict, intervention_type: str) -> str:
    """Generate message from static templates when Groq is unavailable."""
    template = FALLBACK_TEMPLATES.get(intervention_type,
                                       FALLBACK_TEMPLATES["wellness_checkin"])
    return template.format(
        name=customer.get("first_name", "Customer"),
        amount="XX,XXX",
        rm_name="your dedicated RM",
    )


def generate_intervention_with_genai(
    customer: Dict,
    intervention: Dict,
) -> Dict:
    """
    Enhance an intervention record with a GenAI-generated message.
    Returns the intervention dict with an added 'genai_message' field.
    """
    message = generate_message_groq(
        customer=customer,
        intervention_type=intervention.get("intervention_type", "wellness_checkin"),
        risk_score=intervention.get("risk_score", 0),
        shap_drivers=intervention.get("shap_drivers", []),
        channel=customer.get("preferred_channel", "sms"),
    )

    intervention["genai_message"] = message
    intervention["message_channel"] = customer.get("preferred_channel", "sms")
    intervention["genai_model"] = GROQ_MODEL if GROQ_API_KEY else "fallback_template"

    return intervention


if __name__ == "__main__":
    # Quick test
    test_customer = {
        "customer_id": "CUST_TEST001",
        "first_name": "Priya",
        "last_name": "Sharma",
        "city": "Chennai",
        "region": "South",
        "income_bracket": "middle",
        "tenure_months": 36,
        "preferred_channel": "sms",
    }

    test_intervention = {
        "intervention_type": "payment_holiday",
        "risk_score": 0.72,
        "shap_drivers": [
            {"feature": "salary_delay_days", "value": 12, "shap_value": 0.18},
            {"feature": "savings_balance_pct_change_7d", "value": -0.35, "shap_value": 0.15},
        ],
    }

    result = generate_intervention_with_genai(test_customer, test_intervention)
    print(f"\nGenerated message ({result['genai_model']}):")
    print(f"  Channel: {result['message_channel']}")
    print(f"  Message: {result['genai_message']}")
