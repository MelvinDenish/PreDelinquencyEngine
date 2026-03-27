# pyre-ignore-all-errors
"""
GenAI-Powered Multi-Channel Message Generator
Uses Groq's LLM API to generate personalized, empathetic intervention messages
across SMS, Email, WhatsApp, RM Call Scripts, and Collector Briefs.
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
# Fallback templates per channel
# ─────────────────────────────────────────────
FALLBACK_SMS = {
    "payment_holiday": "Dear {name}, we'd like to offer a payment holiday on your EMI. Reply YES to activate. Barclays 1800-XXX-XXXX",
    "emi_restructuring": "Dear {name}, lower your EMI by up to 30% — no fees. Reply RESTRUCTURE or visit your branch. Barclays",
    "wellness_checkin": "Dear {name}, free financial wellness check available. Book at barclays.in/wellness or call 1800-XXX-XXXX. Barclays",
    "budget_nudge": "Hi {name}, spending trending higher this week. Set a daily limit in the Barclays app → barclays.in/budget",
    "payment_reminder": "Hi {name}, EMI of ₹{emi_amount} due in 3 days. Pay via app to avoid late fees. Barclays",
    "escalation_call": "Dear {name}, your RM would like to discuss some financial support options. Expect a call today. Barclays",
}

FALLBACK_EMAIL_HTML = """
<html><body style="font-family:Arial,sans-serif;max-width:600px;margin:auto;background:#f9f9f9;padding:20px;">
<div style="background:#003087;padding:20px;text-align:center;">
  <h2 style="color:white;margin:0;">Barclays India</h2>
  <p style="color:#72b0e0;margin:5px 0 0;">Financial Wellness Team</p>
</div>
<div style="background:white;padding:30px;border:1px solid #e0e0e0;">
  <p>Dear {name},</p>
  <p>{body}</p>
  <p style="margin-top:20px;"><a href="https://barclays.in/wellness" style="background:#003087;color:white;padding:12px 24px;text-decoration:none;border-radius:4px;">Explore Your Options</a></p>
  <p style="color:#666;font-size:12px;margin-top:30px;">This is an automated message from Barclays India's Financial Wellness Team. If you did not expect this, please call 1800-XXX-XXXX.</p>
</div>
</body></html>
"""

FALLBACK_RM_SCRIPT = """
RELATIONSHIP MANAGER CALL BRIEF
================================
Customer: {name} ({customer_id})
City: {city} | Tenure: {tenure}mo | Income: ₹{salary}/mo
Risk Score: {risk_score:.1%} | Tier: {risk_tier}
Top Driver: {top_driver}

TALKING POINTS:
1. Open with genuine care — ask about their day
2. Reference their long relationship with Barclays ({tenure} months)
3. Mention you noticed they might benefit from a financial check-up
4. Present the {intervention_type} option as something available to valued customers
5. NEVER mention risk score, delinquency, or collections
6. If they accept, guide them through next steps
7. If they decline, offer to schedule a callback and note preferences

OFFERS TO PRESENT:
- {offer_details}

ESCALATION: If customer shows signs of severe distress, escalate to collections team.
"""

FALLBACK_COLLECTOR_BRIEF = """
COLLECTIONS ASSIGNMENT BRIEF
===============================
Priority: {priority}
Customer: {name} ({customer_id})
Phone: {phone} | Email: {email}
City: {city} | State: {state}
Employment: {employment_type} | Industry: {industry}

FINANCIAL PROFILE:
- Monthly Income: ₹{salary}/mo
- DTI Ratio: {dti:.1%}
- Active Loans: {num_loans}
- Total Outstanding: ₹{total_debt}
- Credit Score: {credit_score}
- Risk Score: {risk_score:.1%}
- Tier: {risk_tier}

STRESS TRIGGERS:
- Life Event: {life_event} ({life_event_severity} severity, {life_event_months_ago}mo ago)
- Top Risk Driver: {top_driver}

RESTRUCTURING OFFER AUTHORIZED:
- EMI reduction: up to {emi_reduction}%
- Tenure extension: {tenure_ext} months
- Payment holiday: {payment_holiday} months
- Interest concession: {interest_concession} bps

APPROACH GUIDELINES:
1. Initial contact via {preferred_channel}
2. Be empathetic — this is pre-delinquency intervention, not collections
3. Lead with support options, not payment demands
4. Document all interactions in CRM
5. Escalate to branch head if customer unresponsive after 3 attempts
"""

# ─────────────────────────────────────────────
# System prompts
# ─────────────────────────────────────────────
SYSTEM_PROMPT_SMS = """You are an empathetic financial communication specialist at Barclays India.
Write a SHORT SMS (under 160 chars) that is warm, supportive, and suggests help — never accusatory.
Use ₹ symbol and Indian English. Never say "delinquency", "default", "risk", or "stress".
Include CTA: reply keyword, call number, or app link."""

SYSTEM_PROMPT_EMAIL = """You are a financial wellness advisor at Barclays India.
Write a warm, professional EMAIL BODY (3-5 sentences). Focus on HELP, not collections.
Use Indian English and ₹. Include specific offer details. Never mention risk scores.
Return ONLY the email body text (no subject line, no HTML tags)."""

SYSTEM_PROMPT_WHATSAPP = """You are Barclays India's wellness bot. Write a WhatsApp message (2-3 lines).
Be friendly, use emojis sparingly (max 2). Include CTA. Never mention risk/default.
Use Indian English and ₹ for amounts."""

SYSTEM_PROMPT_RM = """You are writing a CALL SCRIPT for a relationship manager at Barclays India.
Include: opening, talking points, offer details, objection handling, and next steps.
Be specific with customer details. Structure clearly with bullet points.
Tone: supportive, proactive, never threatening."""

SYSTEM_PROMPT_COLLECTOR = """You are writing a COLLECTION CASE BRIEF for Barclays India's recovery team.
Be factual and structured. Include customer profile, financial situation, stress triggers,
authorized restructuring offer, and recommended approach. Use professional tone.
Lead with empathy — this is PRE-delinquency, not post-default."""


def _call_groq(system_prompt: str, user_prompt: str, max_tokens: int = 400) -> Optional[str]:
    """Call Groq LLM API. Returns generated text or None."""
    if not GROQ_API_KEY:
        return None
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=max_tokens,
            top_p=0.9,
        )
        text = response.choices[0].message.content.strip()
        # Remove quotes LLM might wrap
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        return text
    except ImportError:
        logger.warning("[GenAI] groq package not installed")
        return None
    except Exception as e:
        logger.warning(f"[GenAI] Groq error: {e}")
        return None


def _build_context(customer: Dict, intervention: Dict) -> str:
    """Build customer context string for LLM prompts."""
    drivers = intervention.get("shap_drivers", [])
    driver_text = "\n".join([
        f"  - {d['feature']}: impact={d.get('shap_value', 0):.3f}"
        for d in drivers[:3]
    ]) if drivers else "  (no specific drivers identified)"

    return f"""Customer: {customer.get('first_name', '')} {customer.get('last_name', '')}
City: {customer.get('city', '?')}, {customer.get('region', '?')}
Age: {customer.get('age', '?')} | Gender: {customer.get('gender', '?')}
Employment: {customer.get('employment_type', '?')} in {customer.get('industry_sector', '?')}
Monthly Income: ₹{customer.get('monthly_salary', 0):,.0f}
DTI Ratio: {customer.get('dti_ratio', 0):.1%}
Tenure: {customer.get('tenure_months', 0)} months
Credit Score: {customer.get('credit_score', 0)}
Products: {', '.join(customer.get('product_holdings', []))}
Life Event: {customer.get('life_event', 'none')} ({customer.get('life_event_severity', 'none')})

Intervention: {intervention.get('intervention_type', '?')}
Risk Score: {intervention.get('risk_score', 0):.2f}
Risk Tier: {intervention.get('risk_tier', '?')}
Top stress signals:
{driver_text}"""


def generate_sms(customer: Dict, intervention: Dict) -> str:
    """Generate SMS message."""
    context = _build_context(customer, intervention)
    result = _call_groq(SYSTEM_PROMPT_SMS, f"Generate SMS for:\n{context}", max_tokens=80)
    if result:
        return result[:160]  # Hard SMS limit

    # Fallback
    itype = intervention.get("intervention_type", "wellness_checkin")
    template = FALLBACK_SMS.get(itype, FALLBACK_SMS["wellness_checkin"])
    return template.format(
        name=customer.get("first_name", "Customer"),
        emi_amount=f"{customer.get('monthly_salary', 50000) * 0.2:,.0f}",
    )


def generate_email(customer: Dict, intervention: Dict) -> Dict:
    """Generate email subject + HTML body."""
    context = _build_context(customer, intervention)
    body_text = _call_groq(SYSTEM_PROMPT_EMAIL,
        f"Generate email body for:\n{context}", max_tokens=300)

    if not body_text:
        itype = intervention.get("intervention_type", "wellness_checkin")
        body_text = {
            "payment_holiday": f"We've noticed some changes in your recent financial activity, "
                f"and we want you to know we're here to help. As a valued customer of {customer.get('tenure_months', 0)} months, "
                f"you're eligible for a payment holiday that lets you skip your next EMI — "
                f"with zero impact on your credit score.",
            "emi_restructuring": f"Managing monthly payments can sometimes feel like a juggling act. "
                f"We'd like to offer you an EMI restructuring option that could reduce your monthly "
                f"payment by up to 30%, with no processing fees.",
            "wellness_checkin": f"At Barclays, your financial wellness matters to us. "
                f"We have certified financial advisors available for a free, confidential wellness "
                f"check that can help you plan better for the months ahead.",
            "escalation_call": f"Your relationship manager would like to personally connect with you "
                f"to discuss some exclusive financial support options available to you.",
        }.get(itype, "We have some financial wellness options we'd like to share with you.")

    subject = {
        "payment_holiday": "A special payment flexibility option for you — Barclays",
        "emi_restructuring": "Lower your EMI — exclusive offer inside — Barclays",
        "wellness_checkin": "Your free financial wellness check-up — Barclays",
        "budget_nudge": "Quick spending insights for you — Barclays",
        "payment_reminder": "Upcoming payment reminder — Barclays",
        "escalation_call": "Your RM would like to connect — Barclays",
    }.get(intervention.get("intervention_type", ""), "Important update — Barclays India")

    html = FALLBACK_EMAIL_HTML.format(
        name=customer.get("first_name", "Customer"),
        body=body_text.replace("\n", "<br>"),
    )

    return {"subject": subject, "html": html, "text": body_text}


def generate_whatsapp(customer: Dict, intervention: Dict) -> str:
    """Generate WhatsApp message."""
    context = _build_context(customer, intervention)
    result = _call_groq(SYSTEM_PROMPT_WHATSAPP,
        f"Generate WhatsApp message:\n{context}", max_tokens=150)
    if result:
        return result

    name = customer.get("first_name", "Customer")
    return (f"Hi {name} 👋\n"
            f"Barclays here — we have a financial support option for you. "
            f"Reply HI to learn more or call 1800-XXX-XXXX.")


def generate_rm_call_script(customer: Dict, intervention: Dict) -> str:
    """Generate RM call script with full briefing."""
    context = _build_context(customer, intervention)
    result = _call_groq(SYSTEM_PROMPT_RM,
        f"Generate RM call script:\n{context}", max_tokens=500)
    if result:
        return result

    drivers = intervention.get("shap_drivers", [])
    top_driver = drivers[0]["feature"] if drivers else "general_risk_indicators"

    return FALLBACK_RM_SCRIPT.format(
        name=f"{customer.get('first_name', '')} {customer.get('last_name', '')}",
        customer_id=customer.get("customer_id", ""),
        city=customer.get("city", ""), tenure=customer.get("tenure_months", 0),
        salary=f"{customer.get('monthly_salary', 0):,.0f}",
        risk_score=intervention.get("risk_score", 0),
        risk_tier=intervention.get("risk_tier", ""),
        top_driver=top_driver,
        intervention_type=intervention.get("intervention_type", "wellness_checkin"),
        offer_details="EMI restructuring (up to 30% reduction) or payment holiday (1-3 months)",
    )


def generate_collector_brief(customer: Dict, intervention: Dict) -> str:
    """Generate collector assignment brief."""
    context = _build_context(customer, intervention)
    result = _call_groq(SYSTEM_PROMPT_COLLECTOR,
        f"Generate collections brief:\n{context}", max_tokens=600)
    if result:
        return result

    risk_score = intervention.get("risk_score", 0)
    drivers = intervention.get("shap_drivers", [])
    top_driver = drivers[0]["feature"] if drivers else "general"

    return FALLBACK_COLLECTOR_BRIEF.format(
        priority="P0" if risk_score >= 0.9 else "P1",
        name=f"{customer.get('first_name', '')} {customer.get('last_name', '')}",
        customer_id=customer.get("customer_id", ""),
        phone=customer.get("phone", "N/A"), email=customer.get("email", "N/A"),
        city=customer.get("city", ""), state=customer.get("state", ""),
        employment_type=customer.get("employment_type", ""),
        industry=customer.get("industry_sector", ""),
        salary=f"{customer.get('monthly_salary', 0):,.0f}",
        dti=customer.get("dti_ratio", 0),
        num_loans=customer.get("num_active_loans", 0),
        total_debt=f"{customer.get('total_debt_outstanding', 0):,.0f}",
        credit_score=customer.get("credit_score", 0),
        risk_score=risk_score, risk_tier=intervention.get("risk_tier", ""),
        life_event=customer.get("life_event", "none"),
        life_event_severity=customer.get("life_event_severity", "none"),
        life_event_months_ago=customer.get("life_event_months_ago", 0),
        top_driver=top_driver,
        emi_reduction=min(int((customer.get("dti_ratio", 0.5) - 0.4) * 100), 50),
        tenure_ext=12 if risk_score > 0.9 else 6,
        payment_holiday=3 if risk_score > 0.9 else 1,
        interest_concession=50 if risk_score > 0.9 else 25,
        preferred_channel=customer.get("preferred_channel", "sms"),
    )


# ─────────────────────────────────────────────
# Main: Generate all channel messages in one call
# ─────────────────────────────────────────────
def generate_multi_channel_messages(customer: Dict, intervention: Dict) -> Dict:
    """
    Generate messages for ALL channels in one call.
    Returns dict with keys: sms, email_html, email_text, email_subject,
    whatsapp, push_title, push_body, rm_call_script, collector_brief
    """
    risk_score = intervention.get("risk_score", 0)

    # Always generate SMS (used as fallback for other channels too)
    sms = generate_sms(customer, intervention)

    messages = {
        "sms": sms,
        "whatsapp": sms,  # Default; override below if score warrants
        "push_title": "Financial wellness update",
        "push_body": sms,
        "fallback": sms,
    }

    # Generate channel-specific content based on risk level
    if risk_score >= 0.50:
        messages["whatsapp"] = generate_whatsapp(customer, intervention)

    if risk_score >= 0.65:
        email = generate_email(customer, intervention)
        messages["email_subject"] = email["subject"]
        messages["email_html"] = email["html"]
        messages["email_text"] = email["text"]

    if risk_score >= 0.75:
        messages["rm_call_script"] = generate_rm_call_script(customer, intervention)

    if risk_score >= 0.85:
        messages["collector_brief"] = generate_collector_brief(customer, intervention)

    logger.info(f"[GenAI] Generated {len(messages)} message types for "
                f"{customer.get('customer_id')} (score={risk_score:.2f})")

    return messages


# Legacy compatibility
def generate_intervention_with_genai(customer: Dict, intervention: Dict) -> Dict:
    """Legacy wrapper — generates messages and adds to intervention dict."""
    messages = generate_multi_channel_messages(customer, intervention)
    intervention["genai_message"] = messages.get("sms", "")
    intervention["message_channel"] = customer.get("preferred_channel", "sms")
    intervention["genai_model"] = GROQ_MODEL if GROQ_API_KEY else "fallback_template"
    intervention["multi_channel_messages"] = messages
    return intervention
