"""
Product Action Engine (M10)
Generates proactive product-level intervention proposals that the bank
can offer without customer-initiated contact: EMI date shift, micro-payment
split, and interest rate concession review.
"""
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

import psycopg2
from psycopg2.extras import Json

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config.settings import PostgresConfig

logger = logging.getLogger(__name__)


class ProductActionEngine:
    """Generates product-level intervention proposals."""

    def offer_emi_date_shift(self, customer_id: str, current_due_day: int,
                              salary_credit_day: int) -> Optional[dict]:
        """
        Offer to shift EMI due date to align with salary credit.
        If the customer's salary credit day has drifted away from EMI due day,
        offer to move EMI to salary_credit_day + 3 days.
        """
        if abs(current_due_day - salary_credit_day) < 5:
            return None  # Already aligned

        proposed_due_day = min(28, salary_credit_day + 3)

        proposal = {
            "customer_id": customer_id,
            "action_type": "emi_date_shift",
            "proposed_params": {
                "current_due_day": current_due_day,
                "salary_credit_day": salary_credit_day,
                "proposed_due_day": proposed_due_day,
                "reason": f"Salary credits on day {salary_credit_day}, "
                          f"EMI due on day {current_due_day}. "
                          f"Shift to day {proposed_due_day} for better alignment.",
            },
        }
        self._save_proposal(proposal)
        return proposal

    def offer_micro_payment_split(self, customer_id: str, emi_amount: float,
                                    risk_score: float) -> Optional[dict]:
        """
        Offer to split a large EMI into 4 weekly micro-payments.
        Only for watch-tier customers with EMI > ₹5,000.
        """
        if emi_amount < 5000 or risk_score < 0.50:
            return None

        weekly_amount = round(emi_amount / 4, 2)

        proposal = {
            "customer_id": customer_id,
            "action_type": "micro_payment_split",
            "proposed_params": {
                "original_emi": emi_amount,
                "weekly_amount": weekly_amount,
                "num_installments": 4,
                "reason": f"Monthly EMI of ₹{emi_amount:,.0f} split into "
                          f"4 weekly payments of ₹{weekly_amount:,.0f}.",
            },
        }
        self._save_proposal(proposal)
        return proposal

    def flag_for_interest_rate_review(self, customer_id: str,
                                       risk_score: float,
                                       credit_score: int) -> Optional[dict]:
        """
        Flag a watch-tier customer for interest rate concession review.
        25-50 bps reduction depending on score.
        """
        if risk_score < 0.50 or risk_score > 0.65:
            return None

        bps_reduction = 50 if credit_score > 700 else 25

        proposal = {
            "customer_id": customer_id,
            "action_type": "interest_rate_review",
            "proposed_params": {
                "current_risk_score": round(risk_score, 4),
                "credit_score": credit_score,
                "proposed_bps_reduction": bps_reduction,
                "reason": f"Watch-tier customer with credit score {credit_score}. "
                          f"Propose {bps_reduction} bps rate reduction to prevent escalation.",
            },
        }
        self._save_proposal(proposal)
        return proposal

    def generate_proposals(self, customer_id: str, customer: dict,
                            risk_score: float, risk_tier: str) -> List[dict]:
        """
        Generate all applicable product action proposals for a customer.
        """
        proposals = []

        if risk_tier not in ("watch", "critical"):
            return proposals

        # EMI date shift
        emi_proposal = self.offer_emi_date_shift(
            customer_id,
            current_due_day=customer.get("emi_due_day", 5),
            salary_credit_day=customer.get("salary_credit_day", 1),
        )
        if emi_proposal:
            proposals.append(emi_proposal)

        # Micro-payment split
        emi_amount = customer.get("monthly_salary", 50000) * customer.get("dti_ratio", 0.3)
        split_proposal = self.offer_micro_payment_split(
            customer_id, emi_amount, risk_score
        )
        if split_proposal:
            proposals.append(split_proposal)

        # Interest rate review (watch only, not critical)
        if risk_tier == "watch":
            rate_proposal = self.flag_for_interest_rate_review(
                customer_id, risk_score,
                credit_score=customer.get("credit_score", 700),
            )
            if rate_proposal:
                proposals.append(rate_proposal)

        return proposals

    def _save_proposal(self, proposal: dict):
        """Save proposal to product_action_proposals table."""
        try:
            conn = psycopg2.connect(
                host=PostgresConfig.HOST, port=PostgresConfig.PORT,
                user=PostgresConfig.USER, password=PostgresConfig.PASSWORD,
                dbname=PostgresConfig.DB,
            )
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO product_action_proposals
                   (customer_id, action_type, proposed_params)
                   VALUES (%s, %s, %s)""",
                (proposal["customer_id"], proposal["action_type"],
                 Json(proposal["proposed_params"])),
            )
            conn.commit()
            cursor.close()
            conn.close()
            logger.info(
                f"[ProductAction] Saved {proposal['action_type']} "
                f"proposal for {proposal['customer_id']}"
            )
        except Exception as e:
            logger.warning(f"[ProductAction] Failed to save proposal: {e}")
