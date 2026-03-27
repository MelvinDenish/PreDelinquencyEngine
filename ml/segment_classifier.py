"""
Customer Segment Classifier
Classifies each customer into a segment type to ensure the right
feature logic, model weights, and intervention rules are applied.
"""
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

SEGMENTS = [
    "salaried", "self_employed", "gig_worker", "retiree",
    "agricultural", "nri", "student",
]


class CustomerSegmentClassifier:
    """Rule-based customer segment classifier."""

    # Employment types that map directly to segments
    EMPLOYMENT_SEGMENT_MAP = {
        "salaried_private": "salaried",
        "salaried_govt": "salaried",
        "retired": "retiree",
        "self_employed": "self_employed",
        "business_owner": "self_employed",
        "professional": "self_employed",
        "freelancer": "gig_worker",
        "gig_worker": "gig_worker",
        "contract_worker": "gig_worker",
    }

    # Agricultural regions / industry patterns
    AGRICULTURAL_INDUSTRIES = ["Agriculture", "Farming", "Dairy", "Fishery", "Horticulture"]

    def classify(self, customer: dict, transaction_summary: dict = None) -> str:
        """
        Classify a customer into a segment type.

        Args:
            customer: Customer profile dict from database
            transaction_summary: Optional dict with {
                'income_stddev_ratio': float,
                'has_regular_salary': bool,
                'primary_income_type': str,
                'remittance_pct': float,
            }

        Returns:
            Segment string: one of SEGMENTS
        """
        employment_type = customer.get("employment_type", "")
        age = customer.get("age", 30)
        region = customer.get("region", "")
        industry = customer.get("industry_sector", "")

        # 1. Retiree: age >= 60 or retired employment type
        if age >= 60 or employment_type == "retired":
            return "retiree"

        # 2. NRI: income primarily from inward remittances
        if transaction_summary:
            remittance_pct = transaction_summary.get("remittance_pct", 0)
            if remittance_pct > 0.80:
                return "nri"

        # 3. Agricultural: rural region with agricultural industry
        if industry in self.AGRICULTURAL_INDUSTRIES:
            return "agricultural"
        if region and region.lower() in ("rural",) and not customer.get("monthly_salary"):
            return "agricultural"

        # 4. Gig worker: check income volatility if transaction summary available
        if transaction_summary:
            income_stddev_ratio = transaction_summary.get("income_stddev_ratio", 0)
            has_regular_salary = transaction_summary.get("has_regular_salary", True)
            if income_stddev_ratio > 0.45 and not has_regular_salary:
                return "gig_worker"

        # 5. Employment type direct mapping
        segment = self.EMPLOYMENT_SEGMENT_MAP.get(employment_type)
        if segment:
            return segment

        # 6. Fallback to salaried
        return "salaried"

    def get_segment_feature_weights(self, segment: str) -> dict:
        """
        Get per-segment feature dampening weights.
        Weight < 1.0 reduces the feature's influence for this segment.
        Weight = 0.0 suppresses the feature entirely.
        """
        weights = {
            "salaried": {},  # No dampening — default behaviour
            "self_employed": {
                "salary_delay_days": 0.0,  # Not applicable
            },
            "gig_worker": {
                "salary_delay_days": 0.0,
                "spend_volatility_3m": 0.5,  # High volatility is normal
            },
            "retiree": {
                "gambling_lottery_spend_7d": 0.0,  # Medical misclassification
                "gambling_lottery_spend_30d": 0.0,
            },
            "agricultural": {
                "salary_delay_days": 0.0,  # Seasonal income, not salaried
                "spend_volatility_3m": 0.5,  # Seasonal patterns normal
            },
            "nri": {
                "salary_delay_days": 0.3,  # Forex delays expected
            },
            "student": {
                "salary_delay_days": 0.0,
            },
        }
        return weights.get(segment, {})

    def get_segment_thresholds(self, segment: str, base_thresholds: dict = None) -> dict:
        """
        Get risk tier thresholds adjusted for segment.
        Returns dict with 'watch' and 'critical' threshold values.
        """
        base = base_thresholds or {"watch": 0.50, "critical": 0.70}

        overrides = {
            "gig_worker": {"watch": 0.45, "critical": 0.75},
            "agricultural": {"watch": 0.45, "critical": 0.75},
            "retiree": {"watch": 0.50, "critical": 0.72},
            "nri": {"watch": 0.50, "critical": 0.72},
            "self_employed": {"watch": 0.48, "critical": 0.72},
        }

        result = dict(base)
        if segment in overrides:
            result.update(overrides[segment])
        return result
