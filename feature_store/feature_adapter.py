"""
Feature Adapter — Bank-Aware Feature Translation Layer
Maps each bank's proprietary transaction taxonomy to the engine's
canonical feature names. Handles segment-specific feature dampening.
"""
import logging
from typing import Dict, Optional

from config.bank_config import BankProfile, BankProfileLoader

logger = logging.getLogger(__name__)


class FeatureAdapter:
    """Adapts raw bank data to engine-canonical features."""

    def __init__(self, bank_profile: BankProfile = None):
        self.profile = bank_profile or BankProfileLoader.get_active_profile()

    def adapt_transaction_category(self, raw_category: str) -> str:
        """Map a bank-specific MCC/category label to engine canonical category."""
        if not raw_category:
            return "other"
        canonical = self.profile.category_mapping.get(raw_category)
        if canonical:
            return canonical
        # Identity mapping if bank uses engine-standard names
        return raw_category.lower().strip()

    def is_income_credit(self, txn_type: str) -> bool:
        """Check if a transaction type represents income credit for this bank."""
        return txn_type in self.profile.income_credit_tags

    def is_discretionary(self, merchant_category: str) -> bool:
        """Check if a merchant category is discretionary for this bank."""
        canonical = self.adapt_transaction_category(merchant_category)
        return canonical in self.profile.discretionary_categories

    def is_lending_app(self, merchant_id: str, merchant_category: str = "") -> bool:
        """Check if merchant is a known lending app."""
        if merchant_id and any(
            lender in merchant_id.lower() for lender in self.profile.lending_app_merchants
        ):
            return True
        canonical = self.adapt_transaction_category(merchant_category)
        return canonical in ("lending_app", "payday_lender", "cash_advance")

    def is_distress_merchant(self, merchant_category: str) -> bool:
        """Check if merchant indicates financial distress."""
        canonical = self.adapt_transaction_category(merchant_category)
        return canonical in self.profile.distress_merchant_tags

    def is_medical_merchant(self, merchant_category: str) -> bool:
        """Check if merchant is medical/healthcare."""
        canonical = self.adapt_transaction_category(merchant_category)
        return canonical in self.profile.medical_merchant_tags

    def get_risk_thresholds(self, segment_type: str = None) -> Dict[str, float]:
        """Get risk tier thresholds, optionally segment-adjusted."""
        base = dict(self.profile.risk_thresholds)
        if segment_type and segment_type in self.profile.segment_thresholds:
            base.update(self.profile.segment_thresholds[segment_type])
        return base

    def adapt_feature_vector(self, raw_features: Dict, segment_type: str = None) -> Dict:
        """
        Apply segment-specific feature dampening and normalisation.
        Returns adapted feature dict with NaN for unavailable features
        (not 0 — model must distinguish zero from missing).
        """
        features = dict(raw_features)

        if segment_type == "agricultural":
            # Salary delay not meaningful for agricultural customers
            features["salary_delay_days"] = 0
            # Reduce spend volatility weight (seasonal income is normal)
            if "spend_volatility_3m" in features:
                features["spend_volatility_3m"] = features["spend_volatility_3m"] * 0.5

        elif segment_type == "retiree":
            # Medical spend should not count as discretionary stress
            features.pop("gambling_lottery_spend_7d", None)
            features.pop("gambling_lottery_spend_30d", None)

        elif segment_type == "nri":
            # Add forex delay buffer — remittances can take up to 7 business days
            if "salary_delay_days" in features:
                features["salary_delay_days"] = max(0, features["salary_delay_days"] - 7)

        elif segment_type == "gig_worker":
            # Use wider baseline for spend normalisation
            if "spend_volatility_3m" in features:
                features["spend_volatility_3m"] = features["spend_volatility_3m"] * 0.5

        elif segment_type == "self_employed":
            # Salary delay not applicable — suppress
            features["salary_delay_days"] = 0

        return features

    def get_intervention_cost(self, channel: str) -> float:
        """Get cost of an intervention channel in local currency."""
        return self.profile.intervention_costs.get(channel, 0.0)
