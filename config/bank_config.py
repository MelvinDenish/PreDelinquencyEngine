# pyre-ignore-all-errors
"""
Multi-Bank Adapter — BankProfile Configuration
Loads bank-specific YAML configuration files that define taxonomy,
thresholds, channel providers, and regulatory flags per bank.
"""
import os
import yaml
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).parent / "banks"


@dataclass
class BankProfile:
    """Complete bank configuration profile."""
    bank_id: str = "GENERIC"
    display_name: str = "Bank"
    currency: str = "INR"
    locale: str = "en-IN"

    # Transaction taxonomy mapping (bank's terms → engine's canonical terms)
    income_credit_tags: List[str] = field(default_factory=lambda: ["salary_credit"])
    discretionary_categories: List[str] = field(
        default_factory=lambda: ["dining", "entertainment", "clothing", "luxury_goods", "travel"]
    )
    emi_merchant_tags: List[str] = field(default_factory=lambda: ["emi", "auto_debit"])
    lending_app_merchants: List[str] = field(
        default_factory=lambda: [
            "kreditbee", "moneytap", "paysense", "navi", "kissht",
            "cashe", "early_salary", "flexsalary", "mpokket", "dhani",
        ]
    )
    distress_merchant_tags: List[str] = field(
        default_factory=lambda: ["pawnshop", "gold_loan", "chit_fund", "money_lender", "payday_lender"]
    )
    medical_merchant_tags: List[str] = field(
        default_factory=lambda: ["healthcare", "hospital", "pharmacy", "medical", "diagnostic"]
    )
    salary_credit_day_field: str = "salary_credit_day"

    # Feature availability flags (not all banks have all data sources)
    feature_availability: Dict[str, bool] = field(default_factory=lambda: {
        "app_events": False,
        "gst_data": False,
        "fd_rd_data": True,
        "insurance_data": True,
        "mutual_fund_data": True,
        "employer_data": True,
    })

    # Channel providers
    channel_providers: Dict[str, str] = field(default_factory=lambda: {
        "sms": "twilio",
        "whatsapp": "twilio",
        "email": "smtp",
        "push": "webhook",
        "rm_call": "internal",
        "collector": "internal",
    })

    # Risk thresholds (tunable per bank)
    risk_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "watch": 0.50,
        "critical": 0.70,
    })

    # Segment-specific threshold overrides
    segment_thresholds: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "gig_worker": {"watch": 0.45, "critical": 0.75},
        "agricultural": {"watch": 0.45, "critical": 0.75},
        "retiree": {"watch": 0.50, "critical": 0.72},
    })

    # Regulatory flags
    regulatory_flags: Dict[str, object] = field(default_factory=lambda: {
        "fca_fair_practice": True,
        "cooldown_hours": 24,
        "max_sms_per_day": 3,
        "max_rm_calls_per_week": 2,
        "data_retention_days": 365,
    })

    # Intervention cost config (for ROI calculation)
    intervention_costs: Dict[str, float] = field(default_factory=lambda: {
        "sms": 1.0,
        "email": 0.5,
        "whatsapp": 2.0,
        "app_push": 0.1,
        "rm_call": 200.0,
        "collector_assignment": 2000.0,
    })

    # Category mapping: bank-specific MCC codes → engine canonical categories
    category_mapping: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "BankProfile":
        """Load bank profile from a YAML configuration file."""
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        profile = cls()
        for key, value in data.items():
            if hasattr(profile, key) and value is not None:
                setattr(profile, key, value)
        return profile


class BankProfileLoader:
    """Loads and caches the active bank profile."""

    _active_profile: Optional[BankProfile] = None

    @classmethod
    def get_active_profile(cls) -> BankProfile:
        """Get the active bank profile based on BANK_ID environment variable."""
        if cls._active_profile is not None:
            return cls._active_profile

        bank_id = os.getenv("BANK_ID", "BARCLAYS_IN")
        yaml_file = CONFIG_DIR / f"{bank_id.lower()}.yaml"

        if yaml_file.exists():
            cls._active_profile = BankProfile.from_yaml(str(yaml_file))
            logger.info(f"[BankConfig] Loaded profile for {cls._active_profile.display_name} ({bank_id})")
        else:
            logger.warning(f"[BankConfig] No config found for {bank_id}, using defaults")
            cls._active_profile = BankProfile()
            cls._active_profile.bank_id = bank_id

        return cls._active_profile

    @classmethod
    def reload(cls):
        """Force reload of the active profile."""
        cls._active_profile = None
        return cls.get_active_profile()
