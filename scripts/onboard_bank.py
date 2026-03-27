"""
Bank Onboarding CLI
Validates a bank's YAML configuration, checks schema compatibility,
and runs a test scoring on synthetic customers.

Usage:
    python scripts/onboard_bank.py --bank-id HDFC --validate-schema --dry-run
"""
import argparse
import sys
import os
import yaml
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.bank_config import BankProfile, BankProfileLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REQUIRED_FIELDS = [
    "bank_id", "display_name", "currency", "locale",
    "income_credit_tags", "discretionary_categories",
    "lending_app_merchants", "risk_thresholds",
    "channel_providers",
]

OPTIONAL_FIELDS = [
    "emi_merchant_tags", "distress_merchant_tags", "medical_merchant_tags",
    "salary_credit_day_field", "feature_availability", "segment_thresholds",
    "regulatory_flags", "intervention_costs", "category_mapping",
]


def validate_yaml(yaml_path: str) -> dict:
    """Validate that a bank YAML file has all required fields."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    report = {"errors": [], "warnings": [], "coverage": {}}

    # Check required fields
    for field in REQUIRED_FIELDS:
        if field not in data or not data[field]:
            report["errors"].append(f"MISSING required field: '{field}'")
        else:
            report["coverage"][field] = "✅"

    # Check optional fields
    for field in OPTIONAL_FIELDS:
        if field not in data or not data[field]:
            report["warnings"].append(f"OPTIONAL field not set: '{field}' (defaults will be used)")
        else:
            report["coverage"][field] = "✅"

    # Validate risk thresholds
    thresholds = data.get("risk_thresholds", {})
    if thresholds:
        watch = thresholds.get("watch", 0)
        critical = thresholds.get("critical", 0)
        if watch >= critical:
            report["errors"].append(f"watch threshold ({watch}) must be < critical ({critical})")

    # Validate channel providers
    channels = data.get("channel_providers", {})
    supported_providers = {
        "sms": ["twilio", "gupshup", "msg91"],
        "whatsapp": ["twilio", "gupshup"],
        "email": ["smtp", "sendgrid"],
        "push": ["webhook", "firebase"],
        "rm_call": ["internal"],
        "collector": ["internal"],
    }
    for channel, provider in channels.items():
        if channel in supported_providers and provider not in supported_providers[channel]:
            report["warnings"].append(
                f"Channel '{channel}' provider '{provider}' not in supported list: "
                f"{supported_providers[channel]}"
            )

    return report


def run_test_scoring(bank_profile: BankProfile) -> dict:
    """Run a test scoring with synthetic data to verify config works."""
    from feature_store.feature_adapter import FeatureAdapter
    adapter = FeatureAdapter(bank_profile)

    # Test taxonomy mapping
    test_cases = [
        ("dining", True, "discretionary"),
        ("grocery", False, "discretionary"),
        ("healthcare", False, "discretionary"),
        ("salary_credit", True, "income"),
    ]

    results = []
    for category, expected_disc, test_type in test_cases:
        if test_type == "discretionary":
            actual = adapter.is_discretionary(category)
        else:
            actual = adapter.is_income_credit(category)
        status = "✅" if actual == expected_disc else "❌"
        results.append(f"  {status} {test_type}('{category}') = {actual} (expected {expected_disc})")

    # Test thresholds per segment
    for segment in ["salaried", "gig_worker", "agricultural", "retiree"]:
        thresholds = adapter.get_risk_thresholds(segment)
        results.append(f"  ℹ️  {segment}: watch={thresholds['watch']}, critical={thresholds['critical']}")

    return {"test_results": results}


def main():
    parser = argparse.ArgumentParser(description="PDI Engine — Bank Onboarding Tool")
    parser.add_argument("--bank-id", required=True, help="Bank identifier (e.g., HDFC, SBI)")
    parser.add_argument("--config", help="Path to bank YAML config file")
    parser.add_argument("--validate-schema", action="store_true", help="Validate YAML schema")
    parser.add_argument("--dry-run", action="store_true", help="Run test scoring without DB")

    args = parser.parse_args()

    config_path = args.config
    if not config_path:
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "config", "banks", f"{args.bank_id.lower()}.yaml"
        )

    print("=" * 60)
    print(f"PDI Engine — Bank Onboarding: {args.bank_id}")
    print("=" * 60)

    if not os.path.exists(config_path):
        print(f"\n❌ Config file not found: {config_path}")
        print(f"   Copy config/banks/generic_template.yaml to config/banks/{args.bank_id.lower()}.yaml")
        sys.exit(1)

    # Step 1: Validate YAML
    if args.validate_schema:
        print("\n[1] Validating YAML schema...")
        report = validate_yaml(config_path)

        if report["errors"]:
            print("  ❌ ERRORS:")
            for err in report["errors"]:
                print(f"     {err}")

        if report["warnings"]:
            print("  ⚠️  WARNINGS:")
            for warn in report["warnings"]:
                print(f"     {warn}")

        covered = sum(1 for v in report["coverage"].values() if v == "✅")
        total = len(REQUIRED_FIELDS) + len(OPTIONAL_FIELDS)
        print(f"\n  Coverage: {covered}/{total} fields configured")

        if report["errors"]:
            print("\n❌ Validation FAILED — fix errors above before proceeding.")
            sys.exit(1)
        else:
            print("\n✅ Validation PASSED")

    # Step 2: Load profile
    print(f"\n[2] Loading bank profile from {config_path}...")
    profile = BankProfile.from_yaml(config_path)
    print(f"  Bank: {profile.display_name} ({profile.bank_id})")
    print(f"  Currency: {profile.currency}, Locale: {profile.locale}")
    print(f"  Income tags: {', '.join(profile.income_credit_tags)}")
    print(f"  Channels: {', '.join(f'{k}={v}' for k, v in profile.channel_providers.items())}")

    # Step 3: Dry-run test scoring
    if args.dry_run:
        print("\n[3] Running test scoring (dry-run)...")
        results = run_test_scoring(profile)
        for line in results["test_results"]:
            print(line)

    print("\n" + "=" * 60)
    print("✅ Onboarding validation complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
