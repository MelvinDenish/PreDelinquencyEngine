"""
Live Demo Simulation Script
Feeds realistic transactions in real-time to demonstrate the full pipeline:
  - Transactions appear on Kafka
  - Stream processor computes features
  - Scoring service evaluates risk
  - Intervention engine fires when thresholds breach
  - GenAI generates personalized messages

Usage:
  1. Start Docker services:   docker-compose up -d
  2. Start scoring service:   python main.py scoring-service
  3. Start stream processor:  python main.py stream-process
  4. Run this demo:           python demo/live_simulation.py
"""
import os
import sys
import time
import json
import random
import uuid
import requests
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config.settings import KafkaConfig
from intervention.genai_messages import generate_message_groq

# ─────────────────────────────────────────────
# Demo customers
# ─────────────────────────────────────────────
DEMO_CUSTOMERS = [
    {
        "customer_id": "DEMO_HEALTHY_001",
        "first_name": "Arjun", "last_name": "Nair",
        "city": "Bangalore", "region": "South",
        "income_bracket": "upper_middle",
        "monthly_salary": 120000,
        "tenure_months": 48,
        "preferred_channel": "app",
        "is_stressed": False,
    },
    {
        "customer_id": "DEMO_HEALTHY_002",
        "first_name": "Meera", "last_name": "Iyer",
        "city": "Chennai", "region": "South",
        "income_bracket": "middle",
        "monthly_salary": 65000,
        "tenure_months": 36,
        "preferred_channel": "sms",
        "is_stressed": False,
    },
    {
        "customer_id": "DEMO_STRESSED_001",
        "first_name": "Rahul", "last_name": "Verma",
        "city": "Delhi", "region": "North",
        "income_bracket": "lower_middle",
        "monthly_salary": 40000,
        "tenure_months": 24,
        "preferred_channel": "sms",
        "is_stressed": True,
    },
    {
        "customer_id": "DEMO_STRESSED_002",
        "first_name": "Priya", "last_name": "Sharma",
        "city": "Mumbai", "region": "West",
        "income_bracket": "middle",
        "monthly_salary": 55000,
        "tenure_months": 60,
        "preferred_channel": "app",
        "is_stressed": True,
    },
    {
        "customer_id": "DEMO_CRITICAL_001",
        "first_name": "Vikram", "last_name": "Singh",
        "city": "Jaipur", "region": "North",
        "income_bracket": "low",
        "monthly_salary": 22000,
        "tenure_months": 12,
        "preferred_channel": "sms",
        "is_stressed": True,
    },
]

# ─────────────────────────────────────────────
# Transaction patterns
# ─────────────────────────────────────────────
HEALTHY_PATTERNS = [
    {"merchant_category": "grocery", "amount_range": (200, 2000), "txn_type": "upi"},
    {"merchant_category": "dining", "amount_range": (150, 1500), "txn_type": "upi"},
    {"merchant_category": "utility", "amount_range": (500, 3000), "txn_type": "upi"},
]

STRESSED_PATTERNS = [
    {"merchant_category": "lending_app", "amount_range": (5000, 25000), "txn_type": "upi"},
    {"merchant_category": "gambling", "amount_range": (1000, 8000), "txn_type": "upi"},
    {"merchant_category": "lottery", "amount_range": (200, 3000), "txn_type": "upi"},
    {"merchant_category": "payday_lender", "amount_range": (3000, 15000), "txn_type": "upi"},
    {"merchant_category": "cash_advance", "amount_range": (2000, 10000), "txn_type": "upi"},
]

CRITICAL_PATTERNS = STRESSED_PATTERNS + [
    {"merchant_category": "crypto_exchange", "amount_range": (5000, 20000), "txn_type": "upi"},
]


def create_transaction(customer, pattern):
    """Create a single transaction event."""
    amount = round(random.uniform(*pattern["amount_range"]), 2)
    return {
        "txn_id": f"TXN_DEMO_{uuid.uuid4().hex[:12].upper()}",
        "customer_id": customer["customer_id"],
        "txn_type": pattern["txn_type"],
        "merchant_category": pattern["merchant_category"],
        "amount": amount,
        "direction": "debit",
        "channel": random.choice(["upi", "netbanking", "mobile"]),
        "status": "success",
        "timestamp": datetime.now().isoformat(),
    }


def create_failed_emi(customer):
    """Create a failed EMI auto-debit event."""
    return {
        "txn_id": f"TXN_DEMO_{uuid.uuid4().hex[:12].upper()}",
        "customer_id": customer["customer_id"],
        "txn_type": "auto_debit",
        "merchant_category": "transfer",
        "amount": round(customer["monthly_salary"] * 0.15, 2),
        "direction": "debit",
        "channel": "auto",
        "status": "failed",  # ← FAILED!
        "timestamp": datetime.now().isoformat(),
    }


def publish_to_kafka(transaction):
    """Publish transaction to Kafka via the ingestion module."""
    try:
        from kafka import KafkaProducer
        producer = KafkaProducer(
            bootstrap_servers=KafkaConfig.BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k else None,
        )
        producer.send(
            KafkaConfig.TOPIC_TRANSACTIONS,
            key=transaction["customer_id"],
            value=transaction,
        )
        producer.flush()
        producer.close()
        return True
    except Exception as e:
        print(f"  ⚠ Kafka unavailable: {e}")
        return False


def try_score_customer(customer_id):
    """Try to score a customer via the scoring service."""
    try:
        resp = requests.post(
            "http://localhost:8000/score",
            json={"customer_id": customer_id},
            timeout=5,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data
    except Exception:
        pass
    return None


def print_banner():
    """Print demo header."""
    print("\n" + "═" * 70)
    print("  🏦  PRE-DELINQUENCY INTERVENTION ENGINE — LIVE DEMO")
    print("  📊  Real-time Transaction Simulation")
    print("═" * 70)
    print()
    print("  Customers in demo:")
    for c in DEMO_CUSTOMERS:
        stress = "🔴 STRESSED" if c["is_stressed"] else "🟢 HEALTHY"
        print(f"    {c['customer_id']}: {c['first_name']} {c['last_name']} "
              f"({c['city']}) — {stress}")
    print()
    print("  Press Ctrl+C to stop the simulation")
    print("─" * 70)


def run_simulation():
    """Main simulation loop."""
    print_banner()

    round_num = 0
    while True:
        round_num += 1
        print(f"\n🔄 Round {round_num} — {datetime.now().strftime('%H:%M:%S')}")
        print("─" * 50)

        for customer in DEMO_CUSTOMERS:
            # Choose transaction pattern based on stress level
            if customer["customer_id"] == "DEMO_CRITICAL_001":
                patterns = CRITICAL_PATTERNS
                # Also inject failed EMI every 3rd round
                if round_num % 3 == 0:
                    failed = create_failed_emi(customer)
                    print(f"  ❌ {customer['first_name']}: "
                          f"FAILED auto-debit ₹{failed['amount']:,.0f}")
                    publish_to_kafka(failed)
            elif customer["is_stressed"]:
                patterns = STRESSED_PATTERNS if round_num > 2 else HEALTHY_PATTERNS
            else:
                patterns = HEALTHY_PATTERNS

            # Generate 1-3 transactions per customer per round
            num_txns = random.randint(1, 3 if customer["is_stressed"] else 2)
            for _ in range(num_txns):
                pattern = random.choice(patterns)
                txn = create_transaction(customer, pattern)

                emoji = "💳"
                if txn["merchant_category"] in ("lending_app", "payday_lender"):
                    emoji = "⚠️"
                elif txn["merchant_category"] in ("gambling", "lottery"):
                    emoji = "🎰"
                elif txn["merchant_category"] == "cash_advance":
                    emoji = "💸"

                print(f"  {emoji} {customer['first_name']}: "
                      f"{txn['merchant_category']} ₹{txn['amount']:,.0f}")
                publish_to_kafka(txn)

            # Try to score after transactions
            score_result = try_score_customer(customer["customer_id"])
            if score_result:
                risk_score = score_result.get("risk_score", 0)
                risk_tier = score_result.get("risk_tier", "unknown")
                tier_emoji = {"stable": "🟢", "watch": "🟡", "critical": "🔴"}.get(
                    risk_tier, "⚪")
                print(f"  {tier_emoji} Score: {risk_score:.3f} ({risk_tier})")

                # If critical, generate GenAI intervention message
                if risk_tier in ("watch", "critical"):
                    shap_drivers = score_result.get("shap_drivers", [])
                    message = generate_message_groq(
                        customer=customer,
                        intervention_type="wellness_checkin" if risk_tier == "watch"
                            else "escalation_call",
                        risk_score=risk_score,
                        shap_drivers=shap_drivers,
                        channel=customer["preferred_channel"],
                    )
                    print(f"  📱 GenAI Message → \"{message[:80]}...\"")

        # Wait between rounds
        delay = random.uniform(2, 5)
        print(f"\n⏳ Next round in {delay:.1f}s...")
        time.sleep(delay)


if __name__ == "__main__":
    try:
        run_simulation()
    except KeyboardInterrupt:
        print("\n\n✅ Demo simulation stopped.")
        print("   Check the dashboard at http://localhost:8050 for results.")
