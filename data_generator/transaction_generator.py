"""
Transaction Generator
Generates 6 months of realistic transaction history for all customers,
including stress patterns for at-risk customers.
Publishes to Kafka and stores in PostgreSQL.
"""
import random
import uuid
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional

from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable
import psycopg2
from psycopg2.extras import execute_values
from tqdm import tqdm

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config.settings import PostgresConfig, KafkaConfig, DataGenConfig

# ─────────────────────────────────────────────
# Transaction type definitions with distributions
# ─────────────────────────────────────────────
MERCHANT_CATEGORIES = {
    # category: (avg_amount_low, avg_amount_high, direction)
    "grocery": (200, 3000, "debit"),
    "dining": (150, 2500, "debit"),
    "entertainment": (100, 2000, "debit"),
    "utility": (500, 5000, "debit"),
    "rent": (5000, 50000, "debit"),
    "clothing": (500, 8000, "debit"),
    "electronics": (1000, 50000, "debit"),
    "healthcare": (200, 10000, "debit"),
    "education": (500, 20000, "debit"),
    "travel": (500, 30000, "debit"),
    "insurance": (1000, 15000, "debit"),
    "transfer": (500, 20000, "debit"),
    "lending_app": (1000, 50000, "debit"),
    "gambling": (500, 10000, "debit"),
    "lottery": (200, 5000, "debit"),
    "luxury_goods": (5000, 100000, "debit"),
    "crypto_exchange": (1000, 30000, "debit"),
    "payday_lender": (2000, 30000, "debit"),
    "cash_advance": (1000, 20000, "debit"),
}

# Normal daily transaction probabilities by category
NORMAL_TXN_PROBS = {
    "grocery": 0.40,
    "dining": 0.20,
    "entertainment": 0.10,
    "utility": 0.03,       # ~once a month
    "rent": 0.01,           # once a month
    "clothing": 0.05,
    "electronics": 0.02,
    "healthcare": 0.02,
    "education": 0.01,
    "travel": 0.02,
    "insurance": 0.01,
    "transfer": 0.15,
}

# Stressed customer additional transaction probabilities
STRESSED_TXN_PROBS = {
    "lending_app": 0.08,
    "payday_lender": 0.03,
    "cash_advance": 0.04,
    "gambling": 0.03,
    "lottery": 0.04,
    "crypto_exchange": 0.02,
}

TXN_CHANNELS = {
    "grocery": ["upi", "upi", "upi", "netbanking"],
    "dining": ["upi", "upi", "upi"],
    "entertainment": ["upi", "netbanking"],
    "utility": ["netbanking", "auto", "upi"],
    "rent": ["netbanking", "upi", "auto"],
    "clothing": ["upi", "netbanking"],
    "electronics": ["upi", "netbanking"],
    "healthcare": ["upi", "netbanking"],
    "education": ["netbanking"],
    "travel": ["netbanking", "upi"],
    "insurance": ["auto", "netbanking"],
    "transfer": ["upi", "netbanking"],
    "lending_app": ["upi", "netbanking"],
    "gambling": ["upi"],
    "lottery": ["upi", "netbanking"],
    "luxury_goods": ["netbanking", "upi"],
    "crypto_exchange": ["netbanking", "upi"],
    "payday_lender": ["netbanking"],
    "cash_advance": ["netbanking", "upi"],
}


def _generate_atm_withdrawal(customer: Dict, is_stressed: bool, day_of_month: int) -> Optional[Dict]:
    """Generate ATM withdrawal with stress-aware patterns."""
    # Normal: 2-3 ATM withdrawals per month
    # Stressed: 6-10 ATM withdrawals per month, especially near month-end
    if is_stressed:
        prob = 0.25 if day_of_month > 20 else 0.15
        amount_range = (2000, 10000)
    else:
        prob = 0.08
        amount_range = (1000, 5000)

    if random.random() < prob:
        return {
            "txn_type": "atm",
            "merchant_category": "transfer",
            "amount": round(random.uniform(*amount_range), 2),
            "direction": "debit",
            "channel": "atm",
            "status": "success",
        }
    return None


def _generate_salary_credit(customer: Dict, current_date: datetime, is_stressed: bool) -> Optional[Dict]:
    """Generate salary credit on the expected day (with delays for stressed customers)."""
    expected_day = customer["salary_credit_day"]
    day_of_month = current_date.day

    if is_stressed:
        # Stressed customers get salary late (1-10 days late)
        actual_day = expected_day + random.randint(1, 10)
        if actual_day > 28:
            actual_day = actual_day % 28 + 1
    else:
        # Normal customers: salary on time or 0-1 day late
        actual_day = expected_day + random.choice([0, 0, 0, 1])
        if actual_day > 28:
            actual_day = actual_day % 28 + 1

    if day_of_month == actual_day:
        # Salary varies slightly each month
        base_salary = customer["monthly_salary"]
        variation = random.uniform(-0.02, 0.05)  # -2% to +5% variation
        amount = round(base_salary * (1 + variation), 2)
        return {
            "txn_type": "salary_credit",
            "merchant_category": "salary",
            "amount": amount,
            "direction": "credit",
            "channel": "netbanking",
            "status": "success",
        }
    return None


def _generate_emi_payment(customer: Dict, current_date: datetime, is_stressed: bool) -> Optional[Dict]:
    """Generate EMI auto-debit payments."""
    # EMI due dates: typically 5th or 15th of month
    if current_date.day not in [5, 15]:
        return None

    has_loan = ("personal_loan" in customer["product_holdings"] or
                "home_loan" in customer["product_holdings"])
    if not has_loan:
        return None

    if "home_loan" in customer["product_holdings"]:
        emi_amount = round(customer["monthly_salary"] * random.uniform(0.25, 0.40), 2)
    else:
        emi_amount = round(customer["monthly_salary"] * random.uniform(0.10, 0.20), 2)

    # Stressed customers: EMIs sometimes fail
    if is_stressed:
        status = random.choices(["success", "failed"], weights=[60, 40], k=1)[0]
    else:
        status = random.choices(["success", "failed"], weights=[97, 3], k=1)[0]

    return {
        "txn_type": "emi" if status == "success" else "auto_debit",
        "merchant_category": "insurance" if "home_loan" in customer["product_holdings"] else "transfer",
        "amount": emi_amount,
        "direction": "debit",
        "channel": "auto",
        "status": status,
    }


def _apply_stress_progression(customer: Dict, months_in: int, total_months: int) -> float:
    """
    Stress intensifies over time for stressed customers.
    Returns a stress multiplier (1.0 = normal, higher = more stressed).
    """
    if not customer.get("is_stressed", False):
        return 1.0

    # Stress ramps up in the last 2-3 months
    if months_in < total_months - 3:
        return 1.0  # Early months are normal
    elif months_in < total_months - 1:
        return 1.0 + (months_in - (total_months - 3)) * 0.3  # Gradual increase
    else:
        return 2.0  # Last month: peak stress


def generate_transactions_for_customer(
    customer: Dict,
    start_date: datetime,
    end_date: datetime,
    total_months: int = 6,
) -> List[Dict]:
    """Generate all transactions for a single customer over the date range."""
    transactions = []
    is_stressed = customer.get("is_stressed", False)
    current_date = start_date

    while current_date <= end_date:
        months_in = (current_date.year - start_date.year) * 12 + (current_date.month - start_date.month)
        stress_mult = _apply_stress_progression(customer, months_in, total_months)

        # 1. Salary credit
        salary_txn = _generate_salary_credit(customer, current_date, is_stressed)
        if salary_txn:
            salary_txn["customer_id"] = customer["customer_id"]
            salary_txn["txn_id"] = f"TXN_{uuid.uuid4().hex[:16].upper()}"
            salary_txn["timestamp"] = current_date.replace(
                hour=random.randint(8, 12),
                minute=random.randint(0, 59),
            ).isoformat()
            transactions.append(salary_txn)

        # 2. EMI payment
        emi_txn = _generate_emi_payment(customer, current_date, is_stressed)
        if emi_txn:
            emi_txn["customer_id"] = customer["customer_id"]
            emi_txn["txn_id"] = f"TXN_{uuid.uuid4().hex[:16].upper()}"
            emi_txn["timestamp"] = current_date.replace(
                hour=random.randint(6, 8),
                minute=random.randint(0, 59),
            ).isoformat()
            transactions.append(emi_txn)

        # 3. ATM withdrawal
        atm_txn = _generate_atm_withdrawal(customer, is_stressed, current_date.day)
        if atm_txn:
            atm_txn["customer_id"] = customer["customer_id"]
            atm_txn["txn_id"] = f"TXN_{uuid.uuid4().hex[:16].upper()}"
            atm_txn["timestamp"] = current_date.replace(
                hour=random.randint(9, 21),
                minute=random.randint(0, 59),
            ).isoformat()
            transactions.append(atm_txn)

        # 4. Normal spending transactions
        for category, prob in NORMAL_TXN_PROBS.items():
            # Scale amount by income
            income_scale = customer["monthly_salary"] / 50000  # Normalize around Rs.50k
            if random.random() < prob:
                low, high, direction = MERCHANT_CATEGORIES[category]
                amount = round(random.uniform(low, high) * income_scale, 2)

                # Utility and rent: monthly, roughly fixed amount
                if category in ("utility", "rent") and current_date.day != random.randint(1, 5):
                    continue

                txn = {
                    "txn_id": f"TXN_{uuid.uuid4().hex[:16].upper()}",
                    "customer_id": customer["customer_id"],
                    "txn_type": "upi" if "upi" in TXN_CHANNELS.get(category, ["upi"]) else "bill_payment",
                    "merchant_category": category,
                    "amount": amount,
                    "direction": direction,
                    "channel": random.choice(TXN_CHANNELS.get(category, ["upi"])),
                    "status": "success",
                    "timestamp": current_date.replace(
                        hour=random.randint(7, 22),
                        minute=random.randint(0, 59),
                        second=random.randint(0, 59),
                    ).isoformat(),
                }
                transactions.append(txn)

        # 5. Stressed customer: risky transactions (increase with stress_mult)
        if is_stressed and stress_mult > 1.0:
            for category, prob in STRESSED_TXN_PROBS.items():
                adjusted_prob = prob * stress_mult
                if random.random() < adjusted_prob:
                    low, high, direction = MERCHANT_CATEGORIES[category]
                    income_scale = customer["monthly_salary"] / 50000
                    amount = round(random.uniform(low, high) * income_scale * stress_mult, 2)

                    txn = {
                        "txn_id": f"TXN_{uuid.uuid4().hex[:16].upper()}",
                        "customer_id": customer["customer_id"],
                        "txn_type": "upi",
                        "merchant_category": category,
                        "amount": amount,
                        "direction": direction,
                        "channel": random.choice(TXN_CHANNELS.get(category, ["upi"])),
                        "status": "success",
                        "timestamp": current_date.replace(
                            hour=random.randint(7, 23),
                            minute=random.randint(0, 59),
                            second=random.randint(0, 59),
                        ).isoformat(),
                    }
                    transactions.append(txn)

        current_date += timedelta(days=1)

    return transactions


def generate_account_balances(customer: Dict, transactions: List[Dict]) -> List[Dict]:
    """Generate daily account balance snapshots from transaction history."""
    initial_balance = round(customer["monthly_salary"] * random.uniform(1.5, 6.0), 2)
    initial_savings = round(customer["monthly_salary"] * random.uniform(2.0, 12.0), 2)

    balances = []
    current_balance = initial_balance
    current_savings = initial_savings

    # Sort transactions by timestamp
    sorted_txns = sorted(transactions, key=lambda t: t["timestamp"])

    # Group by date
    txn_by_date = {}
    for txn in sorted_txns:
        date_key = txn["timestamp"][:10]
        if date_key not in txn_by_date:
            txn_by_date[date_key] = []
        txn_by_date[date_key].append(txn)

    for date_key in sorted(txn_by_date.keys()):
        for txn in txn_by_date[date_key]:
            if txn["status"] != "success":
                continue
            if txn["direction"] == "credit":
                current_balance += txn["amount"]
            else:
                current_balance -= txn["amount"]
                # Also draw from savings if balance is low
                if current_balance < 0:
                    draw = min(abs(current_balance), current_savings * 0.1)
                    current_savings -= draw
                    current_balance += draw

        balances.append({
            "customer_id": customer["customer_id"],
            "balance": round(max(current_balance, 0), 2),
            "savings_balance": round(max(current_savings, 0), 2),
            "timestamp": f"{date_key}T23:59:59",
        })

    return balances


def save_transactions_to_db(transactions: List[Dict]):
    """Save transactions to PostgreSQL."""
    conn = psycopg2.connect(
        host=PostgresConfig.HOST, port=PostgresConfig.PORT,
        user=PostgresConfig.USER, password=PostgresConfig.PASSWORD,
        dbname=PostgresConfig.DB,
    )
    cursor = conn.cursor()

    values = [
        (
            t["txn_id"], t["customer_id"], t["txn_type"],
            t["merchant_category"], None, t["amount"],
            t["direction"], t["channel"], t["status"],
            t["timestamp"],
        )
        for t in transactions
    ]

    # Batch insert in chunks
    chunk_size = 5000
    for i in range(0, len(values), chunk_size):
        chunk = values[i:i + chunk_size]
        execute_values(
            cursor,
            """INSERT INTO transactions (
                txn_id, customer_id, txn_type, merchant_category,
                merchant_id, amount, direction, channel, status, timestamp
            ) VALUES %s ON CONFLICT (txn_id) DO NOTHING""",
            chunk,
        )
        conn.commit()

    cursor.close()
    conn.close()
    print(f"[TransactionGenerator] Saved {len(transactions)} transactions to PostgreSQL")


def save_balances_to_db(balances: List[Dict]):
    """Save account balances to PostgreSQL."""
    conn = psycopg2.connect(
        host=PostgresConfig.HOST, port=PostgresConfig.PORT,
        user=PostgresConfig.USER, password=PostgresConfig.PASSWORD,
        dbname=PostgresConfig.DB,
    )
    cursor = conn.cursor()

    values = [
        (b["customer_id"], b["balance"], b["savings_balance"], b["timestamp"])
        for b in balances
    ]

    chunk_size = 5000
    for i in range(0, len(values), chunk_size):
        chunk = values[i:i + chunk_size]
        execute_values(
            cursor,
            """INSERT INTO account_balances (customer_id, balance, savings_balance, timestamp)
            VALUES %s""",
            chunk,
        )
        conn.commit()

    cursor.close()
    conn.close()
    print(f"[TransactionGenerator] Saved {len(balances)} balance snapshots to PostgreSQL")


def publish_transactions_to_kafka(transactions: List[Dict], producer: KafkaProducer = None):
    """Publish transactions to Kafka topic."""
    own_producer = False
    if producer is None:
        try:
            producer = KafkaProducer(
                bootstrap_servers=KafkaConfig.BOOTSTRAP_SERVERS,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if k else None,
            )
            own_producer = True
        except NoBrokersAvailable:
            print("[TransactionGenerator] WARNING: Kafka not available, skipping publish")
            return

    for txn in transactions:
        producer.send(
            KafkaConfig.TOPIC_TRANSACTIONS,
            key=txn["customer_id"],
            value=txn,
        )

    producer.flush()
    if own_producer:
        producer.close()
    print(f"[TransactionGenerator] Published {len(transactions)} transactions to Kafka")


def publish_balances_to_kafka(balances: List[Dict], producer: KafkaProducer = None):
    """Publish account balance updates to Kafka."""
    own_producer = False
    if producer is None:
        try:
            producer = KafkaProducer(
                bootstrap_servers=KafkaConfig.BOOTSTRAP_SERVERS,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if k else None,
            )
            own_producer = True
        except NoBrokersAvailable:
            print("[TransactionGenerator] WARNING: Kafka not available, skipping publish")
            return

    for balance in balances:
        producer.send(
            KafkaConfig.TOPIC_ACCOUNT_UPDATES,
            key=balance["customer_id"],
            value=balance,
        )

    producer.flush()
    if own_producer:
        producer.close()
    print(f"[TransactionGenerator] Published {len(balances)} balance updates to Kafka")

