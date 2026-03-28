# pyre-ignore-all-errors
"""
Main Data Generator Entry Point
Generates customers, transactions, account balances, and publishes everything
to PostgreSQL and Kafka.
"""
import sys
import os
import json
import time
from datetime import datetime, timedelta

from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config.settings import DataGenConfig, KafkaConfig, RedisConfig, PostgresConfig
from data_generator.customer_generator import generate_customers, save_customers_to_db
from data_generator.transaction_generator import (
    generate_transactions_for_customer,
    generate_account_balances,
    save_transactions_to_db,
    save_balances_to_db,
    save_payment_events_to_db,
    publish_transactions_to_kafka,
    publish_balances_to_kafka,
)


def seed_merchant_risk_to_redis():
    """Load merchant risk scores from PostgreSQL into Redis for fast lookups."""
    import redis as redis_lib
    import psycopg2

    r = redis_lib.Redis(
        host=RedisConfig.HOST, port=RedisConfig.PORT, db=RedisConfig.DB,
        decode_responses=True,
    )

    conn = psycopg2.connect(
        host=PostgresConfig.HOST, port=PostgresConfig.PORT,
        user=PostgresConfig.USER, password=PostgresConfig.PASSWORD,
        dbname=PostgresConfig.DB,
    )
    cursor = conn.cursor()
    cursor.execute("SELECT merchant_category, risk_score, risk_category FROM merchant_risk_scores")
    rows = cursor.fetchall()

    for category, score, risk_cat in rows:
        r.hset(f"merchant_risk:{category}", mapping={
            "risk_score": str(score),
            "risk_category": risk_cat,
        })

    cursor.close()
    conn.close()
    print(f"[DataGen] Seeded {len(rows)} merchant risk scores to Redis")


def run_full_generation():
    """Run the complete data generation pipeline."""
    print("=" * 70)
    print("Pre-Delinquency Intervention Engine - Data Generation")
    print("=" * 70)

    # Step 1: Generate customers
    print(f"\n[1/5] Generating {DataGenConfig.NUM_CUSTOMERS} customer profiles...")
    customers = generate_customers()
    stressed_count = sum(1 for c in customers if c.get("is_stressed"))
    print(f"  -> {len(customers)} customers generated ({stressed_count} stressed / {len(customers) - stressed_count} normal)")

    # Step 2: Save customers to PostgreSQL
    print("\n[2/5] Saving customers to PostgreSQL...")
    save_customers_to_db(customers)

    # Step 3: Seed merchant risk scores to Redis
    print("\n[3/5] Seeding merchant risk scores to Redis...")
    seed_merchant_risk_to_redis()

    # Step 4: Generate transactions and balances
    print(f"\n[4/5] Generating {DataGenConfig.TRANSACTION_MONTHS} months of transaction history...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=DataGenConfig.TRANSACTION_MONTHS * 30)

    all_transactions = []
    all_balances = []
    all_payment_events = []

    for customer in tqdm(customers, desc="  Generating transactions"):
        txns, payment_events = generate_transactions_for_customer(
            customer, start_date, end_date,
            total_months=DataGenConfig.TRANSACTION_MONTHS,
        )
        balances = generate_account_balances(customer, txns)
        all_transactions.extend(txns)
        all_balances.extend(balances)
        all_payment_events.extend(payment_events)

    delinquent_customers = len(set(pe["customer_id"] for pe in all_payment_events))
    print(f"  -> {len(all_transactions)} total transactions generated")
    print(f"  -> {len(all_balances)} balance snapshots generated")
    print(f"  -> {len(all_payment_events)} missed payment events ({delinquent_customers} customers)")
    print(f"  -> Delinquency rate: {delinquent_customers/len(customers)*100:.1f}%")

    # Step 5: Save to PostgreSQL
    print("\n[5/5] Saving transactions, balances, and payment events to PostgreSQL...")
    save_transactions_to_db(all_transactions)
    save_balances_to_db(all_balances)
    save_payment_events_to_db(all_payment_events)

    # Step 6: Publish to Kafka (recent transactions only - last 30 days)
    print("\n[Bonus] Publishing recent transactions to Kafka...")
    cutoff = (end_date - timedelta(days=30)).isoformat()
    recent_txns = [t for t in all_transactions if t["timestamp"] >= cutoff]
    recent_balances = [b for b in all_balances if b["timestamp"] >= cutoff]

    try:
        publish_transactions_to_kafka(recent_txns)
        publish_balances_to_kafka(recent_balances)
    except Exception as e:
        print(f"  WARNING: Kafka publish failed ({e}). Data is still in PostgreSQL.")

    # Summary
    print("\n" + "=" * 70)
    print("DATA GENERATION COMPLETE")
    print("=" * 70)
    print(f"  Customers:     {len(customers)}")
    print(f"  Transactions:  {len(all_transactions)}")
    print(f"  Balances:      {len(all_balances)}")
    print(f"  Stressed:      {stressed_count} ({stressed_count/len(customers)*100:.1f}%)")
    print(f"  Delinquent:    {delinquent_customers} ({delinquent_customers/len(customers)*100:.1f}%)")
    print(f"  Payment Events:{len(all_payment_events)}")
    print(f"  Date Range:    {start_date.strftime('%Y-%m-%d')} -> {end_date.strftime('%Y-%m-%d')}")
    print(f"  Kafka Topics:  {KafkaConfig.TOPIC_TRANSACTIONS}, {KafkaConfig.TOPIC_ACCOUNT_UPDATES}")
    print("=" * 70)

    return customers, all_transactions, all_balances, all_payment_events


if __name__ == "__main__":
    run_full_generation()

