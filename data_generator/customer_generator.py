"""
Customer Profile Generator
Generates realistic customer profiles with demographics, salary patterns,
product holdings, and behavioral tendencies.
"""
import random
import uuid
from datetime import datetime, timedelta
from typing import List, Dict

from faker import Faker
import psycopg2
from psycopg2.extras import execute_values

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config.settings import PostgresConfig, DataGenConfig

fake = Faker('en_IN')  # Indian locale for realistic Indian bank data

# ─────────────────────────────────────────────
# Constants for realistic distributions
# ─────────────────────────────────────────────
REGIONS = ["North", "South", "East", "West", "Central"]
REGION_CITIES = {
    "North": ["Delhi", "Noida", "Gurgaon", "Jaipur", "Lucknow", "Chandigarh"],
    "South": ["Chennai", "Bangalore", "Hyderabad", "Kochi", "Coimbatore", "Mysore"],
    "East": ["Kolkata", "Bhubaneswar", "Patna", "Guwahati", "Ranchi"],
    "West": ["Mumbai", "Pune", "Ahmedabad", "Surat", "Nagpur", "Goa"],
    "Central": ["Bhopal", "Indore", "Raipur", "Jabalpur", "Nagpur"],
}
STATES_BY_CITY = {
    "Delhi": "Delhi", "Noida": "Uttar Pradesh", "Gurgaon": "Haryana",
    "Jaipur": "Rajasthan", "Lucknow": "Uttar Pradesh", "Chandigarh": "Chandigarh",
    "Chennai": "Tamil Nadu", "Bangalore": "Karnataka", "Hyderabad": "Telangana",
    "Kochi": "Kerala", "Coimbatore": "Tamil Nadu", "Mysore": "Karnataka",
    "Kolkata": "West Bengal", "Bhubaneswar": "Odisha", "Patna": "Bihar",
    "Guwahati": "Assam", "Ranchi": "Jharkhand",
    "Mumbai": "Maharashtra", "Pune": "Maharashtra", "Ahmedabad": "Gujarat",
    "Surat": "Gujarat", "Nagpur": "Maharashtra", "Goa": "Goa",
    "Bhopal": "Madhya Pradesh", "Indore": "Madhya Pradesh",
    "Raipur": "Chhattisgarh", "Jabalpur": "Madhya Pradesh",
}

INCOME_BRACKETS = [
    ("low", 15000, 30000),
    ("lower_middle", 30000, 50000),
    ("middle", 50000, 80000),
    ("upper_middle", 80000, 150000),
    ("high", 150000, 500000),
]

PRODUCTS = ["savings_account", "credit_card", "personal_loan", "home_loan",
            "fixed_deposit", "recurring_deposit", "demat_account", "insurance"]

CHANNELS = ["sms", "email", "app", "rm_call"]


def _generate_salary_day() -> int:
    """Most salaries arrive between 25th-5th of month."""
    # Days 1-31: weight higher around 25th-5th (salary window)
    days = list(range(1, 32))
    # Days 1-5 (salary window end): high, Days 6-24: low, Days 25-31 (salary window start): high
    day_weights = [12, 10, 8, 5, 3] + [1]*19 + [5, 8, 10, 12, 15, 15, 12]  # 5+19+7 = 31
    return random.choices(days, weights=day_weights, k=1)[0]


def _generate_credit_score(income_bracket: str, is_stressed: bool) -> int:
    """Generate realistic credit score based on income and stress."""
    base_ranges = {
        "low": (580, 700),
        "lower_middle": (620, 740),
        "middle": (660, 780),
        "upper_middle": (700, 820),
        "high": (740, 850),
    }
    low, high = base_ranges.get(income_bracket, (650, 750))
    score = random.randint(low, high)
    if is_stressed:
        score -= random.randint(30, 100)  # Stressed customers have lower scores
    return max(300, min(900, score))


def _generate_products(income_bracket: str) -> List[str]:
    """Generate product holdings based on income."""
    products = ["savings_account"]  # Everyone has a savings account
    product_probs = {
        "credit_card": {"low": 0.2, "lower_middle": 0.4, "middle": 0.7, "upper_middle": 0.85, "high": 0.95},
        "personal_loan": {"low": 0.15, "lower_middle": 0.25, "middle": 0.3, "upper_middle": 0.35, "high": 0.2},
        "home_loan": {"low": 0.02, "lower_middle": 0.05, "middle": 0.15, "upper_middle": 0.3, "high": 0.4},
        "fixed_deposit": {"low": 0.1, "lower_middle": 0.2, "middle": 0.4, "upper_middle": 0.55, "high": 0.7},
        "recurring_deposit": {"low": 0.15, "lower_middle": 0.25, "middle": 0.3, "upper_middle": 0.3, "high": 0.2},
        "demat_account": {"low": 0.05, "lower_middle": 0.1, "middle": 0.3, "upper_middle": 0.5, "high": 0.7},
        "insurance": {"low": 0.1, "lower_middle": 0.2, "middle": 0.4, "upper_middle": 0.6, "high": 0.8},
    }
    for product, probs in product_probs.items():
        if random.random() < probs.get(income_bracket, 0.3):
            products.append(product)
    return products


def generate_customers(num_customers: int = None, stress_pct: float = None) -> List[Dict]:
    """Generate realistic customer profiles."""
    num_customers = num_customers or DataGenConfig.NUM_CUSTOMERS
    stress_pct = stress_pct or DataGenConfig.STRESS_CUSTOMER_PCT

    num_stressed = int(num_customers * stress_pct)
    stressed_indices = set(random.sample(range(num_customers), num_stressed))

    customers = []
    for i in range(num_customers):
        is_stressed = i in stressed_indices
        gender = random.choice(["M", "F"])
        region = random.choice(REGIONS)
        city = random.choice(REGION_CITIES[region])
        state = STATES_BY_CITY.get(city, "Unknown")

        # Income distribution: more people in middle brackets
        bracket_name, salary_low, salary_high = random.choices(
            INCOME_BRACKETS,
            weights=[15, 25, 30, 20, 10],
            k=1
        )[0]

        monthly_salary = round(random.uniform(salary_low, salary_high), 2)
        products = _generate_products(bracket_name)
        credit_score = _generate_credit_score(bracket_name, is_stressed)

        customer = {
            "customer_id": f"CUST_{uuid.uuid4().hex[:12].upper()}",
            "first_name": fake.first_name_male() if gender == "M" else fake.first_name_female(),
            "last_name": fake.last_name(),
            "age": random.randint(21, 65),
            "gender": gender,
            "city": city,
            "state": state,
            "region": region,
            "income_bracket": bracket_name,
            "monthly_salary": monthly_salary,
            "salary_credit_day": _generate_salary_day(),
            "tenure_months": random.randint(6, 240),
            "credit_score": credit_score,
            "product_holdings": products,
            "preferred_channel": random.choices(CHANNELS, weights=[35, 25, 30, 10], k=1)[0],
            "is_stressed": is_stressed,  # internal flag, not stored in DB
        }
        customers.append(customer)

    return customers


def save_customers_to_db(customers: List[Dict]):
    """Save generated customers to PostgreSQL."""
    conn = psycopg2.connect(
        host=PostgresConfig.HOST,
        port=PostgresConfig.PORT,
        user=PostgresConfig.USER,
        password=PostgresConfig.PASSWORD,
        dbname=PostgresConfig.DB,
    )
    cursor = conn.cursor()

    values = [
        (
            c["customer_id"], c["first_name"], c["last_name"], c["age"],
            c["gender"], c["city"], c["state"], c["region"],
            c["income_bracket"], c["monthly_salary"], c["salary_credit_day"],
            c["tenure_months"], c["credit_score"], c["product_holdings"],
            c["preferred_channel"],
        )
        for c in customers
    ]

    execute_values(
        cursor,
        """INSERT INTO customers (
            customer_id, first_name, last_name, age, gender, city, state, region,
            income_bracket, monthly_salary, salary_credit_day, tenure_months,
            credit_score, product_holdings, preferred_channel
        ) VALUES %s ON CONFLICT (customer_id) DO NOTHING""",
        values,
    )

    conn.commit()
    cursor.close()
    conn.close()
    print(f"[CustomerGenerator] Saved {len(customers)} customers to PostgreSQL")


if __name__ == "__main__":
    customers = generate_customers()
    print(f"Generated {len(customers)} customer profiles")
    print(f"Stressed customers: {sum(1 for c in customers if c['is_stressed'])}")
    # Preview
    for c in customers[:3]:
        print(f"  {c['customer_id']}: {c['first_name']} {c['last_name']}, "
              f"Rs.{c['monthly_salary']:,.0f}/mo, {c['income_bracket']}, "
              f"Score: {c['credit_score']}, Products: {c['product_holdings']}")

