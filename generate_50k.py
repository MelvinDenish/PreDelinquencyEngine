# pyre-ignore-all-errors
"""
Production-Grade Data Pipeline: 50K Customers, 5M+ Transactions
================================================================
Generates realistic Indian banking data with 8 subtle delinquency archetypes.

Key Design Decisions:
  - Realistic 5-7% delinquency rate (not 80/20)
  - 8 nuanced deterioration patterns (gradual decline, seasonal stress, etc.)
  - 12 months of transaction history for seasonality detection
  - Proper temporal ordering for contamination-free train/cal/test splits
  - Batch processing for memory efficiency (5K customers at a time)
  - Bulk SQL inserts via execute_values for speed

Usage:
  docker exec pdi-app python generate_50k.py
"""
import os, sys, gc, time, random, uuid, json, math
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional

import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from faker import Faker

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from config.settings import PostgresConfig

fake = Faker('en_IN')
random.seed(42)
np.random.seed(42)

# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════
NUM_CUSTOMERS    = int(os.getenv("NUM_CUSTOMERS", 50000))
TXN_MONTHS       = int(os.getenv("TRANSACTION_MONTHS", 12))
BATCH_SIZE       = int(os.getenv("DATAGEN_BATCH_SIZE", 5000))

# ══════════════════════════════════════════════════════════════
# REALISTIC INDIAN BANKING CONSTANTS
# ══════════════════════════════════════════════════════════════
REGIONS = ["North", "South", "East", "West", "Central"]
REGION_CITIES = {
    "North": ["Delhi", "Noida", "Gurgaon", "Jaipur", "Lucknow", "Chandigarh", "Amritsar"],
    "South": ["Chennai", "Bangalore", "Hyderabad", "Kochi", "Coimbatore", "Mysore", "Trivandrum"],
    "East": ["Kolkata", "Bhubaneswar", "Patna", "Guwahati", "Ranchi"],
    "West": ["Mumbai", "Pune", "Ahmedabad", "Surat", "Nagpur", "Goa"],
    "Central": ["Bhopal", "Indore", "Raipur", "Varanasi"],
}
STATES_BY_CITY = {
    "Delhi": "Delhi", "Noida": "UP", "Gurgaon": "Haryana", "Jaipur": "Rajasthan",
    "Lucknow": "UP", "Chandigarh": "Chandigarh", "Amritsar": "Punjab",
    "Chennai": "Tamil Nadu", "Bangalore": "Karnataka", "Hyderabad": "Telangana",
    "Kochi": "Kerala", "Coimbatore": "Tamil Nadu", "Mysore": "Karnataka", "Trivandrum": "Kerala",
    "Kolkata": "West Bengal", "Bhubaneswar": "Odisha", "Patna": "Bihar",
    "Guwahati": "Assam", "Ranchi": "Jharkhand",
    "Mumbai": "Maharashtra", "Pune": "Maharashtra", "Ahmedabad": "Gujarat",
    "Surat": "Gujarat", "Nagpur": "Maharashtra", "Goa": "Goa",
    "Bhopal": "MP", "Indore": "MP", "Raipur": "Chhattisgarh", "Varanasi": "UP",
}

INCOME_BRACKETS = [
    ("ews",          10000,  20000),
    ("low",          20000,  35000),
    ("lower_middle", 35000,  55000),
    ("middle",       55000,  90000),
    ("upper_middle", 90000,  200000),
    ("high",        200000,  500000),
    ("ultra_high",  500000, 2000000),
]
INCOME_WEIGHTS = [8, 18, 25, 25, 15, 7, 2]

EMPLOYMENT_TYPES = [
    ("salaried_private", 35), ("salaried_govt", 12), ("self_employed", 18),
    ("professional", 8), ("business_owner", 10), ("contract_worker", 7),
    ("gig_worker", 5), ("freelancer", 3), ("retired", 2),
]

# ── 8 SUBTLE DELINQUENCY ARCHETYPES ──────────────────────────
# These define HOW a customer deteriorates, not just IF
ARCHETYPES = {
    # archetype: (pct_of_total, delinquency_probability, description)
    "gradual_decline":    (0.020, 0.70, "Slow income erosion over 6 months"),
    "seasonal_stress":    (0.015, 0.45, "Festival/wedding overspend then struggle"),
    "hidden_leverager":   (0.010, 0.60, "Heavy lending app usage, low visible stress"),
    "income_shock":       (0.015, 0.80, "Sudden job loss or salary cut"),
    "over_extender":      (0.010, 0.65, "Multiple new loans, DTI creeping"),
    "lifestyle_creep":    (0.010, 0.40, "Post-promotion spend exceeds income growth"),
    "gig_volatility":     (0.005, 0.55, "Extreme month-to-month income variance"),
    "medical_emergency":  (0.005, 0.75, "Large healthcare spend, delayed recovery"),
    "clean_stable":       (0.910, 0.01, "No significant stress — true negatives"),
}

MERCHANT_CATEGORIES = {
    "grocery": (200, 3000), "dining": (150, 2500), "entertainment": (100, 2000),
    "utility": (500, 5000), "rent": (5000, 50000), "clothing": (500, 8000),
    "electronics": (1000, 50000), "healthcare": (200, 10000), "education": (500, 20000),
    "travel": (500, 30000), "insurance": (1000, 15000), "transfer": (500, 20000),
    "lending_app": (1000, 50000), "gambling": (500, 10000), "lottery": (200, 5000),
    "luxury_goods": (5000, 100000), "crypto_exchange": (1000, 30000),
    "payday_lender": (2000, 30000), "cash_advance": (1000, 20000),
    "fuel": (300, 3000), "subscription": (99, 1500), "emi_payment": (2000, 30000),
}

NORMAL_TXN_PROBS = {
    "grocery": 0.40, "dining": 0.20, "entertainment": 0.10, "utility": 0.03,
    "rent": 0.01, "clothing": 0.05, "electronics": 0.02, "healthcare": 0.02,
    "education": 0.01, "travel": 0.02, "insurance": 0.01, "transfer": 0.15,
    "fuel": 0.08, "subscription": 0.05, "emi_payment": 0.03,
}

CHANNELS = ["upi", "netbanking", "card", "neft", "imps", "auto"]


# ══════════════════════════════════════════════════════════════
# CUSTOMER GENERATION
# ══════════════════════════════════════════════════════════════
def assign_archetypes(n: int) -> List[str]:
    """Assign delinquency archetypes to customers using real-world distribution."""
    archetypes = []
    for name, (pct, _, _) in ARCHETYPES.items():
        count = int(n * pct)
        archetypes.extend([name] * count)
    # Fill remainder with clean_stable
    while len(archetypes) < n:
        archetypes.append("clean_stable")
    random.shuffle(archetypes)
    return archetypes[:n]


def generate_credit_score(income_bracket: str, archetype: str, emp_type: str, tenure: int) -> int:
    """CIBIL-realistic credit score (300-900)."""
    base = {"ews": (550, 680), "low": (580, 700), "lower_middle": (620, 740),
            "middle": (660, 780), "upper_middle": (700, 830),
            "high": (740, 860), "ultra_high": (760, 900)}
    lo, hi = base.get(income_bracket, (650, 750))
    score = random.randint(lo, hi)

    # Tenure bonus
    if tenure > 120: score += random.randint(10, 30)
    elif tenure > 60: score += random.randint(5, 15)

    # Employment adjustments
    if emp_type == "salaried_govt": score += random.randint(10, 25)
    if emp_type in ("gig_worker", "contract_worker"): score -= random.randint(10, 30)

    # Archetype-based deterioration (subtle — not all have bad credit initially)
    if archetype == "gradual_decline": score -= random.randint(20, 60)
    elif archetype == "income_shock": score -= random.randint(30, 80)
    elif archetype == "hidden_leverager": score -= random.randint(10, 40)
    elif archetype == "over_extender": score -= random.randint(15, 50)
    elif archetype == "medical_emergency": score -= random.randint(20, 70)
    elif archetype in ("seasonal_stress", "lifestyle_creep"): score -= random.randint(0, 25)

    return max(300, min(900, score))


def generate_products(income_bracket: str, age: int) -> List[str]:
    """Realistic product holdings."""
    products = ["savings_account"]
    probs = {
        "credit_card": {"ews": 0.05, "low": 0.15, "lower_middle": 0.40, "middle": 0.70,
                        "upper_middle": 0.88, "high": 0.95, "ultra_high": 0.99},
        "personal_loan": {"ews": 0.10, "low": 0.20, "lower_middle": 0.30, "middle": 0.35,
                          "upper_middle": 0.30, "high": 0.20, "ultra_high": 0.10},
        "home_loan": {"ews": 0.01, "low": 0.03, "lower_middle": 0.08, "middle": 0.20,
                      "upper_middle": 0.40, "high": 0.50, "ultra_high": 0.60},
        "vehicle_loan": {"ews": 0.02, "low": 0.05, "lower_middle": 0.12, "middle": 0.25,
                         "upper_middle": 0.35, "high": 0.30, "ultra_high": 0.15},
        "fixed_deposit": {"ews": 0.05, "low": 0.10, "lower_middle": 0.25, "middle": 0.40,
                          "upper_middle": 0.60, "high": 0.75, "ultra_high": 0.90},
        "insurance": {"ews": 0.08, "low": 0.15, "lower_middle": 0.25, "middle": 0.45,
                      "upper_middle": 0.65, "high": 0.80, "ultra_high": 0.95},
    }
    for product, bracket_probs in probs.items():
        p = bracket_probs.get(income_bracket, 0.1)
        if product == "home_loan" and age < 28: p *= 0.3
        if random.random() < p:
            products.append(product)
    return products


def calculate_dti(salary: float, products: List[str], archetype: str) -> float:
    """DTI ratio with archetype-aware adjustments."""
    total_emi = 0.0
    if "home_loan" in products: total_emi += salary * random.uniform(0.25, 0.45)
    if "personal_loan" in products: total_emi += salary * random.uniform(0.08, 0.20)
    if "vehicle_loan" in products: total_emi += salary * random.uniform(0.06, 0.15)
    if "credit_card" in products: total_emi += salary * random.uniform(0.02, 0.08)

    dti = total_emi / salary if salary > 0 else 0

    # Archetype-based DTI inflation
    if archetype == "over_extender": dti *= random.uniform(1.3, 1.8)
    elif archetype == "gradual_decline": dti *= random.uniform(1.1, 1.4)
    elif archetype == "hidden_leverager": dti *= random.uniform(1.2, 1.6)
    elif archetype == "income_shock": dti *= random.uniform(1.2, 1.5)
    elif archetype == "lifestyle_creep": dti *= random.uniform(1.1, 1.3)

    return round(min(dti, 0.95), 3)


def generate_customer_batch(batch_idx: int, batch_size: int, archetypes: List[str]) -> List[Dict]:
    """Generate one batch of customers."""
    customers = []
    start = batch_idx * batch_size
    end = min(start + batch_size, len(archetypes))

    for i in range(start, end):
        archetype = archetypes[i]
        gender = random.choices(["M", "F"], weights=[58, 42], k=1)[0]
        region = random.choice(REGIONS)
        city = random.choice(REGION_CITIES[region])

        # Employment
        emp_names = [e for e, _ in EMPLOYMENT_TYPES]
        emp_weights = [w for _, w in EMPLOYMENT_TYPES]
        # Gig workers are over-represented in gig_volatility archetype
        if archetype == "gig_volatility":
            emp_type = "gig_worker"
        else:
            emp_type = random.choices(emp_names, weights=emp_weights, k=1)[0]

        # Income
        bracket_name, sal_lo, sal_hi = random.choices(INCOME_BRACKETS, weights=INCOME_WEIGHTS, k=1)[0]
        sal_mult = {"salaried_govt": 0.85, "professional": 1.3, "business_owner": 1.2,
                    "gig_worker": 0.7, "contract_worker": 0.8, "retired": 0.6}.get(emp_type, 1.0)
        monthly_salary = round(random.uniform(sal_lo, sal_hi) * sal_mult, 2)

        # Age
        if emp_type == "retired":
            age = random.randint(58, 75)
        else:
            age = random.choices(list(range(21, 66)),
                                 weights=[1]*4 + [3]*6 + [5]*10 + [4]*10 + [2]*10 + [1]*5, k=1)[0]

        tenure = random.randint(3, 300)
        products = generate_products(bracket_name, age)
        credit_score = generate_credit_score(bracket_name, archetype, emp_type, tenure)
        dependents = random.choices([0, 1, 2, 3, 4, 5], weights=[15, 20, 30, 20, 10, 5], k=1)[0]
        dti = calculate_dti(monthly_salary, products, archetype)

        # Determine if this customer ACTUALLY becomes delinquent
        _, delin_prob, _ = ARCHETYPES[archetype]
        is_delinquent = random.random() < delin_prob

        # Salary day (most salaries 25th-5th)
        sal_day = random.choices(list(range(1, 29)),
                                 weights=[12, 10, 8, 5, 3] + [1]*18 + [5, 8, 10, 12, 15], k=1)[0]

        # Life events for stressed archetypes
        life_event = "none"
        if archetype == "income_shock": life_event = random.choice(["job_loss", "salary_cut"])
        elif archetype == "medical_emergency": life_event = "medical_emergency"
        elif archetype == "seasonal_stress": life_event = "wedding_expense"
        elif archetype == "gradual_decline": life_event = random.choice(["salary_cut", "business_failure"])

        contact = {
            "first_name": fake.first_name_male() if gender == "M" else fake.first_name_female(),
            "last_name": fake.last_name(),
        }

        customer = {
            "customer_id": f"CUST_{uuid.uuid4().hex[:12].upper()}",
            "first_name": contact["first_name"],
            "last_name": contact["last_name"],
            "age": age,
            "gender": gender,
            "phone": f"+91{random.choice(['6','7','8','9'])}{random.randint(100000000, 999999999)}",
            "email": f"{contact['first_name'].lower()}.{contact['last_name'].lower()}{random.randint(1,99)}@gmail.com",
            "city": city,
            "state": STATES_BY_CITY.get(city, "Unknown"),
            "region": region,
            "employment_type": emp_type,
            "income_bracket": bracket_name,
            "monthly_salary": monthly_salary,
            "salary_credit_day": sal_day,
            "tenure_months": tenure,
            "credit_score": credit_score,
            "num_dependents": dependents,
            "dti_ratio": dti,
            "product_holdings": products,
            "preferred_channel": random.choice(["sms", "email", "app_push", "whatsapp"]),
            "archetype": archetype,
            "is_delinquent": is_delinquent,
            "life_event": life_event,
        }
        customers.append(customer)

    return customers


# ══════════════════════════════════════════════════════════════
# TRANSACTION GENERATION WITH SUBTLE PATTERNS
# ══════════════════════════════════════════════════════════════
def generate_transactions_for_customer(customer: Dict, start_date: datetime,
                                       end_date: datetime) -> Tuple[List[Dict], List[Dict]]:
    """Generate 12 months of transactions with archetype-specific patterns."""
    txns = []
    payment_events = []
    cid = customer["customer_id"]
    salary = customer["monthly_salary"]
    archetype = customer["archetype"]
    is_delinquent = customer["is_delinquent"]
    sal_day = customer["salary_credit_day"]

    total_days = (end_date - start_date).days
    balance = salary * random.uniform(1.5, 4.0)  # Starting balance

    # ── Archetype-specific parameters ──
    # Month at which deterioration begins (0-indexed)
    if archetype == "gradual_decline":
        stress_start_month = random.randint(2, 5)
        income_decay = random.uniform(0.03, 0.08)  # % salary drop per month
    elif archetype == "seasonal_stress":
        stress_months = {9, 10, 11}  # Oct-Dec (Diwali/wedding season)
        spending_spike = random.uniform(1.5, 2.5)
    elif archetype == "hidden_leverager":
        lending_app_frequency = random.uniform(0.15, 0.30)
    elif archetype == "income_shock":
        shock_month = random.randint(4, 8)
        income_after_shock = salary * random.uniform(0.0, 0.4)
    elif archetype == "over_extender":
        new_loan_months = sorted(random.sample(range(3, 10), 3))
    elif archetype == "lifestyle_creep":
        promotion_month = random.randint(3, 6)
        spend_increase = random.uniform(1.4, 2.0)
    elif archetype == "gig_volatility":
        monthly_income_variance = random.uniform(0.3, 0.6)
    elif archetype == "medical_emergency":
        emergency_month = random.randint(3, 8)
        emergency_cost = salary * random.uniform(3, 8)

    current_date = start_date
    month_idx = 0
    prev_month = -1

    while current_date <= end_date:
        day_of_month = current_date.day
        current_month = current_date.month

        # Track month changes
        if current_month != prev_month:
            month_idx += 1
            prev_month = current_month

            # ── SALARY CREDIT ──
            effective_salary = salary
            if archetype == "gradual_decline" and month_idx > stress_start_month:
                decay_months = month_idx - stress_start_month
                effective_salary = salary * max(0.4, (1 - income_decay * decay_months))
            elif archetype == "income_shock" and month_idx > shock_month:
                effective_salary = income_after_shock
            elif archetype == "gig_volatility":
                jitter = np.random.normal(0, monthly_income_variance)
                effective_salary = salary * max(0.2, 1 + jitter)

            if day_of_month == sal_day or (day_of_month == 1 and sal_day > 28):
                # Salary with occasional late payments
                sal_late = random.randint(0, 3) if archetype in ("gig_volatility", "income_shock") else 0
                sal_date = current_date + timedelta(days=sal_late)
                txns.append({
                    "customer_id": cid,
                    "transaction_id": f"TXN_{uuid.uuid4().hex[:16].upper()}",
                    "timestamp": sal_date.isoformat(),
                    "amount": round(effective_salary, 2),
                    "direction": "credit",
                    "merchant_category": "salary",
                    "channel": "neft",
                    "description": "Monthly Salary Credit",
                })
                balance += effective_salary

        # ── DAILY TRANSACTIONS ──
        # Target: ~100 txns per customer over 12 months = ~8-9 per month = ~2 every 7 days
        # Skip some days randomly to achieve realistic sparse patterns
        if random.random() > 0.30:  # ~30% of days have transactions
            current_date += timedelta(days=1)
            continue

        daily_txn_count = random.randint(1, 3)

        # Seasonal spending boost (Diwali, wedding season)
        is_festival_season = current_date.month in (10, 11, 12)
        if is_festival_season and archetype == "seasonal_stress":
            daily_txn_count = random.randint(2, 5)

        for _ in range(daily_txn_count):
            # Normal transaction
            categories = list(NORMAL_TXN_PROBS.keys())
            weights = list(NORMAL_TXN_PROBS.values())

            # Archetype-specific category adjustments
            if archetype == "hidden_leverager" and random.random() < lending_app_frequency:
                cat = random.choice(["lending_app", "cash_advance", "payday_lender"])
            elif archetype == "lifestyle_creep" and month_idx > promotion_month:
                cat = random.choices(categories + ["luxury_goods", "dining", "travel"],
                                     weights=weights + [0.05, 0.10, 0.05], k=1)[0]
            elif archetype == "medical_emergency" and month_idx == emergency_month:
                cat = "healthcare"
            else:
                cat = random.choices(categories, weights=weights, k=1)[0]

            lo, hi = MERCHANT_CATEGORIES.get(cat, (200, 3000))

            # Scale amount by salary bracket
            salary_scale = min(salary / 50000, 3.0)
            amount = round(random.uniform(lo, hi) * salary_scale, 2)

            # Lifestyle creep: bigger spends after promotion
            if archetype == "lifestyle_creep" and month_idx > promotion_month:
                amount *= spend_increase

            # Seasonal stress: bigger festival spends
            if archetype == "seasonal_stress" and current_date.month in (10, 11):
                amount *= spending_spike

            # Medical emergency: one-time large expense
            if archetype == "medical_emergency" and month_idx == emergency_month and cat == "healthcare":
                amount = emergency_cost

            amount = round(min(amount, salary * 2), 2)  # Cap at 2x salary
            balance -= amount

            txn = {
                "customer_id": cid,
                "transaction_id": f"TXN_{uuid.uuid4().hex[:16].upper()}",
                "timestamp": (current_date + timedelta(
                    hours=random.randint(6, 23), minutes=random.randint(0, 59)
                )).isoformat(),
                "amount": amount,
                "direction": "debit",
                "merchant_category": cat,
                "channel": random.choice(CHANNELS),
                "description": f"{cat.replace('_', ' ').title()} payment",
            }
            txns.append(txn)

        # ── EMI/PAYMENT EVENTS (check at end-of-month) ──
        if day_of_month == 28 and is_delinquent:
            # Determine if this month's EMI is missed
            should_miss = False
            if archetype == "gradual_decline" and month_idx > stress_start_month + 2:
                should_miss = random.random() < 0.3 + 0.1 * (month_idx - stress_start_month)
            elif archetype == "income_shock" and month_idx > shock_month:
                should_miss = random.random() < 0.6
            elif archetype == "seasonal_stress" and current_date.month in (11, 12, 1):
                should_miss = random.random() < 0.4
            elif archetype == "hidden_leverager" and month_idx > 6:
                should_miss = random.random() < 0.35
            elif archetype == "over_extender" and month_idx > max(new_loan_months):
                should_miss = random.random() < 0.45
            elif archetype == "lifestyle_creep" and month_idx > promotion_month + 3:
                should_miss = random.random() < 0.25
            elif archetype == "gig_volatility":
                should_miss = random.random() < 0.3
            elif archetype == "medical_emergency" and month_idx >= emergency_month:
                should_miss = random.random() < 0.5
            elif archetype == "clean_stable":
                should_miss = random.random() < 0.005  # Very rare

            if should_miss and balance < salary * 0.5:
                loan_products = [p for p in customer.get("product_holdings", [])
                                 if p in ("personal_loan", "home_loan", "vehicle_loan", "credit_card")]
                if loan_products:
                    product = random.choice(loan_products)
                    emi_amount = salary * random.uniform(0.08, 0.25)
                    payment_events.append({
                        "customer_id": cid,
                        "event_id": f"PE_{uuid.uuid4().hex[:16].upper()}",
                        "event_date": current_date.isoformat(),
                        "event_type": "missed_emi",
                        "product_type": product,
                        "amount_due": round(emi_amount, 2),
                        "amount_paid": 0.0,
                        "days_past_due": random.randint(1, 30),
                        "running_balance": round(max(balance, 0), 2),
                    })

        current_date += timedelta(days=1)

    return txns, payment_events


# ══════════════════════════════════════════════════════════════
# BALANCE SNAPSHOTS
# ══════════════════════════════════════════════════════════════
def generate_balances(customer: Dict, txns: List[Dict]) -> List[Dict]:
    """Generate weekly balance snapshots from transactions."""
    if not txns:
        return []

    cid = customer["customer_id"]
    salary = customer["monthly_salary"]
    balance = salary * random.uniform(1.5, 4.0)
    balances = []

    # Sort txns by date
    sorted_txns = sorted(txns, key=lambda t: t["timestamp"])

    weekly_dates = set()
    for t in sorted_txns:
        dt = datetime.fromisoformat(t["timestamp"])
        # Weekly snapshot on Sundays
        week_start = dt - timedelta(days=dt.weekday())
        week_key = week_start.strftime("%Y-%m-%d")
        if week_key not in weekly_dates:
            weekly_dates.add(week_key)
            balances.append({
                "customer_id": cid,
                "timestamp": week_start.isoformat(),
                "balance": round(balance, 2),
                "account_type": "savings",
            })

        # Update balance
        if t["direction"] == "credit":
            balance += t["amount"]
        else:
            balance -= t["amount"]
            balance = max(balance, -salary * 0.5)  # Overdraft limit

    return balances


# ══════════════════════════════════════════════════════════════
# DATABASE OPERATIONS
# ══════════════════════════════════════════════════════════════
def get_conn():
    return psycopg2.connect(
        host=PostgresConfig.HOST, port=PostgresConfig.PORT,
        user=PostgresConfig.USER, password=PostgresConfig.PASSWORD,
        dbname=PostgresConfig.DB,
    )


def truncate_tables():
    """Truncate all data tables for a fresh start and ensure payment_events exists."""
    conn = get_conn()
    cursor = conn.cursor()

    # Create payment_events if it doesn't exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS payment_events (
            event_id VARCHAR(50) PRIMARY KEY,
            customer_id VARCHAR(50) REFERENCES customers(customer_id),
            event_date TIMESTAMP,
            event_type VARCHAR(50),
            product_type VARCHAR(50),
            amount_due NUMERIC(12,2),
            amount_paid NUMERIC(12,2) DEFAULT 0,
            days_past_due INTEGER DEFAULT 0,
            running_balance NUMERIC(14,2) DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()

    tables = [
        "risk_scores", "payment_events", "account_balances",
        "transactions", "streaming_features", "batch_features", "customers",
    ]
    for table in tables:
        try:
            cursor.execute(f"TRUNCATE TABLE {table} CASCADE")
            print(f"  ✓ Truncated {table}")
        except Exception as e:
            print(f"  ⚠ {table}: {e}")
            conn.rollback()
    conn.commit()
    cursor.close()
    conn.close()


def bulk_insert_customers(customers: List[Dict]):
    """Bulk insert customers."""
    conn = get_conn()
    cursor = conn.cursor()
    values = [
        (c["customer_id"], c["first_name"], c["last_name"], c["age"],
         c["gender"], c["city"], c["state"], c["region"],
         c["income_bracket"], c["monthly_salary"], c["salary_credit_day"],
         c["tenure_months"], c["credit_score"], c["product_holdings"],
         c["preferred_channel"])
        for c in customers
    ]
    execute_values(cursor, """
        INSERT INTO customers (
            customer_id, first_name, last_name, age, gender, city, state, region,
            income_bracket, monthly_salary, salary_credit_day, tenure_months,
            credit_score, product_holdings, preferred_channel
        ) VALUES %s ON CONFLICT (customer_id) DO NOTHING
    """, values, page_size=2000)
    conn.commit()
    cursor.close()
    conn.close()


def bulk_insert_transactions(txns: List[Dict]):
    """Bulk insert transactions in chunks. Schema: txn_id, customer_id, txn_type, merchant_category, amount, direction, channel, status, timestamp."""
    if not txns:
        return
    conn = get_conn()
    cursor = conn.cursor()
    CHUNK = 50000
    for i in range(0, len(txns), CHUNK):
        chunk = txns[i:i+CHUNK]
        values = [
            (t["transaction_id"], t["customer_id"], t["merchant_category"],
             t["merchant_category"], t["amount"], t["direction"],
             t["channel"], "completed", t["timestamp"])
            for t in chunk
        ]
        execute_values(cursor, """
            INSERT INTO transactions (
                txn_id, customer_id, txn_type, merchant_category,
                amount, direction, channel, status, timestamp
            ) VALUES %s ON CONFLICT DO NOTHING
        """, values, page_size=5000)
        conn.commit()
    cursor.close()
    conn.close()


def bulk_insert_balances(balances: List[Dict]):
    """Bulk insert balances. Schema: customer_id, balance, savings_balance, timestamp."""
    if not balances:
        return
    conn = get_conn()
    cursor = conn.cursor()
    CHUNK = 50000
    for i in range(0, len(balances), CHUNK):
        chunk = balances[i:i+CHUNK]
        values = [
            (b["customer_id"], b["balance"], b["balance"] * 0.3, b["timestamp"])
            for b in chunk
        ]
        execute_values(cursor, """
            INSERT INTO account_balances (customer_id, balance, savings_balance, timestamp)
            VALUES %s
        """, values, page_size=5000)
        conn.commit()
    cursor.close()
    conn.close()


def bulk_insert_payment_events(events: List[Dict]):
    """Bulk insert payment events."""
    if not events:
        return
    conn = get_conn()
    cursor = conn.cursor()
    values = [
        (e["customer_id"], e["event_id"], e["event_date"], e["event_type"],
         e["product_type"], e["amount_due"], e["amount_paid"],
         e["days_past_due"], e["running_balance"])
        for e in events
    ]
    execute_values(cursor, """
        INSERT INTO payment_events (
            customer_id, event_id, event_date, event_type, product_type,
            amount_due, amount_paid, days_past_due, running_balance
        ) VALUES %s ON CONFLICT DO NOTHING
    """, values, page_size=5000)
    conn.commit()
    cursor.close()
    conn.close()


# ══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════
def run_pipeline():
    print("=" * 70)
    print("PRE-DELINQUENCY ENGINE — PRODUCTION DATA GENERATION")
    print(f"  Customers:    {NUM_CUSTOMERS:,}")
    print(f"  Months:       {TXN_MONTHS}")
    print(f"  Batch Size:   {BATCH_SIZE:,}")
    print(f"  Target:       5M+ transactions, 5-7% delinquency rate")
    print("=" * 70)

    pipeline_start = time.time()

    # Step 1: Truncate existing data
    print("\n[1/5] Truncating existing data...")
    truncate_tables()

    # Step 2: Assign archetypes
    print(f"\n[2/5] Assigning delinquency archetypes to {NUM_CUSTOMERS:,} customers...")
    all_archetypes = assign_archetypes(NUM_CUSTOMERS)
    archetype_counts = {}
    for a in all_archetypes:
        archetype_counts[a] = archetype_counts.get(a, 0) + 1
    for name, count in sorted(archetype_counts.items(), key=lambda x: -x[1]):
        _, delin_prob, desc = ARCHETYPES[name]
        print(f"  {name:20s}: {count:6,} customers ({delin_prob*100:.0f}% → delinquent) — {desc}")

    # Step 3: Generate & insert in batches
    end_date = datetime(2026, 3, 1)
    start_date = end_date - timedelta(days=TXN_MONTHS * 30)

    num_batches = math.ceil(NUM_CUSTOMERS / BATCH_SIZE)
    total_txns = 0
    total_events = 0
    total_balances = 0
    delinquent_customers = set()

    print(f"\n[3/5] Generating customers and transactions ({num_batches} batches)...")

    for batch_idx in range(num_batches):
        batch_start = time.time()
        batch_offset = batch_idx * BATCH_SIZE
        batch_end = min(batch_offset + BATCH_SIZE, NUM_CUSTOMERS)
        batch_count = batch_end - batch_offset

        # Generate customers
        customers = generate_customer_batch(batch_idx, BATCH_SIZE, all_archetypes)

        # Insert customers
        bulk_insert_customers(customers)

        # Generate transactions for this batch
        batch_txns = []
        batch_events = []
        batch_balances = []

        for c in customers:
            txns, events = generate_transactions_for_customer(c, start_date, end_date)
            bals = generate_balances(c, txns)
            batch_txns.extend(txns)
            batch_events.extend(events)
            batch_balances.extend(bals)
            if events:
                delinquent_customers.add(c["customer_id"])

        # Insert transactions
        print(f"  Batch {batch_idx+1}/{num_batches}: inserting {len(batch_txns):,} txns, "
              f"{len(batch_events):,} events, {len(batch_balances):,} balances...", end="", flush=True)

        bulk_insert_transactions(batch_txns)
        bulk_insert_balances(batch_balances)
        bulk_insert_payment_events(batch_events)

        total_txns += len(batch_txns)
        total_events += len(batch_events)
        total_balances += len(batch_balances)

        batch_time = time.time() - batch_start
        print(f" ({batch_time:.1f}s)")

        # Free memory
        del batch_txns, batch_events, batch_balances, customers
        gc.collect()

    # Step 4: Compute streaming features via SQL
    print("\n[4/5] Computing streaming features via SQL...")
    try:
        conn = get_conn()
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS streaming_features CASCADE")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS streaming_features (
                customer_id VARCHAR(50) PRIMARY KEY,
                discretionary_spend_7d DOUBLE PRECISION DEFAULT 0,
                discretionary_spend_30d DOUBLE PRECISION DEFAULT 0,
                essential_spend_7d DOUBLE PRECISION DEFAULT 0,
                essential_spend_30d DOUBLE PRECISION DEFAULT 0,
                txn_count_7d INTEGER DEFAULT 0,
                txn_count_30d INTEGER DEFAULT 0,
                avg_txn_amount_7d DOUBLE PRECISION DEFAULT 0,
                avg_txn_amount_30d DOUBLE PRECISION DEFAULT 0,
                max_single_txn_7d DOUBLE PRECISION DEFAULT 0,
                credit_debit_ratio_7d DOUBLE PRECISION DEFAULT 0,
                unique_merchants_7d INTEGER DEFAULT 0,
                late_night_txn_ratio_7d DOUBLE PRECISION DEFAULT 0
            )
        """)

        cursor.execute("""
            INSERT INTO streaming_features (
                customer_id,
                discretionary_spend_7d, discretionary_spend_30d,
                essential_spend_7d, essential_spend_30d,
                txn_count_7d, txn_count_30d,
                avg_txn_amount_7d, avg_txn_amount_30d,
                max_single_txn_7d, credit_debit_ratio_7d,
                unique_merchants_7d, late_night_txn_ratio_7d
            )
            SELECT
                c.customer_id,
                COALESCE(t7.disc_spend, 0), COALESCE(t30.disc_spend, 0),
                COALESCE(t7.ess_spend, 0), COALESCE(t30.ess_spend, 0),
                COALESCE(t7.txn_count, 0), COALESCE(t30.txn_count, 0),
                COALESCE(t7.avg_amt, 0), COALESCE(t30.avg_amt, 0),
                COALESCE(t7.max_amt, 0), COALESCE(t7.cr_dr_ratio, 0),
                COALESCE(t7.uniq_merch, 0), COALESCE(t7.late_night, 0)
            FROM customers c
            LEFT JOIN LATERAL (
                SELECT
                    SUM(CASE WHEN merchant_category IN ('dining','entertainment','clothing','travel','luxury_goods','gambling','lottery','crypto_exchange') THEN amount ELSE 0 END) AS disc_spend,
                    SUM(CASE WHEN merchant_category IN ('grocery','utility','rent','healthcare','education','insurance','emi_payment') THEN amount ELSE 0 END) AS ess_spend,
                    COUNT(*) AS txn_count,
                    AVG(amount) AS avg_amt,
                    MAX(amount) AS max_amt,
                    CASE WHEN SUM(CASE WHEN direction='debit' THEN amount ELSE 0 END) > 0
                         THEN SUM(CASE WHEN direction='credit' THEN amount ELSE 0 END) /
                              SUM(CASE WHEN direction='debit' THEN amount ELSE 0 END)
                         ELSE 0 END AS cr_dr_ratio,
                    COUNT(DISTINCT merchant_category) AS uniq_merch,
                    CASE WHEN COUNT(*) > 0
                         THEN COUNT(CASE WHEN EXTRACT(HOUR FROM timestamp::timestamp) >= 22
                                         OR EXTRACT(HOUR FROM timestamp::timestamp) < 5 THEN 1 END)::float / COUNT(*)
                         ELSE 0 END AS late_night
                FROM transactions t
                WHERE t.customer_id = c.customer_id
                  AND t.timestamp::date >= (CURRENT_DATE - INTERVAL '7 days')
            ) t7 ON true
            LEFT JOIN LATERAL (
                SELECT
                    SUM(CASE WHEN merchant_category IN ('dining','entertainment','clothing','travel','luxury_goods','gambling','lottery','crypto_exchange') THEN amount ELSE 0 END) AS disc_spend,
                    SUM(CASE WHEN merchant_category IN ('grocery','utility','rent','healthcare','education','insurance','emi_payment') THEN amount ELSE 0 END) AS ess_spend,
                    COUNT(*) AS txn_count,
                    AVG(amount) AS avg_amt
                FROM transactions t
                WHERE t.customer_id = c.customer_id
                  AND t.timestamp::date >= (CURRENT_DATE - INTERVAL '30 days')
            ) t30 ON true
            ON CONFLICT (customer_id) DO UPDATE SET
                discretionary_spend_7d = EXCLUDED.discretionary_spend_7d,
                discretionary_spend_30d = EXCLUDED.discretionary_spend_30d,
                essential_spend_7d = EXCLUDED.essential_spend_7d,
                essential_spend_30d = EXCLUDED.essential_spend_30d,
                txn_count_7d = EXCLUDED.txn_count_7d,
                txn_count_30d = EXCLUDED.txn_count_30d,
                avg_txn_amount_7d = EXCLUDED.avg_txn_amount_7d,
                avg_txn_amount_30d = EXCLUDED.avg_txn_amount_30d,
                max_single_txn_7d = EXCLUDED.max_single_txn_7d,
                credit_debit_ratio_7d = EXCLUDED.credit_debit_ratio_7d,
                unique_merchants_7d = EXCLUDED.unique_merchants_7d,
                late_night_txn_ratio_7d = EXCLUDED.late_night_txn_ratio_7d
        """)
        conn.commit()
        print(f"  ✓ Streaming features computed for all customers")

        # Batch features
        cursor.execute("DROP TABLE IF EXISTS batch_features CASCADE")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS batch_features (
                customer_id VARCHAR(50) PRIMARY KEY,
                salary_volatility_3m DOUBLE PRECISION DEFAULT 0,
                balance_trend_slope DOUBLE PRECISION DEFAULT 0,
                spend_velocity_change DOUBLE PRECISION DEFAULT 0,
                merchant_diversity_delta DOUBLE PRECISION DEFAULT 0,
                weekend_spend_ratio DOUBLE PRECISION DEFAULT 0,
                recurring_payment_stability DOUBLE PRECISION DEFAULT 0,
                cash_advance_frequency DOUBLE PRECISION DEFAULT 0,
                high_risk_merchant_ratio DOUBLE PRECISION DEFAULT 0
            )
        """)
        cursor.execute("""
            INSERT INTO batch_features (
                customer_id,
                salary_volatility_3m, balance_trend_slope,
                spend_velocity_change, merchant_diversity_delta,
                weekend_spend_ratio, recurring_payment_stability,
                cash_advance_frequency, high_risk_merchant_ratio
            )
            SELECT
                c.customer_id,
                COALESCE(sal.salary_vol, 0),
                COALESCE(bal.trend_slope, 0),
                COALESCE(vel.spend_change, 0),
                COALESCE(merch.diversity_delta, 0),
                COALESCE(we.weekend_ratio, 0),
                COALESCE(rec.stability, 0),
                COALESCE(ca.ca_freq, 0),
                COALESCE(hr.hr_ratio, 0)
            FROM customers c
            LEFT JOIN LATERAL (
                SELECT STDDEV(amount) / NULLIF(AVG(amount), 0) AS salary_vol
                FROM transactions t WHERE t.customer_id = c.customer_id
                AND t.direction = 'credit' AND t.merchant_category = 'salary'
            ) sal ON true
            LEFT JOIN LATERAL (
                SELECT CASE WHEN COUNT(*) > 1
                    THEN (MAX(balance) - MIN(balance)) / NULLIF(COUNT(*), 0)
                    ELSE 0 END AS trend_slope
                FROM account_balances b WHERE b.customer_id = c.customer_id
            ) bal ON true
            LEFT JOIN LATERAL (
                SELECT CASE WHEN COUNT(*) > 0
                    THEN STDDEV(amount) / NULLIF(AVG(amount), 0) ELSE 0 END AS spend_change
                FROM transactions t WHERE t.customer_id = c.customer_id AND t.direction = 'debit'
            ) vel ON true
            LEFT JOIN LATERAL (
                SELECT COUNT(DISTINCT merchant_category)::float /
                    NULLIF(GREATEST(COUNT(*), 1), 0) AS diversity_delta
                FROM transactions t WHERE t.customer_id = c.customer_id
            ) merch ON true
            LEFT JOIN LATERAL (
                SELECT CASE WHEN COUNT(*) > 0
                    THEN COUNT(CASE WHEN EXTRACT(DOW FROM t.timestamp::timestamp) IN (0, 6) THEN 1 END)::float / COUNT(*)
                    ELSE 0 END AS weekend_ratio
                FROM transactions t WHERE t.customer_id = c.customer_id AND t.direction = 'debit'
            ) we ON true
            LEFT JOIN LATERAL (
                SELECT CASE WHEN COUNT(*) > 0
                    THEN 1.0 - (STDDEV(amount) / NULLIF(AVG(amount), 0))
                    ELSE 0 END AS stability
                FROM transactions t WHERE t.customer_id = c.customer_id
                AND t.merchant_category IN ('emi_payment', 'insurance', 'rent', 'utility')
            ) rec ON true
            LEFT JOIN LATERAL (
                SELECT COUNT(*)::float / NULLIF(GREATEST(
                    (SELECT COUNT(*) FROM transactions t2 WHERE t2.customer_id = c.customer_id), 1), 0) AS ca_freq
                FROM transactions t WHERE t.customer_id = c.customer_id
                AND t.merchant_category IN ('cash_advance', 'payday_lender', 'lending_app')
            ) ca ON true
            LEFT JOIN LATERAL (
                SELECT COUNT(*)::float / NULLIF(GREATEST(
                    (SELECT COUNT(*) FROM transactions t2 WHERE t2.customer_id = c.customer_id), 1), 0) AS hr_ratio
                FROM transactions t WHERE t.customer_id = c.customer_id
                AND t.merchant_category IN ('gambling', 'lottery', 'crypto_exchange', 'cash_advance', 'payday_lender')
            ) hr ON true
            ON CONFLICT (customer_id) DO UPDATE SET
                salary_volatility_3m = EXCLUDED.salary_volatility_3m,
                balance_trend_slope = EXCLUDED.balance_trend_slope,
                spend_velocity_change = EXCLUDED.spend_velocity_change,
                merchant_diversity_delta = EXCLUDED.merchant_diversity_delta,
                weekend_spend_ratio = EXCLUDED.weekend_spend_ratio,
                recurring_payment_stability = EXCLUDED.recurring_payment_stability,
                cash_advance_frequency = EXCLUDED.cash_advance_frequency,
                high_risk_merchant_ratio = EXCLUDED.high_risk_merchant_ratio
        """)
        conn.commit()
        print(f"  ✓ Batch features computed for all customers")
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"  ⚠ Feature computation warning: {e}")

    # Step 5: Validation
    print("\n[5/5] Validating generated data...")
    conn = get_conn()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM customers")
    cust_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM transactions")
    txn_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM account_balances")
    bal_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM payment_events")
    event_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT customer_id) FROM payment_events")
    delin_count = cursor.fetchone()[0]

    cursor.execute("SELECT income_bracket, COUNT(*) FROM customers GROUP BY income_bracket ORDER BY COUNT(*) DESC")
    bracket_dist = cursor.fetchall()

    cursor.execute("SELECT employment_type, COUNT(*) FROM customers GROUP BY employment_type ORDER BY COUNT(*) DESC LIMIT 10")
    emp_dist = cursor.fetchall()

    cursor.close()
    conn.close()

    delin_rate = delin_count / cust_count * 100 if cust_count > 0 else 0

    pipeline_time = time.time() - pipeline_start

    print("\n" + "=" * 70)
    print("DATA GENERATION COMPLETE")
    print("=" * 70)
    print(f"  Customers:       {cust_count:>10,}")
    print(f"  Transactions:    {txn_count:>10,}")
    print(f"  Balances:        {bal_count:>10,}")
    print(f"  Payment Events:  {event_count:>10,}")
    print(f"  Delinquent:      {delin_count:>10,} ({delin_rate:.1f}%)")
    print(f"  Time:            {pipeline_time/60:>10.1f} minutes")
    print()
    print("  Income Distribution:")
    for bracket, count in bracket_dist:
        pct = count / cust_count * 100
        print(f"    {bracket:15s}: {count:6,} ({pct:.1f}%)")
    print()
    print("  Employment Distribution:")
    for emp, count in emp_dist:
        pct = count / cust_count * 100
        print(f"    {emp:20s}: {count:6,} ({pct:.1f}%)")
    print()

    # Validation checks
    checks_passed = 0
    total_checks = 5

    if cust_count >= 50000:
        print("  ✓ Customer count:    PASS (≥50,000)")
        checks_passed += 1
    else:
        print(f"  ✗ Customer count:    FAIL ({cust_count} < 50,000)")

    if txn_count >= 4000000:
        print(f"  ✓ Transaction count: PASS (≥4M) — {txn_count:,}")
        checks_passed += 1
    else:
        print(f"  ✗ Transaction count: FAIL ({txn_count:,} < 4M)")

    if 3.0 <= delin_rate <= 12.0:
        print(f"  ✓ Delinquency rate:  PASS (3-12%) — {delin_rate:.1f}%")
        checks_passed += 1
    else:
        print(f"  ✗ Delinquency rate:  FAIL ({delin_rate:.1f}% outside 3-12%)")

    if len(bracket_dist) >= 5:
        print(f"  ✓ Income diversity:  PASS (≥5 brackets)")
        checks_passed += 1
    else:
        print(f"  ✗ Income diversity:  FAIL ({len(bracket_dist)} brackets)")

    if len(emp_dist) >= 7:
        print(f"  ✓ Employment types:  PASS (≥7 types)")
        checks_passed += 1
    else:
        print(f"  ✗ Employment types:  FAIL ({len(emp_dist)} types)")

    print(f"\n  RESULT: {checks_passed}/{total_checks} checks passed")
    print("=" * 70)

    return {
        "customers": cust_count,
        "transactions": txn_count,
        "balances": bal_count,
        "payment_events": event_count,
        "delinquent_customers": delin_count,
        "delinquency_rate": delin_rate,
        "pipeline_time_minutes": pipeline_time / 60,
    }


if __name__ == "__main__":
    run_pipeline()
