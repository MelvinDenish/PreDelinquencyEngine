"""
Customer Profile Generator — Production Grade
Generates bank-level realistic customer profiles with demographics, salary patterns,
product holdings, employment data, life events, debt metrics, and contact details.
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

fake = Faker('en_IN')  # Indian locale

# ─────────────────────────────────────────────
# Realistic Indian Banking Constants
# ─────────────────────────────────────────────
REGIONS = ["North", "South", "East", "West", "Central"]
REGION_CITIES = {
    "North": ["Delhi", "Noida", "Gurgaon", "Jaipur", "Lucknow", "Chandigarh", "Amritsar", "Dehradun"],
    "South": ["Chennai", "Bangalore", "Hyderabad", "Kochi", "Coimbatore", "Mysore", "Visakhapatnam", "Trivandrum"],
    "East": ["Kolkata", "Bhubaneswar", "Patna", "Guwahati", "Ranchi", "Siliguri"],
    "West": ["Mumbai", "Pune", "Ahmedabad", "Surat", "Nagpur", "Goa", "Vadodara", "Nashik"],
    "Central": ["Bhopal", "Indore", "Raipur", "Jabalpur", "Nagpur", "Varanasi"],
}
STATES_BY_CITY = {
    "Delhi": "Delhi", "Noida": "Uttar Pradesh", "Gurgaon": "Haryana",
    "Jaipur": "Rajasthan", "Lucknow": "Uttar Pradesh", "Chandigarh": "Chandigarh",
    "Amritsar": "Punjab", "Dehradun": "Uttarakhand",
    "Chennai": "Tamil Nadu", "Bangalore": "Karnataka", "Hyderabad": "Telangana",
    "Kochi": "Kerala", "Coimbatore": "Tamil Nadu", "Mysore": "Karnataka",
    "Visakhapatnam": "Andhra Pradesh", "Trivandrum": "Kerala",
    "Kolkata": "West Bengal", "Bhubaneswar": "Odisha", "Patna": "Bihar",
    "Guwahati": "Assam", "Ranchi": "Jharkhand", "Siliguri": "West Bengal",
    "Mumbai": "Maharashtra", "Pune": "Maharashtra", "Ahmedabad": "Gujarat",
    "Surat": "Gujarat", "Nagpur": "Maharashtra", "Goa": "Goa",
    "Vadodara": "Gujarat", "Nashik": "Maharashtra",
    "Bhopal": "Madhya Pradesh", "Indore": "Madhya Pradesh",
    "Raipur": "Chhattisgarh", "Jabalpur": "Madhya Pradesh", "Varanasi": "Uttar Pradesh",
}

# RBI-aligned income brackets (monthly, INR)
INCOME_BRACKETS = [
    ("ews",          10000,  20000),    # Economically Weaker Section
    ("low",          20000,  35000),    # Lower income
    ("lower_middle", 35000,  55000),    # Lower middle
    ("middle",       55000,  90000),    # Middle class
    ("upper_middle", 90000,  200000),   # Upper middle
    ("high",        200000,  500000),   # High net worth
    ("ultra_high",  500000, 2000000),   # Ultra HNI
]
INCOME_WEIGHTS = [8, 18, 25, 25, 15, 7, 2]  # Realistic distribution

EMPLOYMENT_TYPES = [
    ("salaried_private",   35),   # Largest segment
    ("salaried_govt",      12),
    ("self_employed",      18),
    ("professional",        8),   # Doctor, lawyer, CA
    ("business_owner",     10),
    ("contract_worker",     7),
    ("gig_worker",          5),   # Delivery, ride-share
    ("freelancer",          3),
    ("retired",             2),
]

INDUSTRY_SECTORS = {
    "salaried_private": ["IT/ITES", "BFSI", "Manufacturing", "Retail", "Pharma", "Telecom", "FMCG", "Automotive", "E-commerce"],
    "salaried_govt": ["Central Govt", "State Govt", "PSU", "Defence", "Railways", "Education"],
    "self_employed": ["Retail Shop", "Trading", "Agriculture", "Textile", "Food & Beverage", "Construction"],
    "professional": ["Healthcare", "Legal", "Chartered Accountancy", "Architecture", "Consulting"],
    "business_owner": ["SME Manufacturing", "Import/Export", "Hospitality", "Real Estate", "Tech Startup"],
    "contract_worker": ["IT/ITES", "Construction", "Manufacturing", "Logistics"],
    "gig_worker": ["Ride-sharing", "Food Delivery", "Courier", "Home Services"],
    "freelancer": ["IT/ITES", "Design", "Content/Media", "Consulting"],
    "retired": ["Pension"],
}

# Life events that can trigger financial stress
LIFE_EVENTS = [
    ("job_loss",           0.04, "high"),       # 4% of stressed customers
    ("salary_cut",         0.08, "medium"),     # COVID-era pattern
    ("medical_emergency",  0.06, "high"),
    ("wedding_expense",    0.05, "medium"),
    ("new_baby",           0.04, "low"),
    ("divorce",            0.02, "high"),
    ("business_failure",   0.03, "high"),
    ("education_expense",  0.05, "medium"),     # Child's higher education
    ("relocation",         0.03, "low"),
    ("accident",           0.02, "high"),
    ("family_death",       0.02, "medium"),
    ("investment_loss",    0.03, "medium"),      # Stock/crypto loss
    ("none",               0.54, "none"),        # No life event
]

PRODUCTS = ["savings_account", "credit_card", "personal_loan", "home_loan",
            "fixed_deposit", "recurring_deposit", "demat_account", "insurance",
            "vehicle_loan", "gold_loan", "education_loan", "overdraft"]

CHANNELS = ["sms", "email", "app_push", "whatsapp", "rm_call"]


def _generate_salary_day() -> int:
    """Most salaries arrive between 25th-5th of month."""
    days = list(range(1, 29))
    day_weights = [12, 10, 8, 5, 3] + [1]*18 + [5, 8, 10, 12, 15]
    return random.choices(days, weights=day_weights, k=1)[0]


def _generate_credit_score(income_bracket: str, is_stressed: bool,
                           employment_type: str, tenure: int) -> int:
    """Generate CIBIL-realistic credit score (300–900)."""
    base_ranges = {
        "ews": (550, 680), "low": (580, 700), "lower_middle": (620, 740),
        "middle": (660, 780), "upper_middle": (700, 830),
        "high": (740, 860), "ultra_high": (760, 900),
    }
    low, high = base_ranges.get(income_bracket, (650, 750))
    score = random.randint(low, high)

    # Tenure bonus: longer banking history = better score
    if tenure > 120:
        score += random.randint(10, 30)
    elif tenure > 60:
        score += random.randint(5, 15)

    # Government employees tend to have stable scores
    if employment_type == "salaried_govt":
        score += random.randint(10, 25)

    # Gig/contract workers have more volatile scores
    if employment_type in ("gig_worker", "contract_worker", "freelancer"):
        score -= random.randint(10, 30)

    if is_stressed:
        score -= random.randint(40, 120)

    return max(300, min(900, score))


def _generate_products(income_bracket: str, age: int, employment_type: str) -> List[str]:
    """Generate realistic product holdings based on demographics."""
    products = ["savings_account"]
    product_probs = {
        "credit_card":      {"ews": 0.05, "low": 0.15, "lower_middle": 0.40, "middle": 0.70,
                             "upper_middle": 0.88, "high": 0.95, "ultra_high": 0.99},
        "personal_loan":    {"ews": 0.10, "low": 0.20, "lower_middle": 0.30, "middle": 0.35,
                             "upper_middle": 0.30, "high": 0.20, "ultra_high": 0.10},
        "home_loan":        {"ews": 0.01, "low": 0.03, "lower_middle": 0.08, "middle": 0.20,
                             "upper_middle": 0.40, "high": 0.50, "ultra_high": 0.60},
        "vehicle_loan":     {"ews": 0.02, "low": 0.05, "lower_middle": 0.12, "middle": 0.25,
                             "upper_middle": 0.35, "high": 0.30, "ultra_high": 0.15},
        "gold_loan":        {"ews": 0.15, "low": 0.12, "lower_middle": 0.08, "middle": 0.05,
                             "upper_middle": 0.03, "high": 0.01, "ultra_high": 0.005},
        "education_loan":   {"ews": 0.08, "low": 0.10, "lower_middle": 0.12, "middle": 0.10,
                             "upper_middle": 0.08, "high": 0.05, "ultra_high": 0.02},
        "fixed_deposit":    {"ews": 0.05, "low": 0.10, "lower_middle": 0.25, "middle": 0.40,
                             "upper_middle": 0.60, "high": 0.75, "ultra_high": 0.90},
        "recurring_deposit": {"ews": 0.08, "low": 0.15, "lower_middle": 0.25, "middle": 0.30,
                              "upper_middle": 0.30, "high": 0.20, "ultra_high": 0.10},
        "demat_account":    {"ews": 0.02, "low": 0.05, "lower_middle": 0.12, "middle": 0.30,
                             "upper_middle": 0.55, "high": 0.75, "ultra_high": 0.90},
        "insurance":        {"ews": 0.08, "low": 0.15, "lower_middle": 0.25, "middle": 0.45,
                             "upper_middle": 0.65, "high": 0.80, "ultra_high": 0.95},
        "overdraft":        {"ews": 0.01, "low": 0.02, "lower_middle": 0.05, "middle": 0.10,
                             "upper_middle": 0.20, "high": 0.35, "ultra_high": 0.50},
    }

    for product, probs in product_probs.items():
        prob = probs.get(income_bracket, 0.1)
        # Age adjustments
        if product == "home_loan" and age < 28:
            prob *= 0.3
        if product == "education_loan" and age > 45:
            prob *= 0.2
        if product == "demat_account" and age < 25:
            prob *= 0.5
        if random.random() < prob:
            products.append(product)

    return products


def _calculate_dti(salary: float, products: List[str], is_stressed: bool) -> float:
    """Calculate debt-to-income ratio (monthly debt payments / monthly income)."""
    total_emi = 0.0
    if "home_loan" in products:
        total_emi += salary * random.uniform(0.25, 0.45)
    if "personal_loan" in products:
        total_emi += salary * random.uniform(0.08, 0.20)
    if "vehicle_loan" in products:
        total_emi += salary * random.uniform(0.06, 0.15)
    if "education_loan" in products:
        total_emi += salary * random.uniform(0.05, 0.12)
    if "gold_loan" in products:
        total_emi += salary * random.uniform(0.04, 0.10)
    if "credit_card" in products:
        # Minimum payment on revolving balance
        total_emi += salary * random.uniform(0.02, 0.08)

    dti = total_emi / salary if salary > 0 else 0

    # Stressed customers tend to have higher DTI
    if is_stressed:
        dti = min(dti * random.uniform(1.1, 1.5), 0.95)

    return round(min(dti, 0.95), 3)


def _generate_life_event(is_stressed: bool) -> Dict:
    """Assign a life event (primarily to stressed customers)."""
    if not is_stressed:
        return {"event": "none", "severity": "none", "months_ago": 0}

    events = [(e, w, s) for e, w, s in LIFE_EVENTS if e != "none"]
    event_names = [e for e, _, _ in events]
    event_weights = [w for _, w, _ in events]
    event_severities = {e: s for e, _, s in events}

    chosen = random.choices(event_names, weights=event_weights, k=1)[0]
    return {
        "event": chosen,
        "severity": event_severities[chosen],
        "months_ago": random.randint(1, 6),
    }


def _generate_contact(gender: str) -> Dict:
    """Generate realistic Indian contact details."""
    # Indian phone: +91 followed by 10 digits starting with 6-9
    phone = f"+91{random.choice(['6','7','8','9'])}{random.randint(100000000, 999999999)}"
    first = fake.first_name_male() if gender == "M" else fake.first_name_female()
    last = fake.last_name()
    email_domain = random.choice(["gmail.com", "yahoo.co.in", "outlook.com",
                                  "rediffmail.com", "hotmail.com"])
    email = f"{first.lower()}.{last.lower()}{random.randint(1, 99)}@{email_domain}"
    return {"first_name": first, "last_name": last, "phone": phone, "email": email}


def generate_customers(num_customers: int = None, stress_pct: float = None) -> List[Dict]:
    """Generate production-grade realistic customer profiles."""
    num_customers = num_customers or DataGenConfig.NUM_CUSTOMERS
    stress_pct = stress_pct or DataGenConfig.STRESS_CUSTOMER_PCT

    num_stressed = int(num_customers * stress_pct)
    stressed_indices = set(random.sample(range(num_customers), num_stressed))

    customers = []
    for i in range(num_customers):
        is_stressed = i in stressed_indices
        gender = random.choices(["M", "F"], weights=[58, 42], k=1)[0]  # Indian banking skew
        region = random.choice(REGIONS)
        city = random.choice(REGION_CITIES[region])
        state = STATES_BY_CITY.get(city, "Unknown")

        # Employment type
        emp_names = [e for e, _ in EMPLOYMENT_TYPES]
        emp_weights = [w for _, w in EMPLOYMENT_TYPES]
        employment_type = random.choices(emp_names, weights=emp_weights, k=1)[0]

        # Industry
        industry = random.choice(INDUSTRY_SECTORS.get(employment_type, ["Other"]))

        # Income based on employment and bracket
        bracket_name, salary_low, salary_high = random.choices(
            INCOME_BRACKETS, weights=INCOME_WEIGHTS, k=1
        )[0]

        # Adjust salary for employment type
        salary_multiplier = {
            "salaried_govt": 0.85, "professional": 1.3, "business_owner": 1.2,
            "gig_worker": 0.7, "contract_worker": 0.8, "retired": 0.6,
        }.get(employment_type, 1.0)
        monthly_salary = round(random.uniform(salary_low, salary_high) * salary_multiplier, 2)

        # Age distribution (realistic for banking)
        if employment_type == "retired":
            age = random.randint(58, 75)
        else:
            age = random.choices(
                list(range(21, 66)),
                weights=[1]*4 + [3]*6 + [5]*10 + [4]*10 + [2]*10 + [1]*5,  # Peak 30-50
                k=1
            )[0]

        tenure = random.randint(3, 300)  # 3 months to 25 years
        products = _generate_products(bracket_name, age, employment_type)
        credit_score = _generate_credit_score(bracket_name, is_stressed, employment_type, tenure)
        dependents = random.choices([0, 1, 2, 3, 4, 5], weights=[15, 20, 30, 20, 10, 5], k=1)[0]
        dti = _calculate_dti(monthly_salary, products, is_stressed)

        # Life events (stress triggers)
        life_event = _generate_life_event(is_stressed)

        # Contact info
        contact = _generate_contact(gender)

        # Number of existing loans
        loan_products = [p for p in products if p in
                         ("personal_loan", "home_loan", "vehicle_loan", "education_loan", "gold_loan")]
        total_debt = sum([
            monthly_salary * random.uniform(12, 60) for _ in loan_products
        ])

        # Channel preference weighted by age & income
        if age > 55:
            channel_weights = [30, 15, 5, 10, 40]  # Older prefer SMS + RM
        elif bracket_name in ("high", "ultra_high"):
            channel_weights = [10, 25, 20, 15, 30]  # HNI prefer RM + email
        else:
            channel_weights = [20, 15, 30, 25, 10]  # Young prefer app + WhatsApp

        # Risk segment
        if tenure < 12:
            risk_segment = "new_to_bank"
        elif bracket_name in ("high", "ultra_high"):
            risk_segment = "high_value"
        elif is_stressed:
            risk_segment = "vulnerable"
        else:
            risk_segment = "stable_core"

        customer = {
            "customer_id": f"CUST_{uuid.uuid4().hex[:12].upper()}",
            "first_name": contact["first_name"],
            "last_name": contact["last_name"],
            "age": age,
            "gender": gender,
            "phone": contact["phone"],
            "email": contact["email"],
            "city": city,
            "state": state,
            "region": region,
            "employment_type": employment_type,
            "industry_sector": industry,
            "income_bracket": bracket_name,
            "monthly_salary": monthly_salary,
            "salary_credit_day": _generate_salary_day(),
            "tenure_months": tenure,
            "credit_score": credit_score,
            "num_dependents": dependents,
            "dti_ratio": dti,
            "total_debt_outstanding": round(total_debt, 2),
            "num_active_loans": len(loan_products),
            "product_holdings": products,
            "preferred_channel": random.choices(CHANNELS, weights=channel_weights, k=1)[0],
            "secondary_channel": random.choices(CHANNELS, weights=[20, 20, 20, 20, 20], k=1)[0],
            "life_event": life_event["event"],
            "life_event_severity": life_event["severity"],
            "life_event_months_ago": life_event["months_ago"],
            "risk_segment": risk_segment,
            "is_stressed": is_stressed,
        }
        customers.append(customer)

    return customers


def save_customers_to_db(customers: List[Dict]):
    """Save generated customers to PostgreSQL."""
    conn = psycopg2.connect(
        host=PostgresConfig.HOST, port=PostgresConfig.PORT,
        user=PostgresConfig.USER, password=PostgresConfig.PASSWORD,
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
    stressed = [c for c in customers if c['is_stressed']]
    print(f"Generated {len(customers)} customer profiles")
    print(f"Stressed: {len(stressed)} ({len(stressed)/len(customers)*100:.1f}%)")
    print(f"Employment: {', '.join(set(c['employment_type'] for c in customers[:20]))}")
    print(f"Life events: {sum(1 for c in stressed if c['life_event'] != 'none')}/{len(stressed)} stressed")
    for c in customers[:3]:
        print(f"  {c['customer_id']}: {c['first_name']} {c['last_name']}, "
              f"Rs.{c['monthly_salary']:,.0f}/mo, {c['employment_type']}/{c['industry_sector']}, "
              f"DTI: {c['dti_ratio']:.1%}, Score: {c['credit_score']}, "
              f"Event: {c['life_event']}, Segment: {c['risk_segment']}")
