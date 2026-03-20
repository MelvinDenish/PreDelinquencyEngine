# Pre-Delinquency Intervention Engine — Complete System Documentation

**Barclays Hackathon | College of Engineering Guindy, Chennai**

---

## Table of Contents
1. System Overview
2. Architecture Diagram
3. Data Sources & How to Feed Fake Data
4. Component-by-Component Deep Dive
5. How Debezium CDC Works
6. Inter-Component Communication Flow
7. How to Use This System in Production
8. Full Data Flow (End-to-End Walkthrough)

---

## 1. System Overview

The Pre-Delinquency Intervention Engine predicts payment defaults **2–4 weeks before they happen** by analysing real-time transaction streams and historical behavioural patterns. When risk thresholds are breached, it automatically routes tailored interventions through each customer's preferred channel.

**Core principle**: Shift from reactive (after missed payment) to proactive (before the miss).

---

## 2. Architecture Diagram

```
                          ┌─────────────────────────────────────┐
                          │        DATA SOURCES                 │
                          │  (PostgreSQL tables in production    │
                          │   OR fake data generator locally)   │
                          └─────────────┬───────────────────────┘
                                        │
                     ┌──────────────────┼──────────────────┐
                     │                  │                  │
                     ▼                  ▼                  ▼
              ┌──────────┐     ┌──────────────┐    ┌────────────┐
              │ Debezium │     │ Kafka        │    │ Direct     │
              │ CDC      │────►│ Topics       │◄───│ Producer   │
              │ (live DB │     │              │    │ (fake data │
              │  changes)│     │ transactions │    │  generator)│
              └──────────┘     │ account_upd  │    └────────────┘
                               │ risk_scores  │
                               │ interventions│
                               │ feedback     │
                               └──────┬───────┘
                                      │
                 ┌────────────────────┼────────────────────┐
                 ▼                    ▼                    ▼
          ┌────────────┐     ┌──────────────┐     ┌──────────────┐
          │ Flink      │     │ Spark Batch  │     │ Feedback     │
          │ Stream     │     │ Processing   │     │ Consumer     │
          │ Processing │     │ (via Airflow)│     │              │
          │            │     │              │     │              │
          │ • Enrich   │     │ • Baselines  │     │ • Outcome    │
          │ • Rolling  │     │ • Demographics│    │   labeling   │
          │   features │     │ • Salary     │     │ • Retrain    │
          │   (7d/30d) │     │   delay calc │     │   trigger    │
          └──────┬─────┘     └──────┬───────┘     └──────┬───────┘
                 │                  │                     │
                 ▼                  ▼                     │
          ┌─────────────────────────────────┐             │
          │     FEAST FEATURE STORE         │             │
          │  ┌───────────┐  ┌────────────┐  │             │
          │  │ Redis     │  │ PostgreSQL │  │             │
          │  │ (online)  │  │ (offline)  │  │             │
          │  │ Latest    │  │ Historical │  │             │
          │  │ features  │  │ features   │  │             │
          │  └────┬──────┘  └────────────┘  │             │
          └───────┼─────────────────────────┘             │
                  │                                       │
                  ▼                                       │
          ┌─────────────────────────────────┐             │
          │   SCORING SERVICE (FastAPI)     │             │
          │                                 │             │
          │  1. Fetch features from Redis   │             │
          │  2. Run XGBoost + LightGBM      │             │
          │  3. Run LSTM on time series     │             │
          │  4. Combine (weighted ensemble) │             │
          │  5. SHAP explainability         │             │
          │  6. Map to risk tier            │             │
          │  7. Store to Redis + Cassandra  │             │
          │  8. Expose /metrics (Prometheus)│             │
          └──────┬──────────────────────────┘             │
                 │                                        │
                 ▼                                        │
          ┌─────────────────────────────────┐             │
          │  INTERVENTION ENGINE            │             │
          │                                 │             │
          │  • Rules Engine (SHAP-driven)   │             │
          │  • Tier transition detection    │             │
          │  • Channel optimization         │◄────────────┘
          │  • Cooldown (7-day)             │     (outcomes fed
          │  • Celery async delivery        │      back for
          │  • Apprise notifications        │      retraining)
          └──────┬──────────────────────────┘
                 │
                 ▼
          ┌─────────────────────────────────┐
          │  VISUALIZATION & MONITORING     │
          │                                 │
          │  • Plotly Dash (dashboard)       │
          │  • Grafana + Prometheus (ops)    │
          │  • Apache Superset (analytics)  │
          │  • Evidently AI (drift detect)  │
          │  • MLflow (model registry)      │
          └─────────────────────────────────┘
```

---

## 3. Data Sources & How to Feed Fake Data

### What Data Sources Are Needed

In a **production bank** environment, the system would connect to:

| Data Source | What It Contains | How It Gets In |
|---|---|---|
| **Core Banking System** | Account balances, customer profiles, product holdings | Debezium CDC on the core banking DB |
| **UPI Gateway** | UPI transactions (peer-to-peer, merchant payments) | Debezium CDC or direct Kafka feed from gateway |
| **ATM Network** | Cash withdrawals, balance inquiries | Debezium CDC on ATM transaction tables |
| **Card Processing** | Credit/debit card transactions | Debezium CDC on card transaction DB |
| **Bill Payment System** | Utility payments, loan EMIs, auto-debits | Debezium CDC on scheduled payment tables |
| **Credit Bureau** | Credit scores, loan history | Batch import via Spark (periodic refresh) |
| **CRM System** | Customer interaction history, preferred channel | Batch import or API call |

### Since You Don't Have a Bank's Database — How to Feed Fake Data

The project has a **built-in synthetic data generator** (`data_generation/generator.py`) that creates realistic fake data. Here's how it works:

#### What the Generator Creates

```python
python main.py generate-data
```

This single command generates:

1. **1000 Customer Profiles** (configurable via `.env`):
   - Name, age, tenure, credit score, product holdings
   - 20% are flagged as "stressed" customers (will show delinquency patterns)

2. **6 Months of Transaction History** (~180,000 transactions):
   - UPI, ATM, card, auto-debit, and bill payments
   - Merchant categories: grocery, dining, entertainment, lending_app, utilities, etc.
   - Stressed customers get patterns like: increasing lending app usage, rising ATM withdrawals, missed auto-debits

3. **Account Balance Snapshots** (daily snapshots):
   - Savings account balance trends
   - Stressed customers show declining balances

4. **All data is written to TWO places**:
   - **PostgreSQL** (for batch processing and Debezium CDC)
   - **Kafka topics** (for immediate stream processing)

#### How to Feed Custom Fake Data

You can also feed individual transactions in real-time via the Kafka producer:

```python
from ingestion.kafka_producer import PDIKafkaProducer

producer = PDIKafkaProducer()

# Feed a single transaction
producer.publish_transaction({
    "txn_id": "TXN_CUSTOM_001",
    "customer_id": "CUST_0042",
    "txn_type": "upi",
    "merchant_category": "lending_app",      # ← stress signal!
    "amount": 15000.00,
    "direction": "debit",
    "channel": "mobile",
    "status": "success",
    "timestamp": "2026-03-19T10:30:00"
})

# Feed an account balance update
producer.publish_account_update({
    "customer_id": "CUST_0042",
    "balance": 5000.00,               # ← low balance = stress
    "previous_balance": 25000.00,
    "timestamp": "2026-03-19T10:30:00"
})

producer.flush()
producer.close()
```

The stream processor will immediately pick up these events, compute updated features, and trigger re-scoring.

---

## 4. Component-by-Component Deep Dive

### 4.1 Data Ingestion Layer

**Files**: `ingestion/kafka_producer.py`, `ingestion/kafka_consumer.py`, `debezium/register_connector.py`

**What it does**: Gets data from source systems into Kafka topics.

**Two ingestion paths**:

| Path | Source | Mechanism | Use Case |
|---|---|---|---|
| **Path 1: Debezium CDC** | PostgreSQL database tables | Watches DB transaction log (WAL) for INSERT/UPDATE/DELETE | Production: captures real-time changes from bank's core systems |
| **Path 2: Direct Kafka Producer** | Application code | Python `kafka-python` library publishes JSON events | Development: fake data generator pushes synthetic transactions |

**Kafka Topics Created**:
- `transactions` — Every financial transaction (UPI, ATM, card, auto-debit)
- `account_updates` — Balance changes on savings/current accounts
- `risk_scores` — Computed risk scores (downstream consumers)
- `interventions` — Intervention decisions
- `feedback_events` — Outcome of interventions (paid, defaulted, restructured)

**Key Design**: Customer ID is used as the Kafka partition key, so all events for the same customer go to the same partition, preserving chronological order.

---

### 4.2 Stream Processing (Apache Flink)

**File**: `stream_processing/flink_job.py`

**What it does**: Continuously consumes transactions from Kafka and computes real-time features over sliding windows.

**Two execution modes**:

1. **Flink Cluster Mode** (`create_flink_job()`): Runs on the Docker Flink cluster (JobManager + TaskManager). Uses PyFlink SQL to define Kafka sources, JDBC sinks, and tumbling window aggregations.

2. **Local Fallback Mode** (`create_flink_job_local()`): Pure Python process that uses `kafka-python` consumer + in-memory state. Used when Flink cluster is unavailable.

**Enrichment Step**: Each transaction is enriched with a **merchant risk score** from a lookup table:
- `grocery` → 0.1 (low risk)
- `dining` → 0.2
- `payday_lender` → 0.9 (high risk!)
- `cash_advance` → 0.85

**Features Computed (per customer, rolling 7-day and 30-day windows)**:

| Feature | What It Measures |
|---|---|
| `discretionary_spend_7d/30d` | Total spending on dining, entertainment, travel, clothing, luxury |
| `atm_withdrawals_count_7d/30d` | Number of ATM cash withdrawals |
| `lending_app_txn_count_7d/30d` | Count of transactions to lending apps, payday lenders |
| `weighted_lending_risk_7d/30d` | Sum of merchant risk scores for lending transactions |
| `failed_autodebits_count_7d/30d` | Number of failed auto-debit attempts (missed payments) |
| `total_spend_7d/30d` | Total debit spend across all categories |
| `txn_count_7d/30d` | Total transaction count |
| `avg_txn_amount_7d` | Average transaction amount |
| `max_txn_amount_7d` | Largest single transaction |

**Output**: Features are written to:
- **Redis** (Feast online store) — for instant retrieval during scoring
- **PostgreSQL** (via JDBC sink in Flink) — for historical storage

---

### 4.3 Batch Processing (Spark + Airflow)

**Files**: `batch_processing/spark_job.py`, `airflow/dags/`

**What it does**: Daily batch jobs compute features that don't change quickly and require historical lookout.

**Features Computed**:

| Feature | Calculation |
|---|---|
| `salary_delay_days` | Current day minus expected salary credit day (from historical pattern) |
| `utility_payment_delay_avg` | Average days past due for utility bills over last 3–6 months |
| `discretionary_spend_trend` | Compare last 7 days spend vs. same period last month |
| `credit_score` | From customer profile (would come from credit bureau in prod) |
| `age`, `tenure_months` | Customer demographics |
| `product_count`, `has_credit_card`, `has_personal_loan`, `has_mortgage` | Product holdings |
| `avg_monthly_spend_3m` | Average monthly spend over 3 months |
| `spend_volatility_3m` | Standard deviation of monthly spend (instability indicator) |

**Airflow orchestrates**:
1. Run Spark batch feature computation
2. Write to Feast offline store (PostgreSQL)
3. Materialize Feast features to online store (Redis)
4. Trigger model retraining when needed

---

### 4.4 Feature Store (Feast)

**Files**: `feature_store/feature_repo/`

**What it does**: Central registry for all feature definitions. Ensures the same features used in training are used in serving (no training–serving skew).

**Two stores**:
- **Online Store (Redis)**: Latest feature values for every customer. Updated by Flink (streaming) and Feast materialization (batch). Used for real-time scoring.
- **Offline Store (PostgreSQL)**: Historical feature values with timestamps. Used for training data export and point-in-time joins.

**How scoring uses it**:
```python
feature_vector = store.get_online_features(
    feature_refs=["transaction_features:discretionary_spend_7d", ...],
    entity_rows=[{"customer_id": "CUST_0042"}]
).to_dict()
```

---

### 4.5 ML Models (XGBoost + LightGBM + LSTM Ensemble)

**Files**: `ml/xgboost_model.py`, `ml/lightgbm_model.py`, `ml/lstm_model.py`, `ml/ensemble.py`

**Three models, one purpose**:

| Model | Input | What It Captures | Weight |
|---|---|---|---|
| **XGBoost** | 29 tabular features | Current financial state, nonlinear interactions | 40% |
| **LightGBM** | 29 tabular features | Complementary tree-based prediction | 30% |
| **LSTM** | Sequence of feature vectors (30-day window) | Temporal deterioration patterns (gradual decline over time) | 30% |

**Ensemble combination**:
```
final_score = 0.40 × XGBoost_prob + 0.30 × LightGBM_prob + 0.30 × LSTM_prob
```

**Risk Tier Mapping**:
- `final_score > 0.7` → **Critical** (red alert)
- `final_score > 0.5` → **Watch** (yellow)
- `final_score ≤ 0.5` → **Stable** (green)

**Credit Score Mapping**: `credit_score = 850 - (final_score × 550)`

---

### 4.6 Explainability (SHAP + LIME)

**Files**: `ml/explainability.py`, `ml/lime_explainer.py`

**What it does**: For every prediction, SHAP identifies which features contributed most to the risk score.

**Example output**:
```
Top Risk Drivers for CUST_0042:
  1. salary_delay_days = 12    (SHAP: +0.18)  ← salary is 12 days late!
  2. savings_balance_pct_change_7d = -0.35 (SHAP: +0.15)  ← savings dropped 35%
  3. lending_app_txn_count_7d = 5 (SHAP: +0.09)  ← using lending apps
```

These SHAP drivers directly determine the **type of intervention** (see Rules Engine below).

---

### 4.7 Scoring Service (FastAPI)

**File**: `scoring_service/app.py`

**Endpoints**:

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check — returns model load status |
| `/score` | POST | Score a single customer — returns risk score, tier, SHAP explanation |
| `/score/batch` | POST | Score multiple customers at once |
| `/explain/{customer_id}` | GET | Get SHAP/LIME explanation for a customer |
| `/metrics` | GET | Prometheus metrics (request count, latency, etc.) |
| `/docs` | GET | Interactive Swagger API documentation |

**What happens when you call `/score`**:
1. Receive `customer_id`
2. Fetch latest features from Redis (via Feast)
3. Run XGBoost → probability
4. Run LightGBM → probability
5. Run LSTM → probability
6. Combine via weighted ensemble → final risk score
7. Compute SHAP explanation → top 5 risk drivers
8. Map to risk tier (stable/watch/critical)
9. Store result in Redis + Cassandra (time-series history)
10. Return JSON response with score, tier, explanation

---

### 4.8 Risk Score Storage (Redis + Cassandra)

**Files**: `scoring_service/cassandra_client.py`

**Dual storage strategy**:

| Store | Purpose | TTL |
|---|---|---|
| **Redis** | Fast lookup of latest score | 7 days |
| **Cassandra** | Full time-series history of all scores | 90 days |

**Cassandra stores in 3 tables**:
- `risk_scores` — Full history per customer (partition key = customer_id, clustering key = timestamp)
- `risk_scores_latest` — Latest score per customer (fast lookup)
- `risk_scores_by_tier` — All customers in a given risk tier (for dashboard filtering)

---

### 4.9 Intervention Engine (Rules Engine + Celery)

**File**: `intervention/rules_engine.py`

**Signal-Aware Routing** — SHAP explanations determine the action:

| Top SHAP Driver | Intervention Type | Description |
|---|---|---|
| `salary_delay_days` | Payment Holiday | Offer payment holiday during salary disruption |
| `savings_balance_pct_change_7d` | EMI Restructuring | Lower monthly EMI burden |
| `lending_app_txn_count_7d` | Wellness Check-In | Financial counselling and support |
| `discretionary_spend_7d` | Budget Nudge | Spending awareness with budgeting tips |
| `failed_autodebits_count_7d` | EMI Restructuring | Restructure failed auto-debit payments |
| `utility_payment_delay_avg` | Payment Reminder | Friendly reminder before due date |
| Score > 0.85 + Critical tier | Escalation Call | RM direct outreach call |

**Key behaviors**:
- **Tier Transition Alerts**: Only fires on tier changes (e.g., stable→watch), not on every score update
- **7-Day Cooldown**: Won't re-contact the same customer within 7 days
- **Escalation**: If risk worsens with no response: nudge → SMS → RM call
- **Channel Optimization**: Uses customer's historically highest-engagement channel

**Celery + Apprise**: Interventions are dispatched asynchronously via Celery workers. Apprise handles multi-channel delivery (SMS, email, app push, webhook).

---

### 4.10 Feedback Loop & Model Retraining

**Files**: `feedback/feedback_consumer.py`, `feedback/drift_detector.py`

**How the loop works**:

1. **Outcome Labeling**: After each intervention, the CRM (or manual process) publishes a `feedback_event` to Kafka:
   ```json
   {
     "customer_id": "CUST_0042",
     "intervention_id": 123,
     "outcome": "paid",
     "timestamp": "2026-03-25T14:00:00"
   }
   ```

2. **Feedback Consumer**: Reads from `feedback_events` Kafka topic, updates the labels table in PostgreSQL (Feast offline store).

3. **Drift Detection** (Evidently AI): Compares current prediction distributions against baseline. If AUC drops > 5%, flags for retraining.

4. **Scheduled Retraining** (Airflow): Bi-weekly DAG runs:
   - Exports latest labeled data from Feast offline store
   - Trains new XGBoost + LightGBM + LSTM models
   - Runs fairness checks (AIF360, Fairlearn)
   - Registers as challenger model in MLflow
   - Promotes to champion only if it outperforms on precision, recall, and fairness

---

### 4.11 Visualization & Monitoring

**Dashboard (Plotly Dash)** — `dashboard/app.py`:
- Portfolio Risk Heatmap (customer distribution across tiers)
- Trending Customers (fastest-deteriorating scores)
- Intervention Tracker (outreach logs, response rates)
- Customer Deep-Dive (individual timeline)
- Model Health Monitor (precision, recall, AUC, drift)

**Grafana + Prometheus** — `monitoring/prometheus.yml`:
- Scrapes `/metrics` from the scoring service
- Tracks request latency, error rates, throughput
- Infrastructure metrics (Kafka lag, Redis memory, PostgreSQL connections)

**Apache Superset** — Interactive analytics on historical data

**MLflow** — Model registry, experiment tracking, artifact storage

---

## 5. How Debezium CDC Works

### What is Debezium?

Debezium is a **Change Data Capture (CDC)** platform. It watches a database's transaction log (PostgreSQL WAL) and publishes every INSERT, UPDATE, and DELETE as a Kafka event — **without modifying the application code**.

### How It's Configured in This Project

**File**: `debezium/register_connector.py`

When you run `python debezium/register_connector.py`, it sends a REST API call to the Debezium Connect service (running in Docker on port 8083) with this configuration:

```
Connector: PostgreSQL CDC
Plugin: pgoutput (PostgreSQL logical decoding)
Tables Monitored: customers, transactions, account_balances,
                  risk_scores, interventions, feedback_events
Topic Prefix: pdi.cdc
```

**What happens after registration**:

1. Debezium creates a PostgreSQL logical replication slot (`pdi_debezium_slot`)
2. It takes an **initial snapshot** of all monitored tables (sends all existing rows as INSERT events)
3. After the snapshot, it **continuously streams** new changes from WAL

**Kafka topics created by Debezium**:
- `pdi.cdc.customers` — customer profile changes
- `pdi.cdc.transactions` — new transactions inserted
- `pdi.cdc.account_balances` — balance updates
- `pdi.cdc.risk_scores` — computed scores
- `pdi.cdc.interventions` — intervention decisions
- `pdi.cdc.feedback_events` — outcome feedback

**Each event looks like**:
```json
{
  "customer_id": "CUST_0042",
  "txn_type": "upi",
  "amount": 5000.00,
  "timestamp": "2026-03-19T10:30:00",
  "__op": "c",                    // c=create, u=update, d=delete
  "__table": "transactions",
  "__source_ts_ms": 1710848200000 // original DB timestamp
}
```

### How to Use Debezium in Production

1. **Point Debezium at the bank's PostgreSQL** (or MySQL/Oracle):
   ```python
   # Change config in debezium/register_connector.py:
   "database.hostname": "core-banking-db.bank.internal",
   "database.port": "5432",
   "database.user": "cdc_reader",
   "database.dbname": "core_banking",
   ```

2. **Monitor the tables you care about**:
   ```python
   "table.include.list": "banking.transactions,banking.accounts,banking.balances"
   ```

3. **Every time a teller enters a new transaction or an ATM processes a withdrawal**, Debezium captures it in under 100ms and publishes to Kafka.

4. **Your stream processor (Flink) picks it up immediately**, computes updated features, and the scoring service re-evaluates risk.

### Why CDC Instead of Direct API?

| Approach | Latency | Impact on Source | Data Completeness |
|---|---|---|---|
| **Polling** (SELECT every N seconds) | Seconds to minutes | Heavy DB load | May miss rapid changes |
| **Direct API hooks** | Immediate | Requires code changes in every system | Depends on developer discipline |
| **Debezium CDC** | Sub-second | **Zero** — reads from WAL, not the table | **Complete** — captures everything, even batch jobs |

---

## 6. Inter-Component Communication

### Communication Protocol Table

| From | To | Protocol | Channel |
|---|---|---|---|
| Data Sources → | Kafka | TCP (Kafka protocol) | Kafka topics |
| Debezium → | Kafka | TCP (Kafka Connect) | CDC topics |
| Kafka → | Flink | TCP (Kafka consumer) | Stream processing |
| Kafka → | Feedback Consumer | TCP (Kafka consumer) | feedback_events topic |
| Flink → | Redis | TCP (Redis protocol) | Feature writes |
| Flink → | PostgreSQL | TCP (JDBC) | Feature writes |
| Spark → | PostgreSQL | TCP (JDBC) | Batch feature writes |
| Airflow → | Spark | HTTP (Spark submit) | Job orchestration |
| Airflow → | Feast | Python SDK | Materialization |
| Scoring Service → | Redis (Feast) | TCP (Redis protocol) | Feature reads |
| Scoring Service → | Models (in-memory) | Function call | Inference |
| Scoring Service → | Cassandra | TCP (CQL protocol) | Score history writes |
| Scoring Service → | Prometheus | HTTP (`/metrics` scrape) | Metrics |
| Rules Engine → | Redis | TCP (Redis protocol) | Tier tracking, cooldown |
| Rules Engine → | PostgreSQL | TCP (psycopg2) | Intervention save |
| Celery Worker → | Redis | TCP (broker) | Task queue |
| Celery Worker → | Apprise | HTTPS (API calls) | SMS/email/push delivery |
| Dashboard → | PostgreSQL | TCP (psycopg2) | Data reads |
| Dashboard → | Redis | TCP (Redis protocol) | Live feature reads |
| Grafana → | Prometheus | HTTP | Metrics visualization |
| MLflow → | PostgreSQL | TCP | Model metadata |
| MLflow → | File system | Local/S3 | Model artifacts |

---

## 7. How to Use This System in Production

### Step-by-Step Production Deployment

1. **Replace fake data with real data sources**:
   - Point Debezium at the bank's core banking database
   - Configure table names matching the bank's schema
   - Map column names to the expected fields

2. **Scale infrastructure**:
   - Deploy on Kubernetes (k8s/ manifests are included)
   - Kafka: 3+ brokers with replication factor 3
   - Cassandra: 3+ nodes for durability
   - Redis: Sentinel or cluster for HA

3. **Set up model retraining pipeline**:
   - Airflow DAG runs bi-weekly
   - Trains on latest labeled data
   - Runs fairness checks before promotion
   - Champion-challenger testing in shadow mode

4. **Connect notification channels**:
   - Configure Apprise with bank's SMS gateway, email server, app push service
   - Set up preferred channel database per customer

5. **Enable monitoring**:
   - Grafana dashboards for infrastructure health
   - Prometheus alerts for scoring service latency
   - Evidently AI reports for model drift

### What You'd Need to Change for Your Bank

| Component | Current (Development) | Production |
|---|---|---|
| Data source | Fake generator + local PostgreSQL | Debezium CDC on core banking DB |
| Kafka | Single broker, no replication | 3+ brokers, replication = 3 |
| Feature Store | Local Redis + PG | Managed Redis cluster + Aurora PG |
| ML Models | Trained on fake data | Retrained on real labelled data |
| Notifications | Apprise (stdout) | Bank's SMS, email, app push systems |
| Auth | None | OAuth2 / API keys |
| HTTPS | None | TLS everywhere |

---

## 8. Full Data Flow — End-to-End Walkthrough

**Scenario**: Customer CUST_0042 takes a loan from a payday lending app.

### Step 1: Transaction Enters the System
```
Core Banking DB (INSERT INTO transactions) ───► PostgreSQL WAL
                                                     │
                                              Debezium CDC
                                                     │
                                                     ▼
                                              Kafka topic: "transactions"
                                              Key: "CUST_0042"
                                              Payload: {txn_type: "upi",
                                                        merchant: "lending_app",
                                                        amount: 15000, ...}
```

### Step 2: Flink Enriches and Computes Features
```
Flink reads from Kafka
    │
    ├── Lookup merchant_risk_score: lending_app → 0.9 (high risk!)
    │
    ├── Update rolling window state for CUST_0042:
    │     lending_app_txn_count_7d: 3 → 4
    │     weighted_lending_risk_7d: 2.5 → 3.4
    │
    └── Write updated features to Redis (Feast online store)
```

### Step 3: Scoring Service Scores the Customer
```
Trigger: New features written (or scheduled cron)
    │
    ├── Fetch 29 features from Redis for CUST_0042
    │
    ├── XGBoost says: P(delinquency) = 0.68
    ├── LightGBM says: P(delinquency) = 0.72
    ├── LSTM says: P(delinquency) = 0.65
    │
    ├── Ensemble: 0.40(0.68) + 0.30(0.72) + 0.30(0.65) = 0.683
    │
    ├── SHAP top drivers:
    │     1. lending_app_txn_count_7d = 4  (SHAP: +0.12)
    │     2. salary_delay_days = 8          (SHAP: +0.09)
    │     3. savings_balance_pct_change = -0.22 (SHAP: +0.07)
    │
    ├── Risk tier: "watch" (0.5 < 0.683 < 0.7)
    │
    └── Store: Redis (latest) + Cassandra (history)
```

### Step 4: Intervention Engine Decides Action
```
Rules Engine receives score event:
    │
    ├── Check tier transition: CUST_0042 was "stable", now "watch" ← TRANSITION!
    │
    ├── Top SHAP driver: lending_app_txn_count_7d
    │     → SHAP_INTERVENTION_MAP: "wellness_checkin"
    │
    ├── Check cooldown: No intervention in last 7 days ← OK
    │
    ├── Determine channel: CUST_0042's best response rate = "app" push
    │
    └── Create intervention: "Financial wellness check-in"
        → Celery queues async delivery
        → Apprise sends app push notification
        → Logged in PostgreSQL interventions table
```

### Step 5: Feedback Closes the Loop
```
2 weeks later, CUST_0042 makes their payment on time.

CRM publishes to Kafka topic "feedback_events":
    {customer_id: "CUST_0042", outcome: "paid"}
    │
    Feedback Consumer reads event
    │
    ├── Updates labels table in PostgreSQL
    │     → This intervention was successful!
    │
    └── Next time Airflow retrains models,
        this positive outcome is included,
        improving future predictions.
```

---

## Service Ports Quick Reference

| Service | Port | URL |
|---|---|---|
| Scoring API | 8000 | http://localhost:8000 |
| Dashboard | 8050 | http://localhost:8050 |
| Flink UI | 8081 | http://localhost:8081 |
| Spark UI | 8082 | http://localhost:8082 |
| Debezium Connect | 8083 | http://localhost:8083 |
| Airflow UI | 8085 | http://localhost:8085 |
| MLflow | 5000 | http://localhost:5000 |
| Grafana | 3000 | http://localhost:3000 |
| Superset | 8088 | http://localhost:8088 |
| Prometheus | 9090 | http://localhost:9090 |
| Kafka (internal) | 9092 | pdi-kafka:9092 |
| Kafka (external) | 29092 | localhost:29092 |
| Cassandra CQL | 9042 | localhost:9042 |
| PostgreSQL | 5432 | localhost:5432 |
| Redis | 6379 | localhost:6379 |
| ClickHouse HTTP | 8123 | http://localhost:8123 |

---

*Document generated for Barclays Hackathon — College of Engineering Guindy, Chennai*
