# Pre-Delinquency Intervention Engine

> **Barclays Hackathon** | College of Engineering Guindy, Chennai
>
> AI-driven, event-driven platform that predicts payment defaults **2–4 weeks in advance** using real-time transaction streams, ML ensemble (XGBoost + LightGBM + LSTM), SHAP explainability, GenAI-powered interventions (Groq), and proactive intervention routing.

📄 **[Full System Documentation](SYSTEM_ARCHITECTURE.md)** | 🐛 **[Troubleshooting Guide](TROUBLESHOOTING.md)** | 🎬 **[Demo Guide (n8n)](DEMO_GUIDE.md)**

---

## Architecture

```
Transactions (UPI/ATM/Cards)
        │
        ▼
  Debezium CDC ──► Apache Kafka ──► Apache Flink ──► Redis (Feast Online)
                                           │                    │
                                      Enrichment +     Feature Assembly
                                    Rolling Features           │
                                           │                    ▼
  PostgreSQL ◄── Spark Batch ◄── Airflow    └──► FastAPI Scoring Service
       │                                              │  (XGBoost+LightGBM+LSTM)
  Feast Offline                               SHAP + LIME Explainability
       │                                              │
  MLflow Registry ◄────────────────────────── Ensemble Risk Score
                                                      │
                             ┌────────────────────────┤
                             │                        │
                     Cassandra (history)        Intervention Engine
                     Redis (fast lookup)              │
                             │               Python Rules Engine
                             │               Celery + Apprise
                             ▼                (SMS/Email/App/RM)
                    Plotly Dash Dashboard              │
                    Apache Superset                    ▼
                    Grafana + Prometheus       Kafka Feedback Loop
                             │                        │
                             └────────► Evidently AI ─┘
                                        Airflow Retraining
                                        AIF360 + Fairlearn (Bias)
```

---

## Prerequisites

| Requirement | Minimum Version | Notes |
|---|---|---|
| **Docker Desktop** | 4.x | Enable WSL2 backend on Windows |
| **Docker Compose** | v2.x | Bundled with Docker Desktop |
| **RAM** | 16 GB | All services need ~12 GB allocated to Docker |
| **Disk** | 20 GB free | Images + volumes |
| **Python** | 3.10+ | Only needed to run pipeline commands locally |

> **Docker Desktop RAM setting**: Open Docker Desktop → Settings → Resources → Memory → set to at least **12 GB**.

---

## How to Start the Project (Docker)

### Step 1 — Clone and Configure

```bash
git clone <repo-url>
cd PreDelinquencyEngine

# Copy environment file and review settings
copy .env .env.local    # Windows
# cp .env .env.local    # Linux/Mac
```

### Step 2 — Start All Infrastructure Services

```bash
docker compose up -d
```

This starts **all services** in the correct dependency order:
- Zookeeper → Kafka → PostgreSQL → Redis
- Flink (JobManager + TaskManager)
- Spark (Master + Worker)
- Debezium CDC Connector
- MLflow Model Registry
- Apache Airflow (Scheduler + Webserver)
- Apache Cassandra
- ClickHouse *(optional data warehouse)*
- Prometheus + Grafana
- Apache Superset

Wait ~2 minutes for all services to become healthy, then verify:

```bash
docker compose ps
```

All services should show **Up** or **healthy**.

### Step 3 — Install Python Dependencies Locally

> **Note on Windows**: `apache-flink`, `apache-airflow`, and `cassandra-driver` require C compilers and may fail to install locally. **This is fine** — they run inside Docker containers. Skip them with the command below.

```bash
pip install setuptools wheel

# Install all packages except those that only run inside Docker
pip install psycopg2-binary sqlalchemy redis kafka-python pydantic python-dotenv faker numpy pandas scikit-learn requests joblib httpx xgboost lightgbm shap torch fastapi uvicorn celery apprise plotly dash dash-bootstrap-components evidently mlflow prometheus-fastapi-instrumentator prometheus-client groq

# Optional: TensorFlow (for alternate LSTM model)
pip install tensorflow
```

### Step 4 — Verify Core Services are Healthy

```bash
# Check Kafka is ready
docker exec pdi-kafka kafka-topics --bootstrap-server localhost:9092 --list

# Check PostgreSQL
docker exec pdi-postgres pg_isready -U pdi_user -d pdi_db

# Check Redis
docker exec pdi-redis redis-cli ping
# Expected: PONG

# Check Cassandra (takes ~60s on first start)
docker exec pdi-cassandra cqlsh -e "DESCRIBE KEYSPACES;"

# Check Airflow is up
curl http://localhost:8085/health
```

### Step 5 — Run the Full Pipeline

Option A — **Run locally** (recommended — faster, avoids Docker build):

```bash
python main.py full-pipeline
```

Option B — **Run inside Docker container**:

```bash
# Build the app image first
docker compose build pdi-app

# Generate synthetic data (1000+ customers, 6 months of transactions)
docker compose run --rm pdi-app python main.py generate-data

# Run stream feature computation
docker compose run --rm pdi-app python main.py stream-process

# Run Spark batch feature computation
docker compose run --rm pdi-app python main.py batch-process

# Set up Feast feature store (apply + materialize)
docker compose run --rm pdi-app python main.py feast-setup

# Train ML models (XGBoost + LightGBM + LSTM)
docker compose run --rm pdi-app python main.py train
```

Option C — **Single command full pipeline** (inside Docker):

```bash
docker compose run --rm pdi-app python main.py full-pipeline
```

### Step 6 — Start the Scoring Service

```bash
# Run locally
python main.py scoring-service

# OR run in Docker
docker compose run --rm -p 8000:8000 pdi-app python main.py scoring-service

# Verify it's running
# Open http://localhost:8000/health in browser
```

### Step 7 — Score All Customers

```bash
python main.py score-all
```

### Step 8 — Start the Dashboard

```bash
python main.py dashboard
# Open http://localhost:8050
```

### Step 9 — Start Background Services (Optional)

```bash
# Celery worker (async intervention delivery)
python main.py celery-worker

# Feedback event consumer (closes the loop for retraining)
python main.py feedback-consumer
```

---

## Running a Demo

### Option 1: Live Simulation Script (Quickest)

```bash
# Terminal 1: Start scoring service
python main.py scoring-service

# Terminal 2: Run live simulation (5 demo customers, real-time txns)
python demo/live_simulation.py
```

Shows real-time transactions with emojis, risk scores updating, and GenAI-generated intervention messages.

### Option 2: n8n Visual Workflow

See **[DEMO_GUIDE.md](DEMO_GUIDE.md)** for a complete n8n setup that shows data flowing through the pipeline visually — much more impressive for judges.

### GenAI Setup (Optional — Recommended)

To enable AI-generated intervention messages:
1. Get a free API key from https://console.groq.com/keys
2. Edit `.env`: `GROQ_API_KEY=gsk_your_key_here`
3. The system falls back to static templates if the key is not set.

---

## All Pipeline Commands

| Command | Description |
|---|---|
| `python main.py infra-up` | Start Docker infrastructure |
| `python main.py infra-down` | Stop Docker infrastructure |
| `python main.py generate-data` | Generate 1000+ customer profiles + 6 months of transactions |
| `python main.py stream-process` | Start Flink stream processor (real-time features) |
| `python main.py batch-process` | Run Spark batch feature computation |
| `python main.py feast-setup` | Apply Feast definitions + materialize to Redis |
| `python main.py train` | Train XGBoost + LightGBM + LSTM models |
| `python main.py scoring-service` | Start FastAPI scoring service (port 8000) |
| `python main.py score-all` | Score all customers via the scoring service |
| `python main.py dashboard` | Start Plotly Dash dashboard (port 8050) |
| `python main.py celery-worker` | Start Celery worker for async interventions |
| `python main.py feedback-consumer` | Start Kafka feedback event consumer |
| `python main.py full-pipeline` | Run complete pipeline (generate → process → train) |

---

## Service URLs

| Service | URL | Credentials |
|---|---|---|
| **Scoring API** | http://localhost:8000 | — |
| **API Docs (Swagger)** | http://localhost:8000/docs | — |
| **Prometheus Metrics** | http://localhost:8000/metrics | — |
| **Dashboard (Dash)** | http://localhost:8050 | — |
| **Flink Web UI** | http://localhost:8081 | — |
| **Spark Web UI** | http://localhost:8082 | — |
| **MLflow UI** | http://localhost:5000 | — |
| **Airflow UI** | http://localhost:8085 | admin / admin |
| **Grafana** | http://localhost:3000 | admin / pdi_admin |
| **Apache Superset** | http://localhost:8088 | admin / admin |
| **Prometheus** | http://localhost:9090 | — |
| **ClickHouse HTTP** | http://localhost:8123 | pdi_user / pdi_password |

---

## Complete Tech Stack

| Layer | Technology |
|---|---|
| **Data Ingestion** | Apache Kafka, Debezium (CDC) |
| **Stream Processing** | Apache Flink (PyFlink) |
| **Batch Processing** | Apache Spark, Apache Airflow |
| **Feature Store** | Feast — Redis (online), PostgreSQL (offline) |
| **Data Warehouse** | ClickHouse / PostgreSQL |
| **ML Training** | XGBoost, LightGBM, PyTorch (LSTM), TensorFlow (LSTM), scikit-learn |
| **Explainability** | SHAP, LIME |
| **GenAI** | Groq API (Llama 3.3 70B) — personalized intervention messages |
| **Model Serving** | FastAPI, BentoML, MLflow |
| **Risk Score Storage** | Redis (fast lookup), Apache Cassandra (history) |
| **Alerting & Intervention** | Python Rules Engine, Celery |
| **Outreach Channels** | Apprise (SMS, Email, App, RM Call) |
| **Visualization** | Plotly Dash, Apache Superset, Grafana |
| **Monitoring** | Prometheus, Grafana, Evidently AI (drift) |
| **Bias & Fairness** | Fairlearn, AIF360 |
| **Infrastructure** | Docker, Docker Compose, Kubernetes (k8s/) |

---

## Dashboard Views

1. **Portfolio Risk Heatmap** — Customer distribution across Critical / Watch / Stable tiers, filterable by region, product, and segment
2. **Trending Customers** — Fastest-deteriorating risk scores with top stress signals (e.g., "salary delayed 9 days, savings down 18%")
3. **Intervention Tracker** — Outreach logs, channel, timestamp, response rates, and outcomes
4. **Customer Deep-Dive** — Individual timeline: income patterns, spending shifts, balance trends, risk score history
5. **Model Health Monitor** — Live precision/recall/AUC with drift detection alerts and retraining logs

---

## Stopping the Project

```bash
# Stop all containers (keep data volumes)
docker compose down

# Stop and delete all data volumes (full reset)
docker compose down -v
```

---

## Troubleshooting

See **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** for a complete guide covering 16+ common errors with step-by-step solutions, including:

- Port conflicts
- Kafka connection issues
- Cassandra slow startup
- Python pip build failures on Windows
- Model training / scoring errors
- Docker memory issues
- Groq API errors
- And more
