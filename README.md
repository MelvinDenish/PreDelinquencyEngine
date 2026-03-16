# Pre-Delinquency Intervention Engine

Enterprise-grade system that predicts payment defaults 2–4 weeks in advance using real-time transaction streams, ML models (XGBoost + LSTM ensemble), and proactive interventions.

## Architecture

```
Transactions → Kafka → Flink (enrichment + rolling features) → Redis
                                                                  ↕
PostgreSQL ← Spark (batch features) ← Airflow                  Feast
                                                                  ↕
                FastAPI Scoring Service ← XGBoost + LSTM + SHAP
                         ↓
              Intervention Engine → Celery → Apprise (SMS/Email/App/RM)
                         ↓
              Plotly Dash Dashboard (5 views)
                         ↓
              Feedback Loop → Evidently AI → Airflow Retraining
```

## Quick Start

### 1. Start Infrastructure
```bash
docker-compose up -d
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Full Pipeline
```bash
python main.py full-pipeline
```

### 4. Start Scoring Service
```bash
python main.py scoring-service
```

### 5. Score All Customers
```bash
python main.py score-all
```

### 6. Start Dashboard
```bash
python main.py dashboard
# Open http://localhost:8050
```

## All Commands

| Command | Description |
|---------|-------------|
| `python main.py infra-up` | Start Docker infrastructure |
| `python main.py infra-down` | Stop Docker infrastructure |
| `python main.py generate-data` | Generate 1000+ customer profiles and 6 months of transactions |
| `python main.py stream-process` | Start Flink stream processor |
| `python main.py batch-process` | Run Spark batch feature computation |
| `python main.py feast-setup` | Set up Feast feature store |
| `python main.py train` | Train XGBoost + LSTM models |
| `python main.py scoring-service` | Start FastAPI scoring service (port 8000) |
| `python main.py score-all` | Score all customers via scoring service |
| `python main.py dashboard` | Start Plotly Dash dashboard (port 8050) |
| `python main.py celery-worker` | Start Celery worker for async interventions |
| `python main.py feedback-consumer` | Start Kafka feedback event consumer |
| `python main.py full-pipeline` | Run complete pipeline (generate → process → train) |

## Tech Stack

- **Data Ingestion**: Apache Kafka, Debezium-style CDC
- **Stream Processing**: Apache Flink (Docker cluster)
- **Batch Processing**: Apache Spark (Docker cluster), Apache Airflow
- **Feature Store**: Feast, Redis (online), PostgreSQL (offline)
- **ML Training**: XGBoost, LSTM (PyTorch), scikit-learn
- **Explainability**: SHAP
- **Model Serving**: FastAPI, MLflow
- **Alerting**: Python Rules Engine, Celery
- **Outreach**: Apprise (SMS, Email, App, RM Call)
- **Dashboard**: Plotly Dash
- **Monitoring**: Evidently AI (drift detection)
- **Fairness**: Fairlearn, AIF360
- **Infrastructure**: Docker, Docker Compose

## Dashboard Views

1. **Portfolio Risk Heatmap** – Customer distribution across risk tiers, filterable by region and income
2. **Trending Customers** – Fastest-deteriorating risk scores with top stress signals
3. **Intervention Tracker** – Outreach logs, channels, outcomes, and effectiveness
4. **Customer Deep-Dive** – Individual timeline: income, spending, balance, risk history
5. **Model Health Monitor** – Prediction metrics, drift detection, retraining logs

## Ports

| Service | Port |
|---------|------|
| Kafka | 9092 |
| PostgreSQL | 5432 |
| Redis | 6379 |
| Flink Web UI | 8081 |
| Spark Web UI | 8082 |
| Scoring API | 8000 |
| Dashboard | 8050 |
