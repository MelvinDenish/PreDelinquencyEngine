# Troubleshooting Guide

Common errors and their solutions when running the Pre-Delinquency Intervention Engine.

---

## Docker & Container Issues

### 1. Port Already in Use

**Error**: `Bind for 0.0.0.0:5432 failed: port is already allocated`

**Solution**: Another service is using the port. Stop it, or change the port mapping:
```bash
# Find what's using port 5432
netstat -ano | findstr 5432          # Windows
lsof -i :5432                       # Mac/Linux

# Stop local PostgreSQL if it's running
net stop postgresql                  # Windows
sudo systemctl stop postgresql       # Linux
brew services stop postgresql        # Mac

# OR change the port in docker-compose.yml:
#   ports:
#     - "5433:5432"   # Use 5433 externally
# Then update .env: POSTGRES_PORT=5433
```

### 2. Docker Containers Fail to Start (Dependency Order)

**Error**: Services exit immediately because dependencies aren't ready yet.

**Solution**: Some services (Airflow, Superset, Debezium) need PostgreSQL/Kafka/Redis to be fully healthy first. Docker healthchecks handle this, but on slow machines:
```bash
# Start infrastructure first, wait, then start the rest
docker-compose up -d pdi-postgres pdi-redis pdi-zookeeper
# Wait 30 seconds
docker-compose up -d pdi-kafka
# Wait 30 seconds
docker-compose up -d
```

### 3. Kafka Not Ready / Connection Refused

**Error**: `NoBrokersAvailable` or `Connection refused to localhost:29092`

**Solution**:
```bash
# Check if Kafka is healthy
docker-compose ps pdi-kafka

# If unhealthy, restart
docker-compose restart pdi-kafka

# Wait 30s, then verify
docker exec pdi-kafka kafka-topics --bootstrap-server localhost:9092 --list
```

**If running Python locally** (not in Docker): ensure `.env` has:
```
KAFKA_BOOTSTRAP_SERVERS=localhost:29092
```

**If running inside Docker**: use `pdi-kafka:9092` (the internal hostname).

### 4. Cassandra Slow to Start

**Error**: `cassandra-driver` connection timeout or `NoHostAvailable`

**Cause**: Cassandra typically takes 60–90 seconds to start.

**Solution**: Just wait. Check status:
```bash
docker logs pdi-cassandra --tail 20
# Look for "Startup complete" message
```

### 5. Superset "Admin already exists" Warning

**Error**: `FAB ERROR: User admin already exists`

**Cause**: Normal on restart — it tries to create the admin user each time.

**Solution**: Ignore this. It's a harmless warning, not an error. The login still works. Credentials: `admin / admin`.

### 6. Airflow Database Migration Errors

**Error**: `relation "ab_user" already exists` or migration errors

**Cause**: Airflow DB already migrated from a previous run.

**Solution**: Ignore. These are safe warnings. If Airflow truly won't start:
```bash
docker-compose down pdi-airflow
docker volume rm predelinquencyengine_pdi-airflow-logs
docker-compose up -d pdi-airflow
```

### 7. ClickHouse "ulimits" Error on Windows

**Error**: `Error response from daemon: failed to create shim task: OCI runtime create failed`

**Solution**: Remove or reduce the `ulimits` section in `docker-compose.yml` for `pdi-clickhouse`:
```yaml
# Comment out or remove these lines:
# ulimits:
#   nofile:
#     soft: 262144
#     hard: 262144
```

Then restart: `docker-compose up -d pdi-clickhouse`

---

## Python / Pip Issues

### 8. `apache-flink` Fails to Install on Windows

**Error**: `error: Microsoft Visual C++ 14.0 or greater is required`

**Cause**: `apache-flink` needs C compilation tools that aren't standard on Windows.

**Solution**: Skip it locally — Flink runs in Docker. The local stream processor uses a pure Python fallback:
```bash
pip install -r requirements.txt --ignore-installed apache-flink apache-airflow
# Or install everything except problematic packages:
pip install psycopg2-binary sqlalchemy redis kafka-python pydantic python-dotenv faker
pip install xgboost lightgbm torch scikit-learn numpy pandas
pip install fastapi uvicorn shap lime mlflow groq
```

### 9. `cassandra-driver` Build Error

**Error**: `error: command 'cl.exe' failed: No such file or directory`

**Solution**: Same as above — skip it locally. Cassandra client runs inside Docker:
```bash
pip install cassandra-driver || echo "Skipping - runs in Docker"
```

### 10. TensorFlow Import Error

**Error**: `ModuleNotFoundError: No module named 'tensorflow'`

**Solution**:
```bash
pip install tensorflow
# If you're on Apple Silicon Mac:
pip install tensorflow-macos tensorflow-metal
```

The system works without TensorFlow — the TF LSTM model is optional. The PyTorch LSTM runs by default.

---

## Pipeline Execution Issues

### 11. "Table does not exist" During Data Generation

**Error**: `relation "customers" does not exist`

**Cause**: The database tables haven't been created yet.

**Solution**: The `init_db.sql` runs automatically when PostgreSQL starts for the first time. If it didn't:
```bash
# Run init script manually
docker exec -i pdi-postgres psql -U pdi_user -d pdi_db < init_db.sql
```

### 12. Model Training Fails with "Not enough samples"

**Error**: `ValueError: The number of classes has to be greater than one`

**Cause**: Not enough diverse training data (all labels are the same class).

**Solution**: Generate more data with a higher stress percentage:
```bash
# In .env, increase:
NUM_CUSTOMERS=2000
STRESS_CUSTOMER_PCT=0.25

# Re-generate
python main.py generate-data
```

### 13. Scoring Service Returns 500 Error

**Error**: `Internal Server Error` on `/score`

**Cause**: Usually means models aren't trained yet, or features aren't in Redis.

**Solution**:
```bash
# 1. Generate data first
python main.py generate-data

# 2. Run stream processing to compute features
python main.py stream-process

# 3. Train models
python main.py train

# 4. Then start scoring service
python main.py scoring-service
```

### 14. Groq API Returns Error

**Error**: `groq.AuthenticationError` or empty `GROQ_API_KEY`

**Solution**:
1. Get a free API key from https://console.groq.com/keys
2. Add to `.env`: `GROQ_API_KEY=gsk_your_key_here`
3. Restart the app

If Groq is unavailable, the system **falls back to static templates** automatically — GenAI is optional.

---

## Network & Connectivity Issues

### 15. Services Can't Reach Each Other

**Error**: `Connection refused` between Docker services

**Solution**: Ensure all services are on the same Docker network:
```bash
docker network inspect predelinquencyengine_pdi-network
# All containers should be listed

# If not, restart all services:
docker-compose down
docker-compose up -d
```

### 16. WSL2/Windows Docker Memory Issues

**Error**: Containers killed with OOM (Out of Memory)

**Solution**: Increase Docker Desktop memory allocation:
1. Docker Desktop → Settings → Resources → Advanced
2. Set Memory to at least **6 GB** (8 GB recommended)
3. Set CPUs to at least **4**
4. Click "Apply & restart"

If still running out of memory, start only essential services:
```bash
# Minimal set for demo
docker-compose up -d pdi-postgres pdi-redis pdi-zookeeper pdi-kafka pdi-prometheus pdi-grafana
```

---

## Quick Health Check Commands

```bash
# Check all container statuses
docker-compose ps

# Check logs for a specific service
docker logs pdi-kafka --tail 50
docker logs pdi-postgres --tail 50

# Verify PostgreSQL
docker exec pdi-postgres psql -U pdi_user -d pdi_db -c "SELECT count(*) FROM customers;"

# Verify Kafka topics exist
docker exec pdi-kafka kafka-topics --bootstrap-server localhost:9092 --list

# Verify Redis
docker exec pdi-redis redis-cli ping

# Verify Scoring Service (run locally)
curl http://localhost:8000/health

# Verify Dashboard (run locally)
curl http://localhost:8050
```
