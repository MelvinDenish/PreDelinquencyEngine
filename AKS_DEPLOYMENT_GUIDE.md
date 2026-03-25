# 🚀 Azure Kubernetes Service (AKS) Deployment Guide

## Pre-Delinquency Intervention Engine — Production Deployment

---

## Prerequisites

Before starting, ensure you have:

| Tool | Install Command | Purpose |
|---|---|---|
| **Azure CLI** | `winget install Microsoft.AzureCLI` | Manage Azure resources |
| **kubectl** | `az aks install-cli` | Kubernetes CLI |
| **Docker Desktop** | Already installed | Build container images |
| **Helm** | `winget install Helm.Helm` | Package manager for K8s |
| **Azure Account** | [portal.azure.com](https://portal.azure.com) | Cloud subscription |

---

## Step-by-Step Deployment

### Step 1: Login to Azure

```powershell
# Login to Azure (opens browser)
az login

# Set your subscription (if you have multiple)
az account list --output table
az account set --subscription "YOUR_SUBSCRIPTION_ID"
```

### Step 2: Create a Resource Group

```powershell
# Create a resource group in Central India (closest to you)
az group create --name rg-pdi-engine --location centralindia
```

> **Why Central India?** Lowest latency for demo from Chennai. Alternative: `southeastasia` or `eastus`.

### Step 3: Create Azure Container Registry (ACR)

ACR stores your Docker images so AKS can pull them.

```powershell
# Create container registry (name must be globally unique, lowercase, no dashes)
az acr create --resource-group rg-pdi-engine --name pdiengineacr --sku Basic

# Login to ACR
az acr login --name pdiengineacr
```

### Step 4: Build & Push Docker Image to ACR

```powershell
# Navigate to project
cd "c:\Users\L Melvin Denish\barclays\PreDelinquencyEngine"

# Build the Docker image
docker build -t pdi-engine:latest .

# Tag for ACR
docker tag pdi-engine:latest pdiengineacr.azurecr.io/pdi-engine:latest

# Push to ACR
docker push pdiengineacr.azurecr.io/pdi-engine:latest
```

### Step 5: Create AKS Cluster

```powershell
# Create AKS cluster with 3 nodes
az aks create \
  --resource-group rg-pdi-engine \
  --name aks-pdi-engine \
  --node-count 3 \
  --node-vm-size Standard_D4s_v3 \
  --enable-managed-identity \
  --attach-acr pdiengineacr \
  --generate-ssh-keys \
  --network-plugin azure \
  --enable-addons monitoring \
  --location centralindia
```

> **Node size `Standard_D4s_v3`**: 4 vCPUs, 16GB RAM — enough for Kafka + Spark + ML models.
> For demo/hackathon, you can use `Standard_D2s_v3` (2 vCPU, 8GB) with `--node-count 2` to save cost.

**This takes ~5-10 minutes.**

### Step 6: Connect kubectl to AKS

```powershell
# Get AKS credentials (configures kubectl)
az aks get-credentials --resource-group rg-pdi-engine --name aks-pdi-engine

# Verify connection
kubectl get nodes
# Should show 3 nodes in "Ready" state
```

### Step 7: Create Kubernetes Secrets

Store sensitive credentials as K8s secrets (NOT in YAML files):

```powershell
# Create namespace first
kubectl apply -f k8s/namespace.yaml

# Create secrets for database credentials
kubectl create secret generic pdi-db-credentials \
  --namespace pdi-engine \
  --from-literal=POSTGRES_USER=pdi_user \
  --from-literal=POSTGRES_PASSWORD=pdi_password \
  --from-literal=POSTGRES_DB=pdi_db

# Create secrets for API keys
kubectl create secret generic pdi-api-keys \
  --namespace pdi-engine \
  --from-literal=GROQ_API_KEY=gsk_your_actual_key_here \
  --from-literal=SMTP_USER=your-email@gmail.com \
  --from-literal=SMTP_PASSWORD=your-app-password \
  --from-literal=TWILIO_ACCOUNT_SID=your_sid \
  --from-literal=TWILIO_AUTH_TOKEN=your_token

# Create secret for Airflow
kubectl create secret generic pdi-airflow \
  --namespace pdi-engine \
  --from-literal=AIRFLOW_FERNET_KEY=46BKJoQYlPPOexq0OhDZnIlNepKFf87WFwLt0nIe3aU=
```

### Step 8: Update K8s Manifests for AKS

You need to update the image references in your K8s YAML files to point to ACR:

**In every YAML file under `k8s/`, change:**
```yaml
# FROM:
image: pdi-engine:latest

# TO:
image: pdiengineacr.azurecr.io/pdi-engine:latest
```

**And add secret references for environment variables:**
```yaml
env:
  - name: POSTGRES_PASSWORD
    valueFrom:
      secretKeyRef:
        name: pdi-db-credentials
        key: POSTGRES_PASSWORD
  - name: GROQ_API_KEY
    valueFrom:
      secretKeyRef:
        name: pdi-api-keys
        key: GROQ_API_KEY
```

### Step 9: Deploy Infrastructure Services

Deploy services in order (dependencies first):

```powershell
# 1. PostgreSQL (database must be up first)
kubectl apply -f k8s/postgres.yaml

# Wait for PostgreSQL to be ready
kubectl wait --for=condition=ready pod -l app=pdi-postgres -n pdi-engine --timeout=120s

# 2. Redis (cache)
kubectl apply -f k8s/redis.yaml

# 3. Kafka (message broker)
kubectl apply -f k8s/kafka.yaml

# Wait for Kafka to be ready
kubectl wait --for=condition=ready pod -l app=pdi-kafka -n pdi-engine --timeout=120s
```

### Step 10: Initialize the Database

```powershell
# Get the PostgreSQL pod name
$PG_POD = kubectl get pods -n pdi-engine -l app=pdi-postgres -o jsonpath='{.items[0].metadata.name}'

# Copy init_db.sql into the pod
kubectl cp init_db.sql pdi-engine/${PG_POD}:/tmp/init_db.sql

# Run the SQL to create tables
kubectl exec -n pdi-engine $PG_POD -- psql -U pdi_user -d pdi_db -f /tmp/init_db.sql
```

### Step 11: Deploy Application Services

```powershell
# 4. Scoring Service (FastAPI — the main API)
kubectl apply -f k8s/scoring-service.yaml

# 5. Dashboard (Plotly Dash)
kubectl apply -f k8s/dashboard.yaml

# 6. Celery Worker (async intervention processing)
kubectl apply -f k8s/celery-worker.yaml
```

### Step 12: Verify All Pods Are Running

```powershell
# Check all pods
kubectl get pods -n pdi-engine -o wide

# Expected output:
# NAME                              READY   STATUS    RESTARTS   AGE
# pdi-postgres-xxx                  1/1     Running   0          5m
# pdi-redis-xxx                     1/1     Running   0          4m
# pdi-kafka-xxx                     1/1     Running   0          4m
# scoring-service-xxx               1/1     Running   0          2m
# scoring-service-yyy               1/1     Running   0          2m  (2 replicas)
# dashboard-xxx                     1/1     Running   0          2m
# celery-worker-xxx                 1/1     Running   0          2m

# Check services (get external IPs)
kubectl get services -n pdi-engine
```

### Step 13: Get External Access URLs

```powershell
# Get the scoring service external IP
kubectl get service pdi-scoring -n pdi-engine -o jsonpath='{.status.loadBalancer.ingress[0].ip}'

# Get the dashboard external IP
kubectl get service pdi-dashboard -n pdi-engine -o jsonpath='{.status.loadBalancer.ingress[0].ip}'
```

> **Note:** LoadBalancer IPs take 2-3 minutes to provision. If it shows `<pending>`, wait and retry.

Once you have the IPs:
- **Scoring API**: `http://<SCORING_IP>:8000/docs` (Swagger UI)
- **Dashboard**: `http://<DASHBOARD_IP>:8050`

### Step 14: Generate Data & Train Models on AKS

```powershell
# Get a scoring service pod name
$SCORING_POD = kubectl get pods -n pdi-engine -l app=scoring-service -o jsonpath='{.items[0].metadata.name}'

# Generate synthetic data
kubectl exec -n pdi-engine $SCORING_POD -- python main.py generate-data

# Run stream processing (processes existing transactions)
kubectl exec -n pdi-engine $SCORING_POD -- python main.py stream-process &

# Train ML models
kubectl exec -n pdi-engine $SCORING_POD -- python main.py train

# Score all customers
kubectl exec -n pdi-engine $SCORING_POD -- python main.py score-all
```

### Step 15: Set Up Ingress (Optional — Custom Domain)

For a clean URL like `pdi.yourdomain.com` instead of raw IPs:

```powershell
# Install NGINX Ingress Controller
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm install nginx-ingress ingress-nginx/ingress-nginx --namespace pdi-engine

# Create ingress rule
kubectl apply -f - <<EOF
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: pdi-ingress
  namespace: pdi-engine
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  ingressClassName: nginx
  rules:
    - host: pdi-api.yourdomain.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: pdi-scoring
                port:
                  number: 8000
    - host: pdi-dashboard.yourdomain.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: pdi-dashboard
                port:
                  number: 8050
EOF
```

---

## 💰 Cost Estimation (Azure)

| Resource | Size | Monthly Cost (approx) |
|---|---|---|
| AKS Cluster (3 × D4s_v3) | 4 vCPU, 16GB each | ~₹18,000 ($215) |
| AKS Cluster (2 × D2s_v3) | 2 vCPU, 8GB each | ~₹6,000 ($72) |
| Azure Container Registry | Basic | ~₹400 ($5) |
| Load Balancer | Standard | ~₹1,500 ($18) |
| **Total (Demo/Hackathon)** | **2 nodes** | **~₹8,000/month** |
| **Total (Production)** | **3 nodes** | **~₹20,000/month** |

> **Free trial tip**: Azure free account gives ₹15,000 ($200) credits for 30 days. That covers the entire hackathon.

---

## 🔧 Troubleshooting

### Pod stuck in CrashLoopBackOff
```powershell
kubectl logs <pod-name> -n pdi-engine --previous
kubectl describe pod <pod-name> -n pdi-engine
```

### Pod stuck in Pending (insufficient resources)
```powershell
# Scale up nodes
az aks scale --resource-group rg-pdi-engine --name aks-pdi-engine --node-count 4
```

### Can't pull image from ACR
```powershell
# Re-attach ACR to AKS
az aks update --resource-group rg-pdi-engine --name aks-pdi-engine --attach-acr pdiengineacr
```

### Database connection refused
```powershell
# Check PostgreSQL is running
kubectl get pods -n pdi-engine -l app=pdi-postgres
# Ensure service name matches env var: POSTGRES_HOST=pdi-postgres
```

---

## 🧹 Cleanup (After Hackathon)

```powershell
# DELETE EVERYTHING (stops billing immediately)
az group delete --name rg-pdi-engine --yes --no-wait
```

> **This deletes**: AKS cluster, ACR, load balancers, all data. Run this after the hackathon to avoid charges.

---

## Quick Reference Card

| Action | Command |
|---|---|
| Check pods | `kubectl get pods -n pdi-engine` |
| Check logs | `kubectl logs <pod> -n pdi-engine -f` |
| Shell into pod | `kubectl exec -it <pod> -n pdi-engine -- bash` |
| Scale scoring | `kubectl scale deployment scoring-service -n pdi-engine --replicas=5` |
| Get external IPs | `kubectl get svc -n pdi-engine` |
| Delete cluster | `az group delete --name rg-pdi-engine --yes` |
