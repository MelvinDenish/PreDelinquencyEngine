# Demo Guide — Using n8n for Live Demonstrations

This guide explains how to use **n8n** (workflow automation tool) to create a visually impressive, automated demo of the Pre-Delinquency Intervention Engine.

---

## Why n8n for the Demo?

n8n gives you a **visual workflow editor** that shows data flowing through the pipeline in real-time. The judges can literally SEE the transactions being processed, risk scores computed, and interventions triggered — much more impressive than terminal output.

---

## Setup n8n

### Option 1: Docker (Recommended)

Add this to the bottom of your `docker-compose.yml` (before `volumes:` section):

```yaml
  pdi-n8n:
    image: n8nio/n8n:latest
    container_name: pdi-n8n
    ports:
      - "5678:5678"
    environment:
      N8N_BASIC_AUTH_ACTIVE: "true"
      N8N_BASIC_AUTH_USER: admin
      N8N_BASIC_AUTH_PASSWORD: admin
      WEBHOOK_URL: http://localhost:5678/
    volumes:
      - pdi-n8n-data:/home/node/.n8n
    networks:
      - pdi-network
```

And add `pdi-n8n-data:` to the `volumes:` section.

Then: `docker-compose up -d pdi-n8n`

Access n8n at: **http://localhost:5678**

### Option 2: npx (Quick Start)

```bash
npx n8n
```

Access at: http://localhost:5678

---

## Demo Workflow Design

### Workflow: "Pre-Delinquency Live Detection"

Create this workflow in n8n's visual editor:

```
┌──────────────┐     ┌──────────────────┐     ┌────────────────┐
│  ⏱ Schedule  │────►│  Generate Fake   │────►│  Push to        │
│  Trigger     │     │  Transaction     │     │  Scoring API    │
│  (every 5s)  │     │  (HTTP Request)  │     │  POST /score    │
└──────────────┘     └──────────────────┘     └───────┬────────┘
                                                      │
                                              ┌───────▼────────┐
                                              │  Check Risk    │
                                              │  Tier          │
                                              │  (IF node)     │
                                              └───┬───────┬────┘
                                                  │       │
                                    ┌─────────────▼─┐ ┌───▼──────────┐
                                    │ Risk < 0.5    │ │ Risk >= 0.5  │
                                    │ ✅ STABLE     │ │ ⚠️ WATCH/    │
                                    │ (No action)   │ │ 🔴 CRITICAL  │
                                    └───────────────┘ └──────┬───────┘
                                                             │
                                                     ┌───────▼───────┐
                                                     │  GenAI Message│
                                                     │  (via Groq    │
                                                     │   HTTP call)  │
                                                     └───────┬───────┘
                                                             │
                                              ┌──────────────▼────────┐
                                              │  Send Notification    │
                                              │  (Slack / Email /     │
                                              │   Discord webhook)    │
                                              └───────────────────────┘
```

---

## Step-by-Step n8n Workflow Setup

### Node 1: Schedule Trigger
- **Type**: Schedule Trigger
- **Interval**: Every 5 seconds
- **Purpose**: Fires every 5 seconds to simulate real-time transaction flow

### Node 2: Generate Transaction (Code Node)
- **Type**: Code (JavaScript)
- **Code**:
```javascript
const customers = [
  { id: "DEMO_STRESSED_001", name: "Rahul Verma" },
  { id: "DEMO_STRESSED_002", name: "Priya Sharma" },
  { id: "DEMO_HEALTHY_001", name: "Arjun Nair" },
];

const categories = ["lending_app", "gambling", "lottery", "grocery", "dining", "payday_lender"];
const customer = customers[Math.floor(Math.random() * customers.length)];
const category = categories[Math.floor(Math.random() * categories.length)];
const amount = Math.floor(Math.random() * 20000) + 500;

return [{
  json: {
    customer_id: customer.id,
    customer_name: customer.name,
    txn_type: "upi",
    merchant_category: category,
    amount: amount,
    direction: "debit",
    channel: "mobile",
    status: "success",
    timestamp: new Date().toISOString()
  }
}];
```

### Node 3: Score Customer (HTTP Request)
- **Type**: HTTP Request
- **Method**: POST
- **URL**: `http://host.docker.internal:8000/score` (if n8n is in Docker) or `http://localhost:8000/score` (if n8n is local)
- **Body (JSON)**:
```json
{
  "customer_id": "{{ $json.customer_id }}"
}
```

### Node 4: Check Risk (IF Node)
- **Type**: IF
- **Condition**: `{{ $json.risk_score }}` is greater than `0.5`

### Node 5: GenAI Message (HTTP Request — only for Watch/Critical)
- **Type**: HTTP Request
- **Method**: POST
- **URL**: `https://api.groq.com/openai/v1/chat/completions`
- **Headers**: `Authorization: Bearer YOUR_GROQ_API_KEY`
- **Body**:
```json
{
  "model": "llama-3.3-70b-versatile",
  "messages": [
    {"role": "system", "content": "Write a short, empathetic SMS intervention message for a bank customer showing financial stress."},
    {"role": "user", "content": "Customer {{ $('Generate Transaction').item.json.customer_name }} has risk score {{ $json.risk_score }}. Top driver: {{ $json.shap_drivers[0].feature }}. Write a 1-sentence SMS."}
  ],
  "max_tokens": 100
}
```

### Node 6: Notification (Slack / Discord / Email)

**Option A — Slack**:
- Type: Slack
- Channel: `#pdi-alerts`
- Message: `🔴 ALERT: {{ $('Generate Transaction').item.json.customer_name }} risk score {{ $('Score Customer').item.json.risk_score }} — {{ $json.choices[0].message.content }}`

**Option B — Discord Webhook**:
- Type: HTTP Request
- URL: Your Discord webhook URL
- Body: Same alert message

**Option C — Email**:
- Type: Send Email (Gmail/SMTP)
- To: demo@example.com
- Subject: PDI Alert — Customer at Risk

---

## What the Judges See During Demo

1. **n8n Dashboard** (http://localhost:5678) — showed on projector
   - Visual workflow with data flowing through nodes in real-time
   - Green/red indicators showing which path each transaction takes
   - Execution history showing every transaction processed

2. **Plotly Dash Dashboard** (http://localhost:8050) — second screen or tab
   - Risk heatmap updating as scores change
   - Customer detail views with SHAP waterfall charts

3. **Slack/Discord Channel** — on phone or third tab
   - Real-time intervention alerts with GenAI-generated messages
   - Shows the system actually doing outreach

---

## Alternative Demo Without n8n

If n8n is too complex to set up, use the built-in demo script:

```bash
# Terminal 1: Start services
docker-compose up -d
python main.py generate-data
python main.py scoring-service

# Terminal 2: Run live simulation
python demo/live_simulation.py
```

This prints real-time transaction flow with emojis, risk scores, and GenAI messages directly in the terminal.

---

## Recommended 10-Minute Demo Flow

| Time | What | Where |
|---|---|---|
| 0:00 | Architecture overview | Slides |
| 1:30 | Show Docker services running (16 containers) | Terminal |
| 2:30 | Show n8n workflow / live_simulation.py running | n8n / Terminal |
| 4:00 | Dashboard — risk heatmap updating | http://localhost:8050 |
| 5:30 | SHAP explanation for flagged customer | API `/score` response |
| 6:30 | Show intervention triggered + GenAI message | Slack/Terminal |
| 7:30 | Grafana metrics (latency, throughput) | http://localhost:3000 |
| 8:30 | MLflow model registry | http://localhost:5000 |
| 9:00 | Fairness audit results | Terminal |
| 10:00 | Q&A | — |
