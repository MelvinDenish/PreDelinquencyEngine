# PDI Engine — Session Handoff Context

> **Purpose**: Feed this file to a new Claude session so it knows exactly what's been done and what remains.
> **Project**: `c:\Users\lokes\OneDrive\Desktop\barclay\PreDelinquencyEngine\`
> **Date**: 2026-03-28

---

## What This Project Is

A **Barclays India Pre-Delinquency Intervention Engine** — an AI-powered credit risk scoring platform that predicts which customers will become delinquent before it happens and automatically dispatches personalized interventions (SMS, WhatsApp, Email, RM calls). Built for a Barclays hackathon; the user needs this to be **production-grade and impressive** to get a job at Barclays.

**Tech stack**: FastAPI backend, 4-model ML ensemble (XGBoost + LightGBM + LSTM + TFT), Kafka streaming, Redis caching, PostgreSQL, Cassandra, Celery workers, Next.js 16 frontend with Recharts, Docker/K8s deployment.

---

## What Has Been Completed

### Phase 1: Security Hardening (DONE)

All P0 critical security items from the plan (`C:\Users\lokes\.claude\plans\inherited-conjuring-cocke.md`) are implemented:

| File | What Was Done |
|------|---------------|
| `scoring_service/auth.py` | JWT (HS256) + API key (SHA-256) dual auth, RBAC with `require_role()` factory, 5 roles |
| `scoring_service/audit.py` | Audit event writer, PII masking (SHA-256 hash customer IDs, mask phone/email/name/salary) |
| `scoring_service/app.py` | Auth deps on all endpoints, CORS lockdown (no wildcard), `NotifyRequest` Pydantic model with HTML sanitization, `slowapi` rate limiting, audit calls, whatif_router included |
| `config/settings.py` | `_require_env()` for production, `SecurityConfig` class, Redis password/TLS support |
| `config/db.py` | Centralized PostgreSQL connection factory with SSL |
| `config/encryption.py` | Fernet field-level PII encryption with dev fallback |
| `config/vault.py` | Azure Key Vault client with ManagedIdentityCredential |
| `.env.example` | Full template with generation instructions for all secrets |
| `docker-compose.yml` | All hardcoded secrets → `${ENV_VAR:?error}` format |
| `init_db.sql` | Added: `users`, `api_keys`, `audit_log`, `customer_consent` tables + `ALTER TABLE interventions` |
| `Dockerfile` | Non-root `appuser` (UID 1001), .env files removed |
| `.github/workflows/ci.yml` | Removed `|| true`, added gitleaks, CodeQL SAST, custom secret grep |
| `k8s/network-policies/` | `default-deny-all.yaml` + `allow-services.yaml` |
| `k8s/security/peer-authentication.yaml` | Istio mTLS STRICT + AuthorizationPolicy |
| `k8s/ingress.yaml` | HTTPS with cert-manager, security headers, rate limiting |
| `k8s/secrets/external-secret.yaml` | ESO for Azure Key Vault (8 secrets) |
| `.gitleaks.toml` | Custom secret scanning rules |
| `ml/model_loader.py` | SHA-256 model integrity verification |
| `models/checksums.json` | Placeholder checksums (update after training) |
| `requirements.txt` | Added: python-jose, passlib, slowapi, bleach, cryptography, azure-keyvault-secrets, azure-identity |

### Phase 2: What-If Portfolio Simulator (DONE)

| File | What Was Done |
|------|---------------|
| `scoring_service/whatif.py` | FastAPI router with 6 scenario types (sector_shock, rate_hike, regional_shock, salary_shock, emi_holiday, custom), 8 pre-built templates, financial impact calculations in ₹ crore/lakh, RBAC protected |
| `frontend/src/app/whatif/page.tsx` | Full Barclays-branded simulator UI — scenario selector, parameter sliders, KPI cards, before/after distribution charts, segment breakdown, top affected employers, recommended actions |
| `frontend/src/app/api.ts` | Added What-If TypeScript interfaces and API functions |

### Phase 3: Frontend Polish (DONE)

| File | What Was Done |
|------|---------------|
| `frontend/src/app/globals.css` | Barclays design tokens (navy #002C6C, blue #00AEEF), radial gradient backgrounds, glass panels, brand badges |
| `frontend/src/app/layout.tsx` | Barclays metadata, security meta tags, theme-color |
| `frontend/src/app/page.tsx` | Barclays-branded header with wordmark, JWT SECURED badge, What-If nav link |

### Phase 4: RM Pre-Call AI Brief (DONE)

| File | What Was Done |
|------|---------------|
| `frontend/src/app/data.ts` | Added to Customer interface: `callBestTime`, `callAnswerRate`, `stressTrigger`, `stressCategory`, `callConversionToday`, `callConversionDelay`, `aiOpener`, `objections`, `guardrails`, `lifeEvent`. Populated for all 6 customers. |
| `frontend/src/app/page.tsx` | New `RMPreCallBrief` component with: stress trigger panel (color-coded by category), best call window with answer rate bar, positive outcome forecast (today vs delay), empathy-first opener, collapsible objection playbook, regulatory guardrails (DO NOT MENTION). Queue cards now show conversion probability. |

### Phase 5: Executive Dashboard Enhancements (DONE)

| File | What Was Done |
|------|---------------|
| `frontend/src/app/page.tsx` → `ExecutiveView` | Added: (1) Executive KPI bar (6 metrics: AUM, customers, NPA, protected AUM, interventions, response rate), (2) Employer Contagion Radar (6 employers with health scores, at-risk counts, signals — Byju's, Paytm, Zomato, Wipro, TCS, Infosys), (3) Early Warning Signals (6 macro indicators like UPI decline rate, lending app registrations, salary delays), (4) Portfolio Health Scorecard (SVG gauge 72.4/100, 8 pass/fail metrics) |

---

## What REMAINS To Be Done (In Progress When Session Ended)

### 1. `intervention/notification_dispatcher.py` — PII masking in logs (PARTIALLY DONE)

**Already done:**
- Added `_mask_email()`, `_mask_phone()`, `_check_consent()` functions at top of file
- Fixed email log line at line 172: `_mask_email(to_email)` ✅

**Still needs:**
- **Line ~198**: Change `logger.info(f"[SMS] ✅ Sent to {phone}, SID: {tw_msg.sid}")` → `logger.info(f"[SMS] ✅ Sent to {_mask_phone(phone)}, SID: {tw_msg.sid}")`
- **Line ~228**: Change `logger.info(f"[WhatsApp] ✅ Sent to {phone}, SID: {tw_msg.sid}")` → `logger.info(f"[WhatsApp] ✅ Sent to {_mask_phone(phone)}, SID: {tw_msg.sid}")`
- **In `dispatch_notification()` function (~line 504)**: Add consent check before each channel dispatch. Before the `for channel in channels_to_use:` loop or inside it, add:
  ```python
  if not _check_consent(customer_id, channel):
      logger.info(f"[Dispatcher] {customer_id} | {channel} skipped (no consent)")
      results.append({"channel": channel, "status": "skipped_no_consent", "customer_id": customer_id})
      continue
  ```
  Insert this right after the cooldown check (after the `if not _check_cooldown(...)` block, before `try:`).
- **In `assign_collector()` function (~line 369-376)**: The webhook payload includes raw PII (`customer_name`, `phone`). Replace with just `customer_id` and `assignment_id`.

### 2. `intervention/rules_engine.py` — `triggered_by` field (NOT DONE)

In the `save_intervention()` function (line ~211-242), the INSERT statement needs to include `triggered_by` and `trigger_ip` columns:

```python
# Current INSERT columns:
# (customer_id, intervention_type, channel, trigger_reason,
#  shap_drivers, risk_score_at_trigger, risk_tier_at_trigger, status)

# Should be:
# (customer_id, intervention_type, channel, trigger_reason,
#  shap_drivers, risk_score_at_trigger, risk_tier_at_trigger, status,
#  triggered_by, trigger_ip)

# And add two more %s placeholders + values:
# intervention.get("triggered_by", "system"),
# intervention.get("trigger_ip"),
```

### 3. `.github/workflows/cd.yml` — Trivy image scan (NOT DONE)

Add a Trivy container vulnerability scan step between "Build Docker image" and "Push Docker image" in the `build-and-push` job:

```yaml
      - name: Trivy vulnerability scan
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: '${{ env.DOCKER_REGISTRY }}/${{ github.repository }}/${{ env.DOCKER_IMAGE }}:${{ github.sha }}'
          format: 'table'
          exit-code: '1'
          severity: 'CRITICAL,HIGH'
          ignore-unfixed: true
```

Also remove the `:latest` tag push (security best practice — use only SHA-tagged images):
- Remove the line: `docker push ${IMAGE_TAG}:latest`
- Remove the line: `-t ${IMAGE_TAG}:latest` from the build step

### 4. `.github/dependabot.yml` (NOT CREATED)

Create this file:
```yaml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 5
    target-branch: "main"

  - package-ecosystem: "docker"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 3

  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 3
```

---

## Verification Checklist (After All Items Complete)

Run these to verify everything works:

1. `curl -X POST http://localhost:8000/score -d '{"customer_id":"C001"}'` → should return `401 Unauthorized`
2. `curl -X POST http://localhost:8000/auth/token -d 'username=admin&password=ChangeMe@2024!'` → returns JWT
3. `curl -X POST http://localhost:8000/score -H "Authorization: Bearer <token>" -d '{"customer_id":"C001"}'` → returns score
4. `curl -X POST http://localhost:8000/notify -d '{"alert_message":"<script>xss</script>"}'` → message is sanitized (401 without auth)
5. Check `audit_log` table after scoring → row present with hashed customer_id
6. `docker inspect <image>` → confirm non-root user
7. Frontend: `cd frontend && npm install && npm run dev` → visit localhost:3000
   - God Mode: live pipeline simulation
   - Executive: KPI bar, employer contagion radar, early warnings, health scorecard
   - RM View: click customer → AI Pre-Call Brief panel appears with stress trigger, call window, conversion forecast, objection playbook, guardrails
   - Customer: mobile simulator with nudge timeline
   - What-If: navigate via header link → run stress scenarios

---

## Key Architecture Decisions

- **JWT + API Key dual auth**: Bearer tokens for human users, X-API-Key header for service accounts
- **RBAC roles**: analyst, risk_officer, admin, read_only, service_account — mapped in `require_role()` dependency
- **Audit log uses SHA-256 hashed customer IDs** — never stores raw PII
- **Fernet encryption** for PII at rest (name, phone, email, salary)
- **What-If simulator uses existing risk scores** + mathematical sensitivity functions (not re-running 4 ML models) for near-instant simulation
- **Financial calculations use Indian banking figures**: avg loan ₹8.5L, crore/lakh denominations, RBI NPA thresholds
- **Frontend is entirely client-side rendered** (Next.js "use client") — works with or without the backend running (falls back to simulated data)

---

## File Structure Reference

```
PreDelinquencyEngine/
├── scoring_service/
│   ├── app.py          # FastAPI main — all endpoints with auth + rate limiting
│   ├── auth.py         # JWT + API key auth, RBAC
│   ├── audit.py        # Audit logging + PII masking
│   ├── whatif.py        # What-If simulator API router
│   └── cassandra_client.py
├── config/
│   ├── settings.py     # All config with _require_env() for production
│   ├── db.py           # Centralized PostgreSQL connection factory
│   ├── encryption.py   # Fernet PII encryption
│   └── vault.py        # Azure Key Vault client
├── intervention/
│   ├── rules_engine.py         # Risk rules + save_intervention()
│   ├── notification_dispatcher.py  # Multi-channel dispatch with consent + masking
│   └── genai_messages.py       # Groq LLM message generation
├── ml/
│   ├── model_loader.py  # SHA-256 model integrity checker
│   └── ...
├── frontend/
│   ├── src/app/
│   │   ├── page.tsx     # Main dashboard (God Mode, Executive, RM, Customer views)
│   │   ├── whatif/page.tsx  # What-If Simulator page
│   │   ├── api.ts       # API client functions
│   │   ├── data.ts      # Customer data with AI brief fields
│   │   ├── globals.css  # Barclays design tokens
│   │   └── layout.tsx   # Root layout with security headers
│   └── package.json     # Next.js 16, React 19, Recharts 3, Lucide React
├── k8s/
│   ├── network-policies/  # default-deny-all + allow-services
│   ├── security/          # Istio mTLS peer-authentication
│   ├── ingress.yaml       # HTTPS with cert-manager
│   └── secrets/           # External Secrets Operator manifests
├── .github/workflows/
│   ├── ci.yml          # Build + lint + security scans (gitleaks, CodeQL, bandit)
│   └── cd.yml          # Docker build + K8s deploy (needs Trivy scan added)
├── docker-compose.yml  # All secrets parameterized
├── Dockerfile          # Non-root appuser
├── init_db.sql         # All tables including security tables
├── requirements.txt    # Python deps including security packages
├── .env.example        # Env var template
├── .gitleaks.toml      # Custom secret scanning rules
└── models/checksums.json  # Model integrity checksums (placeholder)
```
