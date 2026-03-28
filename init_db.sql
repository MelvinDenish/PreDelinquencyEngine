-- ============================================================
-- Pre-Delinquency Intervention Engine - Database Schema
-- ============================================================

-- Customers table
CREATE TABLE IF NOT EXISTS customers (
    customer_id VARCHAR(50) PRIMARY KEY,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    age INTEGER NOT NULL,
    gender VARCHAR(10),
    city VARCHAR(100),
    state VARCHAR(100),
    region VARCHAR(50),
    income_bracket VARCHAR(30),
    monthly_salary DECIMAL(12,2),
    salary_credit_day INTEGER,  -- expected day of month for salary
    tenure_months INTEGER,
    credit_score INTEGER,
    product_holdings TEXT[],  -- array of product names
    preferred_channel VARCHAR(20) DEFAULT 'sms',  -- sms, email, app, rm_call
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Account balances
CREATE TABLE IF NOT EXISTS account_balances (
    id SERIAL PRIMARY KEY,
    customer_id VARCHAR(50) REFERENCES customers(customer_id),
    balance DECIMAL(15,2) NOT NULL,
    savings_balance DECIMAL(15,2) DEFAULT 0,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Transactions table (historical + new)
CREATE TABLE IF NOT EXISTS transactions (
    txn_id VARCHAR(100) PRIMARY KEY,
    customer_id VARCHAR(50) REFERENCES customers(customer_id),
    txn_type VARCHAR(30) NOT NULL,  -- upi, atm, bill_payment, salary_credit, emi, lending_app, auto_debit
    merchant_category VARCHAR(50),
    merchant_id VARCHAR(100),
    amount DECIMAL(12,2) NOT NULL,
    direction VARCHAR(10) NOT NULL,  -- credit, debit
    channel VARCHAR(20),  -- upi, atm, netbanking, auto
    status VARCHAR(15) DEFAULT 'success',  -- success, failed
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    merchant_risk_score DECIMAL(5,3),
    risk_category VARCHAR(20)
);

-- Streaming features (written by Flink)
CREATE TABLE IF NOT EXISTS streaming_features (
    customer_id VARCHAR(50) PRIMARY KEY,
    discretionary_spend_7d DECIMAL(12,2) DEFAULT 0,
    discretionary_spend_30d DECIMAL(12,2) DEFAULT 0,
    atm_withdrawals_count_7d INTEGER DEFAULT 0,
    atm_withdrawals_count_30d INTEGER DEFAULT 0,
    lending_app_txn_count_7d INTEGER DEFAULT 0,
    lending_app_txn_count_30d INTEGER DEFAULT 0,
    weighted_lending_risk_7d DECIMAL(8,4) DEFAULT 0,
    weighted_lending_risk_30d DECIMAL(8,4) DEFAULT 0,
    savings_balance_pct_change_7d DECIMAL(8,4) DEFAULT 0,
    failed_autodebits_count_7d INTEGER DEFAULT 0,
    failed_autodebits_count_30d INTEGER DEFAULT 0,
    total_spend_7d DECIMAL(12,2) DEFAULT 0,
    total_spend_30d DECIMAL(12,2) DEFAULT 0,
    txn_count_7d INTEGER DEFAULT 0,
    txn_count_30d INTEGER DEFAULT 0,
    avg_txn_amount_7d DECIMAL(12,2) DEFAULT 0,
    max_txn_amount_7d DECIMAL(12,2) DEFAULT 0,
    -- M4: Domain-specific banking distress signals
    salary_to_emi_gap_days INTEGER DEFAULT 0,          -- days between salary credit and EMI debit (shrinking = stress)
    upi_failure_rate_7d DECIMAL(8,4) DEFAULT 0,        -- failed UPI / total UPI (rising = insufficient balance)
    weekend_atm_ratio_7d DECIMAL(8,4) DEFAULT 0,       -- weekend ATM / total ATM (high = desperation cash)
    min_balance_velocity_30d DECIMAL(12,2) DEFAULT 0,   -- rate of change of monthly minimum balance
    late_night_txn_ratio_7d DECIMAL(8,4) DEFAULT 0,     -- 11PM-5AM transactions / total (behavioral distress)
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Batch features (written by Spark)
CREATE TABLE IF NOT EXISTS batch_features (
    customer_id VARCHAR(50) PRIMARY KEY,
    salary_delay_days INTEGER DEFAULT 0,
    utility_payment_delay_avg DECIMAL(8,2) DEFAULT 0,
    discretionary_spend_trend DECIMAL(8,4) DEFAULT 0,  -- ratio: current / previous period
    credit_score INTEGER,
    age INTEGER,
    tenure_months INTEGER,
    income_bracket VARCHAR(30),
    region VARCHAR(50),
    gender VARCHAR(10),
    product_count INTEGER DEFAULT 0,
    has_credit_card BOOLEAN DEFAULT FALSE,
    has_personal_loan BOOLEAN DEFAULT FALSE,
    has_mortgage BOOLEAN DEFAULT FALSE,
    avg_monthly_spend_3m DECIMAL(12,2) DEFAULT 0,
    spend_volatility_3m DECIMAL(8,4) DEFAULT 0,
    -- M2: Asset-side features
    fd_closed_count_90d INTEGER DEFAULT 0,
    fd_closure_amount_90d DECIMAL(12,2) DEFAULT 0,
    sip_stopped_flag BOOLEAN DEFAULT FALSE,
    sip_gaps_3m INTEGER DEFAULT 0,
    insurance_lapse_flag BOOLEAN DEFAULT FALSE,
    insurance_missed_payments_3m INTEGER DEFAULT 0,
    -- M3: Employer health
    employer_health_score DECIMAL(5,4) DEFAULT 0,
    employer_payroll_delay_avg DECIMAL(8,2) DEFAULT 0,
    employer_headcount_change_pct DECIMAL(8,2) DEFAULT 0,
    -- M1: Customer segment
    segment_type VARCHAR(30) DEFAULT 'salaried',
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Risk scores
CREATE TABLE IF NOT EXISTS risk_scores (
    id SERIAL PRIMARY KEY,
    customer_id VARCHAR(50) REFERENCES customers(customer_id),
    risk_score DECIMAL(5,4) NOT NULL,
    risk_tier VARCHAR(20) NOT NULL,  -- critical, watch, stable
    credit_score_mapped INTEGER,
    xgboost_score DECIMAL(5,4),
    lightgbm_score DECIMAL(5,4),
    lstm_score DECIMAL(5,4),
    tft_score DECIMAL(5,4),
    ensemble_score DECIMAL(5,4),
    meta_learner_used BOOLEAN DEFAULT FALSE,
    segment_type VARCHAR(30),
    confidence_flag VARCHAR(20) DEFAULT 'full',  -- full, cold_start, limited_history
    -- M5: Survival / time-to-event
    p30d_default DECIMAL(5,4),
    p60d_default DECIMAL(5,4),
    median_tte_days INTEGER,
    top_shap_features JSONB,
    tft_attention_weights JSONB,  -- per-timestep attention weights
    model_version VARCHAR(50),
    scored_at TIMESTAMP DEFAULT NOW()
);

-- Interventions
CREATE TABLE IF NOT EXISTS interventions (
    id SERIAL PRIMARY KEY,
    customer_id VARCHAR(50) REFERENCES customers(customer_id),
    intervention_type VARCHAR(50) NOT NULL,  -- payment_holiday, emi_restructuring, wellness_checkin, nudge
    channel VARCHAR(20) NOT NULL,  -- sms, email, app, rm_call
    trigger_reason TEXT,
    shap_drivers JSONB,
    risk_score_at_trigger DECIMAL(5,4),
    risk_tier_at_trigger VARCHAR(20),
    status VARCHAR(20) DEFAULT 'sent',  -- sent, delivered, responded, ignored
    outcome VARCHAR(20),  -- paid, restructured, defaulted, no_response
    sent_at TIMESTAMP DEFAULT NOW(),
    responded_at TIMESTAMP,
    cooldown_until TIMESTAMP
);

-- Feedback events
CREATE TABLE IF NOT EXISTS feedback_events (
    id SERIAL PRIMARY KEY,
    customer_id VARCHAR(50) REFERENCES customers(customer_id),
    intervention_id INTEGER REFERENCES interventions(id),
    outcome VARCHAR(20) NOT NULL,  -- paid, restructured, defaulted, no_response
    label INTEGER,  -- 0 = no delinquency, 1 = delinquent
    event_timestamp TIMESTAMP DEFAULT NOW()
);

-- Model registry tracking
CREATE TABLE IF NOT EXISTS model_registry (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    model_type VARCHAR(30),  -- xgboost, lstm, ensemble
    metrics JSONB,
    is_champion BOOLEAN DEFAULT FALSE,
    trained_at TIMESTAMP DEFAULT NOW(),
    deployed_at TIMESTAMP,
    mlflow_run_id VARCHAR(100)
);

-- Drift detection logs
CREATE TABLE IF NOT EXISTS drift_logs (
    id SERIAL PRIMARY KEY,
    detection_timestamp TIMESTAMP DEFAULT NOW(),
    drift_score DECIMAL(8,4),
    auc_current DECIMAL(5,4),
    auc_baseline DECIMAL(5,4),
    features_drifted TEXT[],
    action_taken VARCHAR(50),  -- none, alert, retrain_triggered
    details JSONB
);

-- Merchant risk scores lookup
CREATE TABLE IF NOT EXISTS merchant_risk_scores (
    merchant_category VARCHAR(50) PRIMARY KEY,
    risk_score DECIMAL(5,3) NOT NULL,
    risk_category VARCHAR(20) NOT NULL,
    description TEXT
);

-- Insert default merchant risk scores
INSERT INTO merchant_risk_scores (merchant_category, risk_score, risk_category, description) VALUES
    ('payday_lender', 0.95, 'very_high', 'Payday lending services'),
    ('lending_app', 0.90, 'very_high', 'Digital lending applications'),
    ('cash_advance', 0.85, 'high', 'Cash advance services'),
    ('gambling', 0.80, 'high', 'Gambling and betting'),
    ('pawnshop', 0.75, 'high', 'Pawnbroker services'),
    ('crypto_exchange', 0.70, 'high', 'Cryptocurrency exchanges'),
    ('luxury_goods', 0.50, 'medium', 'High-end luxury retail'),
    ('electronics', 0.40, 'medium', 'Electronics retail'),
    ('dining', 0.35, 'medium', 'Restaurants and dining'),
    ('entertainment', 0.35, 'medium', 'Entertainment venues'),
    ('travel', 0.30, 'medium', 'Travel and airlines'),
    ('clothing', 0.25, 'low', 'Apparel retail'),
    ('healthcare', 0.15, 'low', 'Medical services'),
    ('education', 0.10, 'low', 'Educational institutions'),
    ('grocery', 0.10, 'low', 'Grocery and supermarket'),
    ('utility', 0.05, 'very_low', 'Utility bill payments'),
    ('rent', 0.05, 'very_low', 'Rent payments'),
    ('insurance', 0.05, 'very_low', 'Insurance premiums'),
    ('salary', 0.00, 'none', 'Salary credits'),
    ('transfer', 0.15, 'low', 'Peer-to-peer transfers');

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_transactions_customer ON transactions(customer_id);
CREATE INDEX IF NOT EXISTS idx_transactions_timestamp ON transactions(timestamp);
CREATE INDEX IF NOT EXISTS idx_transactions_type ON transactions(txn_type);
CREATE INDEX IF NOT EXISTS idx_risk_scores_customer ON risk_scores(customer_id);
CREATE INDEX IF NOT EXISTS idx_risk_scores_scored_at ON risk_scores(scored_at);
CREATE INDEX IF NOT EXISTS idx_interventions_customer ON interventions(customer_id);
CREATE INDEX IF NOT EXISTS idx_interventions_sent_at ON interventions(sent_at);
CREATE INDEX IF NOT EXISTS idx_feedback_customer ON feedback_events(customer_id);
CREATE INDEX IF NOT EXISTS idx_account_balances_customer ON account_balances(customer_id);

-- ============================================================
-- New tables for M3, M10, M11
-- ============================================================

-- M3: Employer health metrics
CREATE TABLE IF NOT EXISTS employer_health (
    employer_name VARCHAR(200) PRIMARY KEY,
    employer_payroll_delay_avg DECIMAL(8,2) DEFAULT 0,
    employer_headcount_change_pct DECIMAL(8,2) DEFAULT 0,
    employer_health_score DECIMAL(5,4) DEFAULT 0,
    current_headcount INTEGER DEFAULT 0,
    updated_at TIMESTAMP DEFAULT NOW()
);

-- M10: Product action proposals
CREATE TABLE IF NOT EXISTS product_action_proposals (
    proposal_id SERIAL PRIMARY KEY,
    customer_id VARCHAR(50) REFERENCES customers(customer_id),
    action_type VARCHAR(50) NOT NULL,  -- emi_date_shift, micro_payment_split, interest_rate_review
    proposed_params JSONB,
    status VARCHAR(20) DEFAULT 'proposed',  -- proposed, approved, executed, rejected
    created_at TIMESTAMP DEFAULT NOW(),
    approved_by VARCHAR(100),
    executed_at TIMESTAMP
);

-- Additional indexes
CREATE INDEX IF NOT EXISTS idx_employer_health_score ON employer_health(employer_health_score);
CREATE INDEX IF NOT EXISTS idx_product_proposals_customer ON product_action_proposals(customer_id);
CREATE INDEX IF NOT EXISTS idx_product_proposals_status ON product_action_proposals(status);
CREATE INDEX IF NOT EXISTS idx_batch_features_segment ON batch_features(segment_type);

-- ============================================================
-- Phase 2 schema additions
-- ============================================================

-- P1: Survival analysis outputs on risk_scores
ALTER TABLE risk_scores ADD COLUMN IF NOT EXISTS tte_days DECIMAL(6,1);
ALTER TABLE risk_scores ADD COLUMN IF NOT EXISTS p30d DECIMAL(5,4);
ALTER TABLE risk_scores ADD COLUMN IF NOT EXISTS p60d DECIMAL(5,4);
ALTER TABLE risk_scores ADD COLUMN IF NOT EXISTS p90d DECIMAL(5,4);
ALTER TABLE risk_scores ADD COLUMN IF NOT EXISTS risk_score_lower DECIMAL(5,4);
ALTER TABLE risk_scores ADD COLUMN IF NOT EXISTS risk_score_upper DECIMAL(5,4);
ALTER TABLE risk_scores ADD COLUMN IF NOT EXISTS confidence_flag VARCHAR(20) DEFAULT 'full';
ALTER TABLE risk_scores ADD COLUMN IF NOT EXISTS uplift_score DECIMAL(6,4);
ALTER TABLE risk_scores ADD COLUMN IF NOT EXISTS shadow_score DECIMAL(5,4);

-- P4: A/B holdout assignment tracking
CREATE TABLE IF NOT EXISTS ab_holdout_assignments (
    id SERIAL PRIMARY KEY,
    customer_id VARCHAR(50) REFERENCES customers(customer_id),
    experiment_id VARCHAR(100) NOT NULL DEFAULT 'default',
    group_name VARCHAR(20) NOT NULL,        -- 'treated' or 'control'
    risk_tier VARCHAR(20),
    assigned_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(customer_id, experiment_id)
);
CREATE INDEX IF NOT EXISTS idx_ab_holdout_experiment ON ab_holdout_assignments(experiment_id, group_name);

-- P8: Shadow mode scoring log
CREATE TABLE IF NOT EXISTS shadow_scores (
    id SERIAL PRIMARY KEY,
    customer_id VARCHAR(50),
    live_score DECIMAL(5,4),
    shadow_score DECIMAL(5,4),
    divergence DECIMAL(5,4),
    features_hash VARCHAR(64),
    scored_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_shadow_scores_scored_at ON shadow_scores(scored_at);
CREATE INDEX IF NOT EXISTS idx_shadow_scores_divergence ON shadow_scores(divergence);

-- P11: Household linkage
ALTER TABLE customers ADD COLUMN IF NOT EXISTS household_id VARCHAR(100);
ALTER TABLE customers ADD COLUMN IF NOT EXISTS household_role VARCHAR(30) DEFAULT 'primary';
CREATE INDEX IF NOT EXISTS idx_customers_household ON customers(household_id);

-- P2: Nudge journey cancellation reason
ALTER TABLE nudge_journeys ADD COLUMN IF NOT EXISTS cancellation_reason VARCHAR(100);
ALTER TABLE nudge_journeys ADD COLUMN IF NOT EXISTS sent_at TIMESTAMP;
ALTER TABLE nudge_journeys ADD COLUMN IF NOT EXISTS dispatch_result TEXT;
CREATE INDEX IF NOT EXISTS idx_nudge_journey_scheduled ON nudge_journeys(scheduled_at, status);
CREATE INDEX IF NOT EXISTS idx_nudge_journey_customer ON nudge_journeys(customer_id, status);

-- P10: GST feature columns on batch_features
ALTER TABLE batch_features ADD COLUMN IF NOT EXISTS gst_filing_regularity_6m DECIMAL(4,3) DEFAULT 0;
ALTER TABLE batch_features ADD COLUMN IF NOT EXISTS gst_gap_months DECIMAL(4,1) DEFAULT 0;
ALTER TABLE batch_features ADD COLUMN IF NOT EXISTS business_inflow_trend_pct DECIMAL(6,3) DEFAULT 0;
ALTER TABLE batch_features ADD COLUMN IF NOT EXISTS gst_amount_trend_pct DECIMAL(6,3) DEFAULT 0;
ALTER TABLE batch_features ADD COLUMN IF NOT EXISTS vendor_payment_count_90d INTEGER DEFAULT 0;

-- Additional Phase 2 indexes
CREATE INDEX IF NOT EXISTS idx_risk_scores_tte ON risk_scores(tte_days) WHERE tte_days IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_customers_household_id ON customers(household_id) WHERE household_id IS NOT NULL;

-- ============================================================
-- PAYMENT EVENTS (outcome-based labels for ML training)
-- ============================================================

-- Records actual missed payment events from the financial simulation.
-- These are the ML training LABELS — a customer is "delinquent" if they have
-- missed_emi or missed_auto_debit events in the outcome window.
CREATE TABLE IF NOT EXISTS payment_events (
    id BIGSERIAL PRIMARY KEY,
    customer_id VARCHAR(50) REFERENCES customers(customer_id),
    event_type VARCHAR(30) NOT NULL,          -- missed_emi, missed_auto_debit, bounced_cheque
    amount DECIMAL(15, 2),                    -- EMI/payment amount that was missed
    due_date DATE,                            -- when the payment was due
    balance_at_event DECIMAL(15, 2),          -- account balance when the miss occurred
    event_date TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_payment_events_cust ON payment_events(customer_id);
CREATE INDEX IF NOT EXISTS idx_payment_events_date ON payment_events(event_date);

-- ============================================================
-- SECURITY TABLES (Phase: Production Hardening)
-- ============================================================

-- Users table (for JWT /auth/token login)
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,   -- bcrypt hash, never plaintext
    role VARCHAR(30) NOT NULL DEFAULT 'analyst',  -- analyst, risk_officer, admin, read_only
    email VARCHAR(200),
    full_name VARCHAR(200),
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_login_at TIMESTAMPTZ,
    CONSTRAINT users_role_check CHECK (role IN ('analyst', 'risk_officer', 'admin', 'read_only'))
);
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);

-- API Keys table (for service-to-service authentication)
CREATE TABLE IF NOT EXISTS api_keys (
    key_hash VARCHAR(64) PRIMARY KEY,      -- SHA-256 hash of raw key (never store raw key)
    service_name VARCHAR(100) NOT NULL,
    role VARCHAR(30) NOT NULL DEFAULT 'service_account',
    description TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_used_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ,
    revoked BOOLEAN NOT NULL DEFAULT FALSE,
    CONSTRAINT api_keys_role_check CHECK (role IN ('service_account', 'analyst', 'risk_officer', 'admin'))
);
CREATE INDEX IF NOT EXISTS idx_api_keys_revoked ON api_keys(revoked) WHERE revoked = FALSE;

-- Audit log table (tamper-evident, INSERT-only in production)
-- customer_id is stored as one-way SHA-256 hash — never in plaintext
CREATE TABLE IF NOT EXISTS audit_log (
    id BIGSERIAL PRIMARY KEY,
    event_id UUID DEFAULT gen_random_uuid(),
    event_type VARCHAR(50) NOT NULL,
    actor_id VARCHAR(100),                  -- JWT sub (username) or service name
    actor_role VARCHAR(30),
    customer_id_token VARCHAR(64),          -- SHA-256(salt + customer_id) — non-reversible
    action VARCHAR(200) NOT NULL,
    outcome VARCHAR(20),                    -- SUCCESS, FAILURE, BLOCKED, PARTIAL
    request_ip INET,
    details JSONB,                          -- Non-PII operational context only
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_audit_actor ON audit_log(actor_id);
CREATE INDEX IF NOT EXISTS idx_audit_customer_token ON audit_log(customer_id_token);
CREATE INDEX IF NOT EXISTS idx_audit_created_at ON audit_log(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_event_type ON audit_log(event_type);
-- Audit log must be retained for 5+ years (regulatory requirement)
COMMENT ON TABLE audit_log IS 'Tamper-evident audit trail. Minimum 5-year retention per Basel III / FCA SYSC 9. No UPDATE or DELETE in production.';

-- Customer consent tracking (required by GDPR Art. 6 / TCPA / FCA CONC guidelines)
CREATE TABLE IF NOT EXISTS customer_consent (
    id SERIAL PRIMARY KEY,
    customer_id VARCHAR(50) REFERENCES customers(customer_id) ON DELETE CASCADE,
    channel VARCHAR(20) NOT NULL,           -- sms, email, whatsapp, rm_call, app_push
    consent_given BOOLEAN NOT NULL DEFAULT FALSE,
    consent_method VARCHAR(50),             -- app_onboarding, ivr, written_form, digital_mandate
    consent_timestamp TIMESTAMPTZ,
    withdrawal_timestamp TIMESTAMPTZ,       -- NULL means consent is active
    consent_version VARCHAR(20),            -- version of T&C consented to
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(customer_id, channel),
    CONSTRAINT consent_channel_check CHECK (channel IN ('sms', 'email', 'whatsapp', 'rm_call', 'app_push', 'collector'))
);
CREATE INDEX IF NOT EXISTS idx_consent_customer ON customer_consent(customer_id);
CREATE INDEX IF NOT EXISTS idx_consent_active ON customer_consent(customer_id, channel) WHERE withdrawal_timestamp IS NULL AND consent_given = TRUE;

-- Add audit columns to interventions table
ALTER TABLE interventions ADD COLUMN IF NOT EXISTS triggered_by VARCHAR(100);
ALTER TABLE interventions ADD COLUMN IF NOT EXISTS trigger_ip INET;
ALTER TABLE interventions ADD COLUMN IF NOT EXISTS consent_verified BOOLEAN DEFAULT FALSE;

-- Default admin user (password must be changed immediately on first login)
-- Password hash is bcrypt of 'ChangeMe@2024!' — MUST be changed before production
INSERT INTO users (username, password_hash, role, email, full_name)
VALUES (
    'admin',
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBACG0qsMcCxse',
    'admin',
    'admin@barclays.in',
    'PDI System Administrator'
) ON CONFLICT (username) DO NOTHING;

COMMENT ON TABLE users IS 'Default admin password is ChangeMe@2024! — change immediately on first login.';

-- Seed consent data for existing customers (opt-in assumed for demo; in production this requires explicit consent)
INSERT INTO customer_consent (customer_id, channel, consent_given, consent_method, consent_timestamp, consent_version)
SELECT customer_id, 'sms', TRUE, 'demo_seed', NOW(), 'v1.0'
FROM customers
ON CONFLICT (customer_id, channel) DO NOTHING;

INSERT INTO customer_consent (customer_id, channel, consent_given, consent_method, consent_timestamp, consent_version)
SELECT customer_id, 'email', TRUE, 'demo_seed', NOW(), 'v1.0'
FROM customers
ON CONFLICT (customer_id, channel) DO NOTHING;
