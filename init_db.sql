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
    lstm_score DECIMAL(5,4),
    ensemble_score DECIMAL(5,4),
    top_shap_features JSONB,
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
CREATE INDEX idx_transactions_customer ON transactions(customer_id);
CREATE INDEX idx_transactions_timestamp ON transactions(timestamp);
CREATE INDEX idx_transactions_type ON transactions(txn_type);
CREATE INDEX idx_risk_scores_customer ON risk_scores(customer_id);
CREATE INDEX idx_risk_scores_scored_at ON risk_scores(scored_at);
CREATE INDEX idx_interventions_customer ON interventions(customer_id);
CREATE INDEX idx_interventions_sent_at ON interventions(sent_at);
CREATE INDEX idx_feedback_customer ON feedback_events(customer_id);
CREATE INDEX idx_account_balances_customer ON account_balances(customer_id);
