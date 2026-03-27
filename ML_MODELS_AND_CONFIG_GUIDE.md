# Pre-Delinquency Engine: ML Models, Hyperparameters & Configuration Guide

---

## Part 1: Configuration Checklist

These are the things you need to configure before running the engine.

### 1.1 Environment Variables (`.env` file)

| Variable | Default | Description |
|---|---|---|
| `POSTGRES_HOST` | `localhost` | PostgreSQL host |
| `POSTGRES_PORT` | `5432` | PostgreSQL port |
| `POSTGRES_USER` | `pdi_user` | Database user |
| `POSTGRES_PASSWORD` | `pdi_password` | Database password |
| `POSTGRES_DB` | `pdi_db` | Database name |
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092` | Kafka broker address |
| `REDIS_HOST` | `localhost` | Redis host |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow server URL |
| `GROQ_API_KEY` | *(required)* | Groq LLM API key for GenAI messages |
| `SMTP_HOST` / `SMTP_PORT` / `SMTP_USER` / `SMTP_PASSWORD` | *(required)* | Email notification credentials |
| `TWILIO_ACCOUNT_SID` / `TWILIO_AUTH_TOKEN` / `TWILIO_FROM_NUMBER` | *(optional)* | SMS/WhatsApp via Twilio |
| `ENSEMBLE_XGB_WEIGHT` | `0.40` | XGBoost weight in ensemble |
| `ENSEMBLE_LGB_WEIGHT` | `0.30` | LightGBM weight in ensemble |
| `ENSEMBLE_LSTM_WEIGHT` | `0.30` | LSTM weight in ensemble |
| `RISK_CRITICAL_THRESHOLD` | `0.7` | Score above this = "critical" tier |
| `RISK_WATCH_THRESHOLD` | `0.5` | Score above this = "watch" tier |
| `STRESS_CUSTOMER_PCT` | `0.20` | % of generated customers with financial stress |

### 1.2 Infrastructure Services Required
1. **PostgreSQL 16** — primary data store + feature tables
2. **Redis 7** — feature cache + Celery broker
3. **Kafka + Zookeeper** — real-time transaction streaming
4. **MLflow** — model registry and experiment tracking

Start all with: `python main.py up`

---

## Part 2: Current Models — What, Why, and How

### 2.1 Model 1: XGBoost (Tabular — Primary Model)

**Why XGBoost?**
- **Industry standard** for credit risk in banking (used by FICO, Upstart, Capital One).
- Handles tabular data with mixed feature types (categorical + continuous) extremely well.
- Native support for missing values (common in real banking data).
- Fast inference (~0.1ms per prediction) — critical for real-time scoring APIs.
- SHAP has **exact** (not approximate) TreeSHAP for XGBoost, giving us regulatory-grade explainability.

**Current Hyperparameters:**

| Parameter | Value | Why This Value |
|---|---|---|
| `objective` | `binary:logistic` | Binary classification (delinquent vs. not) |
| `max_depth` | `6` | Prevents overfitting. Deeper trees memorize noise. Banking data has ~28 features — depth 6 captures all meaningful interactions without memorizing customer IDs. |
| `learning_rate` | `0.05` | Low LR + many trees = better generalization. 0.05 is the sweet spot between 0.01 (too slow) and 0.1 (too aggressive). |
| `n_estimators` | `300` | Combined with LR=0.05, this gives enough capacity. More trees with lower LR reduces variance. |
| `min_child_weight` | `5` | Prevents splits on tiny groups. In banking, a split based on 2-3 customers is unreliable. 5 ensures statistical significance. |
| `subsample` | `0.8` | Row sampling — each tree sees 80% of data. Reduces overfitting and adds diversity between trees. |
| `colsample_bytree` | `0.8` | Column sampling — each tree sees 80% of features. Prevents any single feature from dominating. |
| `gamma` | `0.1` | Minimum loss reduction for a split. Acts as a regularizer — won't make a split unless it improves by at least 0.1. |
| `reg_alpha` | `0.1` | L1 regularization. Encourages feature sparsity — pushes unimportant features to zero. |
| `reg_lambda` | `1.0` | L2 regularization. Prevents large weights. Standard value for most problems. |
| `scale_pos_weight` | `3.0` | Class imbalance handling. If 20% of customers are stressed, ratio ≈ 4:1. Setting to 3.0 slightly under-corrects to avoid false positives (expensive in banking). |

---

### 2.2 Model 2: LightGBM (Tabular — Complementary Model)

**Why LightGBM alongside XGBoost?**
- Uses **leaf-wise** tree growth (vs. XGBoost's level-wise). This captures different patterns.
- Much faster training than XGBoost on large datasets (histogram-based binning).
- Better handling of categorical features natively.
- Different algorithmic bias → ensembling reduces overall variance.

**Current Hyperparameters:**

| Parameter | Value | Why This Value |
|---|---|---|
| `boosting_type` | `gbdt` | Standard gradient boosting. `dart` would be slower. `goss` is faster but less accurate. |
| `num_leaves` | `63` | LightGBM doesn't use `max_depth` as primary — it uses `num_leaves`. 63 = 2^6 - 1, roughly equivalent to depth-6 XGBoost. |
| `learning_rate` | `0.05` | Same as XGBoost for consistency. |
| `max_depth` | `8` | Safety cap. Leaf-wise growth can go deeper than level-wise, so we cap at 8. |
| `feature_fraction` | `0.8` | Same concept as `colsample_bytree`. |
| `bagging_fraction` | `0.8` | Same concept as `subsample`. |
| `bagging_freq` | `5` | Apply bagging every 5 iterations. Reduces computation while still diversifying. |
| `min_child_samples` | `20` | More conservative than XGBoost's `min_child_weight=5`. LightGBM's leaf-wise growth needs stricter pruning. |
| `lambda_l1` / `lambda_l2` | `0.1` / `0.1` | Light regularization. LightGBM is already faster/lighter, so we don't need heavy regularization. |
| `num_boost_round` | `500` | More rounds than XGBoost (300) because early stopping will kick in. |
| `early_stopping_rounds` | `50` | Stop if no improvement in 50 rounds. Prevents overfitting and wastes no compute. |
| `scale_pos_weight` | *Auto-calculated* | Dynamically set as `neg_count / pos_count`. More precise than XGBoost's static 3.0. |

---

### 2.3 Model 3: LSTM (Temporal — Sequential Patterns)

**Why LSTM?**
- XGBoost and LightGBM see each customer as a **single row** (point-in-time snapshot).
- LSTM sees each customer as a **sequence of daily snapshots** over time.
- Detects **gradual deterioration patterns** invisible in tabular data:
  - Slowly increasing ATM withdrawals over 30 days
  - Creeping salary delays (1 day → 3 days → 7 days)
  - Progressive spending pattern changes
- In banking, a customer who was fine 30 days ago but is slowly deteriorating is **very different** from one who had a sudden spike. LSTM captures this.

**Current Architecture:**

```
Input (batch, 30 days, 7 features)
    ↓
LSTM Layer 1 (hidden_size=64, dropout=0.3)
    ↓
LSTM Layer 2 (hidden_size=64, dropout=0.3)
    ↓
Attention Layer (64 → 32 → 1) — learns WHICH days matter most
    ↓
Weighted Context Vector (64 dims)
    ↓
Classifier (64 → 32 → ReLU → Dropout → 1 → Sigmoid)
    ↓
Output: P(delinquency)
```

**Current Hyperparameters:**

| Parameter | Value | Why This Value |
|---|---|---|
| `input_size` | `7` | 7 daily features per timestep (spend, ATM count, lending app usage, etc.) |
| `hidden_size` | `64` | Standard for sequences of length ~30. 128 would overfit on small datasets. 32 would underfit. |
| `num_layers` | `2` | 2-layer LSTM captures both short-term (layer 1) and long-term (layer 2) dependencies. 3+ layers rarely help for sequences < 90 days. |
| `dropout` | `0.3` | 30% dropout between LSTM layers. Standard for banking where overfitting is dangerous. |
| `learning_rate` | `0.001` | Adam optimizer default. Works well for LSTMs. |
| `epochs` | `30-50` | With early stopping via ReduceLROnPlateau. |
| `batch_size` | `64` | Standard choice. 32 = more noise, 128 = smoother but slower convergence. |
| `gradient_clipping` | `1.0` | Prevents exploding gradients (common in LSTMs). |
| **Attention mechanism** | Tanh + Softmax | Learns to focus on the most important days in the 30-day window. A customer's behavior on the day they missed a payment is more important than a normal Tuesday. |
| **LR Scheduler** | `ReduceLROnPlateau(patience=5, factor=0.5)` | Halves learning rate if loss plateaus for 5 epochs. Helps fine-tune after initial learning. |

---

### 2.4 Ensemble Strategy

**Why Ensemble 3 Models?**

| Approach | AUC (Typical) | Why Not Just Use One? |
|---|---|---|
| XGBoost alone | 0.85-0.88 | Misses temporal patterns |
| LightGBM alone | 0.84-0.87 | Similar limitations |
| LSTM alone | 0.78-0.82 | Weak on tabular features, needs large data |
| **Ensemble (all 3)** | **0.89-0.93** | **Best of all worlds** |

**Current Weights:** `XGBoost: 0.40 | LightGBM: 0.30 | LSTM: 0.30`

**Why these weights?**
- XGBoost gets the **highest weight (0.40)** because it's the most reliable on tabular data and has the best individual AUC.
- LightGBM gets **0.30** because it adds diversity with its leaf-wise approach but is slightly less accurate individually.
- LSTM gets **0.30** because temporal patterns are valuable but the model needs more data to be as reliable as tree models.
- If LSTM is unavailable (insufficient temporal data), weights automatically redistribute to XGBoost 0.57 + LightGBM 0.43.

**Risk Tier Mapping:**

| Ensemble Score | Tier | Action |
|---|---|---|
| ≥ 0.70 | 🔴 Critical | Immediate RM call + collector assignment |
| 0.50 – 0.69 | 🟡 Watch | GenAI SMS/Email + EMI restructuring offer |
| < 0.50 | 🟢 Stable | No action (monitoring only) |

---

## Part 3: The 28 Features Used

These are the exact features the models consume (order matters for inference):

| # | Feature | Type | Source |
|---|---|---|---|
| 1 | `discretionary_spend_7d` | Streaming | Flink real-time |
| 2 | `discretionary_spend_30d` | Batch | Spark batch |
| 3 | `atm_withdrawals_count_7d` | Streaming | Flink real-time |
| 4 | `atm_withdrawals_count_30d` | Batch | Spark batch |
| 5 | `lending_app_txn_count_7d` | Streaming | Flink real-time |
| 6 | `lending_app_txn_count_30d` | Batch | Spark batch |
| 7 | `weighted_lending_risk_7d` | Streaming | Flink real-time |
| 8 | `weighted_lending_risk_30d` | Batch | Spark batch |
| 9 | `savings_balance_pct_change_7d` | Streaming | Flink real-time |
| 10 | `failed_autodebits_count_7d` | Streaming | Flink real-time |
| 11 | `failed_autodebits_count_30d` | Batch | Spark batch |
| 12 | `total_spend_7d` | Streaming | Flink real-time |
| 13 | `total_spend_30d` | Batch | Spark batch |
| 14 | `txn_count_7d` | Streaming | Flink real-time |
| 15 | `txn_count_30d` | Batch | Spark batch |
| 16 | `avg_txn_amount_7d` | Streaming | Flink real-time |
| 17 | `max_txn_amount_7d` | Streaming | Flink real-time |
| 18 | `salary_delay_days` | Batch | Payroll detection |
| 19 | `utility_payment_delay_avg` | Batch | Bill payment analysis |
| 20 | `discretionary_spend_trend` | Batch | 3-month slope |
| 21 | `credit_score` | Static | Customer profile |
| 22 | `age` | Static | Customer profile |
| 23 | `tenure_months` | Static | Customer profile |
| 24 | `product_count` | Static | Product holdings |
| 25 | `has_credit_card` | Static | Product flag |
| 26 | `has_personal_loan` | Static | Product flag |
| 27 | `has_mortgage` | Static | Product flag |
| 28 | `avg_monthly_spend_3m` / `spend_volatility_3m` | Batch | 3-month aggregates |

---

## Part 4: What Could Be Better — Recommended Improvements

### 4.1 Model Improvements

#### 🏆 Add TabNet (Google's Deep Tabular Model)
- **What it is:** A neural network specifically designed for tabular data, using attention to select features at each step.
- **Why better:** Achieves XGBoost-level accuracy with built-in feature selection. Would replace the manual SHAP step — TabNet has native explainability.
- **Implementation:** `pip install pytorch-tabnet`, train alongside XGBoost/LightGBM, add to ensemble with weight ~0.25.

#### 🏆 Replace LSTM with Temporal Fusion Transformer (TFT)
- **What it is:** Google's state-of-the-art model for time-series data. Combines LSTM-like temporal encoding with Transformer attention.
- **Why better:** 
  - Handles both static features (age, credit score) AND temporal features (daily spending) in a **single model**.
  - Built-in multi-horizon forecasting — predicts not just "will they default?" but "when will they default?"
  - Built-in variable importance — tells you which features mattered at which time.
- **Implementation:** `pip install pytorch-forecasting`, would replace both LSTM and part of XGBoost's role.

#### 🥈 Add CatBoost
- **What it is:** Yandex's gradient boosting library.
- **Why better:** Best native handling of categorical features (employment type, industry sector). Uses ordered boosting to reduce overfitting.
- **Implementation:** Add as 4th model in ensemble, weight ~0.15.

#### 🥈 Add Isolation Forest for Anomaly Detection
- **What it is:** Unsupervised anomaly detector.
- **Why better:** Catches **completely new** patterns of stress that supervised models haven't been trained on (e.g., a new type of fraud or economic shock).
- **Implementation:** Run as a pre-filter. If anomaly score > threshold, flag customer regardless of ensemble score.

### 4.2 Hyperparameter Improvements

#### Use Optuna for Automated Tuning
The Jupyter notebook (`PDI_Model_Training_Notebook.ipynb`) already includes Optuna tuning. To use it in production:

```python
import optuna

def objective(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
    }
    model = xgb.XGBClassifier(**params)
    cv_score = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
    return cv_score.mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
```

### 4.3 Data & Feature Improvements

| Improvement | Impact | Effort |
|---|---|---|
| Add **UPI/NEFT transaction metadata** | High — captures digital payment behavior | Medium |
| Add **social network features** (shared accounts, guarantors) | Very High — default contagion | High |
| Add **macroeconomic indicators** (RBI repo rate, inflation, unemployment) | Medium — captures market-level stress | Low |
| Add **geospatial risk** (district-level default rates) | Medium — location bias detection | Medium |
| Use **target encoding** for categorical features | Medium — better than one-hot for rare categories | Low |

### 4.4 Architecture Improvements

| Improvement | What It Does |
|---|---|
| **Online learning** (River/Vowpal Wabbit) | Update model with each new transaction without full retraining |
| **Multi-task learning** | Predict delinquency AND the specific product that will default simultaneously |
| **Survival analysis** (Cox PH / DeepSurv) | Predict **time-to-default**, not just probability. Much more actionable for RMs |
| **Conformal prediction** | Instead of a single probability, output a **confidence interval** (e.g., "72% ± 5%"). Better for decision-making |
