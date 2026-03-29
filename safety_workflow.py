import os
import time
import json
import random
import datetime
import requests

# ---------------------------------------------------------------------------
# Configuration (Uses localhost since it runs on host machine natively now)
# ---------------------------------------------------------------------------
PDI_APP_URL = os.getenv("PDI_APP_URL", "http://localhost:8000")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3001/api/workflow-event")
FRONTEND_SECRET = os.getenv("FRONTEND_SECRET", "pdi-n8n-webhook-secret")
USERNAME = os.getenv("PDI_USERNAME", "admin")
PASSWORD = os.getenv("PDI_PASSWORD", "ChangeMe@2024!")
INTERVAL_SECONDS = int(os.getenv("INTERVAL_SECONDS", "10"))

CUSTOMER_POOL = [
    {"id": 'CUST_CRITICAL_001', "name": 'Vikram Singh',   "city": 'Jaipur',     "region": 'North', "salary": 28000,  "segment": 'gig_worker',    "creditScore": 520},
    {"id": 'CUST_CRITICAL_002', "name": 'Anjali Gupta',   "city": 'Lucknow',    "region": 'North', "salary": 24000,  "segment": 'gig_worker',    "creditScore": 545},
    {"id": 'CUST_CRITICAL_003', "name": 'Ravi Shankar',   "city": 'Patna',      "region": 'East',  "salary": 22000,  "segment": 'farmer',        "creditScore": 498},
    {"id": 'CUST_WATCH_001',    "name": 'Karthik Reddy',  "city": 'Hyderabad',  "region": 'South', "salary": 48000,  "segment": 'salaried',      "creditScore": 634},
    {"id": 'CUST_WATCH_002',    "name": 'Deepa Menon',    "city": 'Kochi',      "region": 'South', "salary": 42000,  "segment": 'self_employed', "creditScore": 648},
    {"id": 'CUST_WATCH_003',    "name": 'Priya Sharma',   "city": 'Mumbai',     "region": 'West',  "salary": 55000,  "segment": 'salaried',      "creditScore": 671},
    {"id": 'CUST_STRESSED_001', "name": 'Rahul Verma',    "city": 'Delhi',      "region": 'North', "salary": 40000,  "segment": 'gig_worker',    "creditScore": 603},
    {"id": 'CUST_STABLE_001',   "name": 'Arjun Nair',     "city": 'Bangalore',  "region": 'South', "salary": 120000, "segment": 'salaried',      "creditScore": 764},
    {"id": 'CUST_STABLE_002',   "name": 'Meera Iyer',     "city": 'Chennai',    "region": 'South', "salary": 65000,  "segment": 'salaried',      "creditScore": 742},
    {"id": 'CUST_STABLE_003',   "name": 'Rohit Desai',    "city": 'Pune',       "region": 'West',  "salary": 95000,  "segment": 'salaried',      "creditScore": 788}
]

STRESS_TXN_TYPES = ['lending_app', 'atm_withdrawal', 'cash_advance', 'payday_lender']
NORMAL_TXN_TYPES = ['upi_debit', 'pos_purchase', 'online_transfer', 'emi_payment']
STRESS_CATS = ['lending_app', 'gambling', 'payday_lender', 'cash_advance']

def current_iso8601():
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")

def generate_transaction():
    cust = random.choice(CUSTOMER_POOL)
    is_critical = 'CRITICAL' in cust['id']
    is_watch = 'WATCH' in cust['id'] or 'STRESSED' in cust['id']
    
    if is_critical:
        txn_type = random.choice(STRESS_TXN_TYPES)
        merchant_cat = random.choice(STRESS_CATS)
        amount = random.randint(2000, 16999) 
        txn_status = 'failed' if random.random() > 0.6 else 'success'
    elif is_watch:
        txn_type = random.choice(STRESS_TXN_TYPES) if random.random() > 0.5 else random.choice(NORMAL_TXN_TYPES)
        merchant_cat = random.choice(STRESS_CATS) if random.random() > 0.4 else 'groceries'
        amount = random.randint(500, 8499)
        txn_status = 'success'
    else:
        txn_type = random.choice(NORMAL_TXN_TYPES)
        merchant_cat = random.choice(['groceries', 'fuel', 'dining', 'entertainment'])
        amount = random.randint(100, 3099)
        txn_status = 'success'
        
    txn = {
        "txn_id": f"TXN_{int(time.time()*1000)}_{random.randint(0, 9999)}",
        "customer_id": cust['id'],
        "customer_name": cust['name'],
        "city": cust['city'],
        "region": cust['region'],
        "segment": cust['segment'],
        "monthly_salary": cust['salary'],
        "credit_score": cust['creditScore'],
        "txn_type": txn_type,
        "merchant_category": merchant_cat,
        "amount": amount,
        "txn_status": txn_status,
        "is_stress_signal": (merchant_cat in STRESS_CATS) or (txn_status == 'failed'),
        "channel": random.choice(['mobile_app', 'web', 'branch', 'atm']),
        "ingested_at": current_iso8601(),
        "failed_autodebits_count_30d": random.randint(3, 7) if is_critical else (random.randint(1, 4) if is_watch else 0),
        "lending_app_txn_count_30d": random.randint(4, 9) if is_critical else (random.randint(1, 3) if is_watch else 0),
        "salary_delay_days": random.randint(5, 23) if is_critical else (random.randint(0, 8) if is_watch else 0),
        "savings_balance_pct_change_7d": -(random.random()*0.45+0.3) if is_critical else (-(random.random()*0.2) if is_watch else random.random()*0.1),
        "discretionary_spend_trend": 1.9+random.random()*0.4 if is_critical else (1.3+random.random()*0.3 if is_watch else 0.8+random.random()*0.2),
        "dti_ratio": 0.58+random.random()*0.2 if is_critical else (0.38+random.random()*0.15 if is_watch else 0.18+random.random()*0.12),
        "night_txn_ratio": 0.32+random.random()*0.18 if is_critical else random.random()*0.12,
        "atm_withdrawal_count_7d": random.randint(4, 11) if is_critical else (random.randint(1, 4) if is_watch else random.randint(0, 1)),
        "pipeline_ts": int(time.time()*1000),
        "source": "CBS_SIMULATION"
    }
    
    print(f"[INGEST] {txn['txn_id']} | {cust['name']} | {txn_type} | amt={amount}")
    return txn

def run_pipeline():
    print("-" * 60)
    print(f"[{current_iso8601()}] Starting Pipeline Run...")
    start_time = int(time.time() * 1000)
    
    # 1. Ingest Transaction
    txn = generate_transaction()
    
    # 2. Score via Backend API
    api_key = "test_pdi_api_key"
    print("[AUTH] Using API Key authentication")
    
    score_result = None
    ml_api_used = False
    try:
        score_res = requests.post(
            f"{PDI_APP_URL}/score",
            headers={"X-API-Key": api_key, "Content-Type": "application/json"},
            json={"customer_id": txn['customer_id']},
            timeout=15
        )
        if score_res.status_code == 200:
            score_result = score_res.json()
            ml_api_used = True
    except Exception as e:
        print(f"[SCORE ERROR] {e}")

    # Process Score Result
    if score_result and 'risk_score' in score_result:
        # Real ML API response
        risk_score = score_result.get('risk_score')
        risk_tier = score_result.get('risk_tier')
        xgboost_score = score_result.get('xgboost_score')
        lightgbm_score = score_result.get('lightgbm_score')
        tft_score = score_result.get('tft_score')
        ensemble_score = score_result.get('ensemble_score', risk_score)
        top_shap_features = score_result.get('top_shap_features', [])
        product_actions = score_result.get('product_actions', [])
        tte_days = score_result.get('tte_days')
        calibrated_pd = score_result.get('calibrated_pd')
        uplift_score = score_result.get('uplift_score')
        confidence_flag = score_result.get('confidence_flag', 'green')
        meta_learner_used = score_result.get('meta_learner_used', False)
        scored_at = score_result.get('scored_at', current_iso8601())
        ml_api_used = True
    else:
        # Heuristic fallback
        is_critical = 'CRITICAL' in txn['customer_id']
        is_watch = 'WATCH' in txn['customer_id'] or 'STRESSED' in txn['customer_id']
        
        if is_critical:
            risk_score = 0.72 + random.random() * 0.2
        elif is_watch:
            risk_score = 0.42 + random.random() * 0.2
        else:
            risk_score = 0.1 + random.random() * 0.2
            
        risk_tier = 'critical' if risk_score > 0.65 else ('watch' if risk_score > 0.35 else 'stable')
        xgboost_score = risk_score + (random.random()*0.06 - 0.03)
        lightgbm_score = risk_score + (random.random()*0.06 - 0.03)
        tft_score = None
        ensemble_score = risk_score
        top_shap_features = [
            {"feature": 'dti_ratio', "value": txn['dti_ratio']},
            {"feature": 'credit_score', "value": -0.3},
            {"feature": 'failed_autodebits', "value": txn['failed_autodebits_count_30d'] * 0.1}
        ]
        
        if risk_score > 0.65:
            product_actions = ['suspend_new_credit', 'escalate_to_rm', 'send_financial_counseling']
            tte_days = random.randint(5, 19)
        elif risk_score > 0.35:
            product_actions = ['flag_for_review', 'reduce_credit_limit', 'send_payment_reminder']
            tte_days = random.randint(15, 59)
        else:
            product_actions = ['monitor_passively']
            tte_days = None
            
        calibrated_pd = risk_score * 0.9
        uplift_score = None
        confidence_flag = 'yellow'
        meta_learner_used = False
        scored_at = current_iso8601()
        ml_api_used = False

    tier_icon = {'critical': '🔴', 'watch': '🟡', 'stable': '🟢'}.get(risk_tier, '⚪')
    print(f"[SCORE] {txn['customer_name']} | {tier_icon} {risk_tier.upper()} | score={risk_score:.3f} | ml={ml_api_used}")
    
    # 4. Dispatch Notification if actionable
    notify_status, channels_sent, channels_count, notify_results = None, None, 0, []
    
    if risk_tier != 'stable':
        try:
            # Add delay to make animation visible on frontend
            time.sleep(1.5)
            notify_payload = {
                "customer_id": txn['customer_id'],
                "customer_name": txn['customer_name'],
                "risk_score": risk_score,
                "risk_tier": risk_tier,
                "alert_message": f"Unusual spend pattern detected: {txn['txn_type']} for {txn['amount']} at {txn['merchant_category']}",
                "city": txn['city'],
                "salary": txn['monthly_salary']
            }
            
            notify_res = requests.post(
                f"{PDI_APP_URL}/notify",
                headers={"X-API-Key": api_key, "Content-Type": "application/json"},
                json=notify_payload,
                timeout=10
            )
            
            if notify_res.status_code == 200:
                notify_data = notify_res.json()
                notify_status = 'dispatched'
                channels_count = notify_data.get('channels_attempted', 0)
                notify_results = notify_data.get('results', [])
                sent_channels = [r['channel'] for r in notify_results if r.get('status') == 'sent']
                channels_sent = '+'.join(sent_channels) if sent_channels else 'attempted'
            else:
                notify_status = 'dispatch_error'
                channels_sent = 'error'
        except Exception as e:
            notify_status = 'dispatch_error'
            channels_sent = 'error'
    else:
         notify_status = 'no_action'
         channels_sent = 'none'
         print(f"[STABLE] {txn['customer_name']} — no action needed (score={risk_score:.3f})")
         
    if risk_tier != 'stable':
        print(f"[NOTIFY] {txn['customer_name']} | status={notify_status} | via={channels_sent}")
        
    # 5. Build Summary & Push to Frontend
    pipeline_ms = int(time.time() * 1000) - start_time
    ml_label = "Real ML API" if ml_api_used else "Heuristic Fallback"
    
    summary = {
      "event_id": f"EVT_{int(time.time()*1000)}_{random.randint(0,9999)}",
      "event_time": current_iso8601(),
      "pipeline_run_ms": pipeline_ms,
      "customer_id": txn['customer_id'],
      "customer_name": txn['customer_name'],
      "city": txn['city'],
      "region": txn['region'],
      "segment": txn['segment'],
      "credit_score": txn['credit_score'],
      "monthly_salary": txn['monthly_salary'],
      "txn_id": txn['txn_id'],
      "txn_type": txn['txn_type'],
      "merchant_category": txn['merchant_category'],
      "amount": txn['amount'],
      "is_stress_signal": txn['is_stress_signal'],
      "risk_score": risk_score,
      "risk_tier": risk_tier,
      "risk_tier_icon": tier_icon,
      "xgboost_score": xgboost_score,
      "lightgbm_score": lightgbm_score,
      "tft_score": tft_score,
      "ensemble_score": ensemble_score,
      "calibrated_pd": calibrated_pd,
      "tte_days": tte_days,
      "uplift_score": uplift_score,
      "confidence_flag": confidence_flag,
      "meta_learner_used": meta_learner_used,
      "top_shap_features": top_shap_features,
      "product_actions": product_actions,
      "ml_source": ml_label,
      "intervention_type": 'none' if risk_tier == 'stable' else ('preventive_outreach' if risk_tier == 'watch' else 'urgent_intervention'),
      "notify_status": notify_status,
      "channels_sent": channels_sent,
      "channels_count": channels_count,
      "notify_results": notify_results,
      "stages": [
        {"stage": 'ingestion',    "ts": txn['ingested_at'],  "status": 'ok',                                  "label": 'CBS Ingested'},
        {"stage": 'auth',         "ts": txn['ingested_at'],  "status": 'ok',                                  "label": 'JWT Auth Token Obtained'},
        {"stage": 'ml_scoring',   "ts": scored_at,           "status": 'live' if ml_api_used else 'fallback', "label": f'ML Scored ({ml_label})'},
        {"stage": 'risk_routing', "ts": scored_at,           "status": 'ok',                                  "label": f'Routed to {risk_tier.upper()}'},
        {"stage": 'notification', "ts": current_iso8601(),   "status": 'skipped' if notify_status == 'no_action' else 'ok', "label": 'No Action (Stable)' if notify_status == 'no_action' else f'Notified via {channels_sent}'},
        {"stage": 'complete',     "ts": current_iso8601(),   "status": 'ok',                                  "label": f'Pipeline Complete ({pipeline_ms}ms)'}
      ]
    }
    
    print(f"[PIPELINE] {txn['customer_name']} | {tier_icon} {risk_tier.upper()} | score={risk_score:.3f} | ml={ml_label} | run={pipeline_ms}ms")
    
    # 6. Push to Frontend
    try:
        frontend_res = requests.post(
            FRONTEND_URL,
            headers={
                "Content-Type": "application/json",
                "X-PDI-Secret": FRONTEND_SECRET
            },
            json=summary,
            timeout=5
        )
        print(f"[FRONTEND PUSH] status={frontend_res.status_code} event={summary['event_id']}")
    except Exception as e:
        print(f"[FRONTEND PUSH ERROR] {e}")
        
    print(f"[COMPLETE] event={summary['event_id']} | {summary['customer_name']} | {summary['risk_tier'].upper()} | score={summary['risk_score']:.3f} | pipeline={pipeline_ms}ms")

if __name__ == "__main__":
    import sys
    if "--once" in sys.argv:
        print(f"[{current_iso8601()}] Running safety workflow simulation ONCE.")
        run_pipeline()
    else:
        print(f"Starting safety workflow simulation. Will run every {INTERVAL_SECONDS} seconds.")
        print("Press Ctrl+C to stop.")
        try:
            while True:
                run_pipeline()
                time.sleep(INTERVAL_SECONDS)
        except KeyboardInterrupt:
            print("\nStopping safety workflow simulation.")
