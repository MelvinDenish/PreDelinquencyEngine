#!/usr/bin/env python3
"""
PDI Model Training Script (Docker-compatible)
Trains XGBoost + LightGBM + TFT ensemble — NO LSTM.
Runs inside pdi-app container, connects to pdi-postgres.
"""
import os, sys, json, time, logging
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (roc_auc_score, classification_report, confusion_matrix,
                              f1_score, precision_score, recall_score, accuracy_score)
import xgboost as xgb
import lightgbm as lgbm
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from config.settings import PostgresConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("PDI-Train")

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)


def load_data():
    """Load features and labels from PostgreSQL."""
    from sqlalchemy import create_engine
    engine = create_engine(PostgresConfig.get_url())

    print("\n" + "="*60)
    print("  LOADING DATA FROM POSTGRESQL")
    print("="*60)

    streaming = pd.read_sql("SELECT * FROM streaming_features", engine)
    batch = pd.read_sql("SELECT * FROM batch_features", engine)
    print(f"  Streaming features: {len(streaming)} customers")
    print(f"  Batch features: {len(batch)} customers")

    merged = streaming.merge(batch, on="customer_id", how="inner",
                             suffixes=("_stream", "_batch"))
    print(f"  Merged: {len(merged)} customers × {len(merged.columns)} columns")

    # Labels from payment_events
    try:
        outcome = pd.read_sql("""
            SELECT customer_id, COUNT(*) as missed_count
            FROM payment_events GROUP BY customer_id
        """, engine)
        outcome["label"] = 1
        merged = merged.merge(outcome[["customer_id", "label"]],
                              on="customer_id", how="left")
        merged["label"] = merged["label"].fillna(0).astype(int)
    except Exception as e:
        print(f"  ⚠ payment_events not found ({e}), creating labels from features")
        # Fallback label creation from behavioral signals
        def create_label(row):
            score = 0
            score += min(row.get('failed_autodebits_count_30d', 0) * 0.3, 1.0)
            score += min(row.get('lending_app_txn_count_30d', 0) * 0.15, 0.5)
            score += min(row.get('salary_delay_days', 0) * 0.05, 0.3)
            sav = row.get('savings_balance_pct_change_7d', 0)
            if sav < -0.2: score += 0.2
            spend = row.get('discretionary_spend_trend', 1.0)
            if spend > 1.5: score += 0.15
            return 1 if score >= 0.5 else 0
        merged["label"] = merged.apply(create_label, axis=1)

    pos = merged["label"].sum()
    total = len(merged)
    print(f"  Labels: {pos}/{total} delinquent ({pos/max(total,1)*100:.1f}%)")
    return merged


def prepare_features(merged):
    """Extract feature matrix and labels."""
    exclude = ['customer_id', 'updated_at_stream', 'updated_at_batch',
               'label', 'income_bracket', 'region', 'gender',
               'created_at_stream', 'created_at_batch']
    feature_cols = [c for c in merged.columns
                    if c not in exclude and merged[c].dtype in ['int64','float64','int32','float32']]

    X = merged[feature_cols].fillna(0).values
    y = merged['label'].values
    cids = merged['customer_id'].values

    print(f"\n  Features: {len(feature_cols)}")
    print(f"  Samples: {len(X)}")
    return X, y, feature_cols, cids


def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost model."""
    print("\n" + "="*60)
    print("  TRAINING XGBOOST")
    print("="*60)

    params = {
        'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.05,
        'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 5,
        'gamma': 0.1, 'reg_alpha': 0.1, 'reg_lambda': 1.0,
        'scale_pos_weight': (y_train==0).sum() / max((y_train==1).sum(), 1),
        'objective': 'binary:logistic', 'eval_metric': 'auc',
        'random_state': 42, 'n_jobs': -1,
    }

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50)

    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    print(f"\n  ✓ XGBoost Test AUC: {auc:.4f}")

    # Save
    path = os.path.join(MODEL_DIR, "xgboost_model.joblib")
    joblib.dump(model, path)
    print(f"  ✓ Saved: {path}")
    return model, probs, auc


def train_lightgbm(X_train, y_train, X_test, y_test):
    """Train LightGBM model."""
    print("\n" + "="*60)
    print("  TRAINING LIGHTGBM")
    print("="*60)

    params = {
        'n_estimators': 300, 'num_leaves': 31, 'learning_rate': 0.05,
        'min_child_samples': 20, 'subsample': 0.8, 'colsample_bytree': 0.8,
        'reg_alpha': 0.1, 'reg_lambda': 1.0, 'is_unbalance': True,
        'objective': 'binary', 'metric': 'auc',
        'random_state': 42, 'n_jobs': -1, 'verbose': -1,
    }

    model = lgbm.LGBMClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)])

    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    print(f"\n  ✓ LightGBM Test AUC: {auc:.4f}")

    path = os.path.join(MODEL_DIR, "lightgbm_model.joblib")
    joblib.dump(model, path)
    print(f"  ✓ Saved: {path}")
    return model, probs, auc


def train_tft(X_train, y_train, X_test, y_test, n_features):
    """Train Temporal Fusion Transformer."""
    print("\n" + "="*60)
    print("  TRAINING TFT (Temporal Fusion Transformer)")
    print("="*60)

    from ml.tft_model import TFTDelinquencyModel

    # Create temporal sequences (30-day windows with noise)
    seq_len = 30
    def create_sequences(X, seq_len=30):
        seqs = []
        for i in range(len(X)):
            seq = np.tile(X[i], (seq_len, 1))
            noise = np.random.normal(0, 0.05, seq.shape)
            seq = seq + noise * seq
            seqs.append(seq)
        return np.array(seqs, dtype=np.float32)

    # Split features: first 5 as static, rest as temporal
    n_static = min(5, n_features)
    n_temporal = n_features

    X_train_static = X_train[:, :n_static].astype(np.float32)
    X_test_static = X_test[:, :n_static].astype(np.float32)
    X_train_seq = create_sequences(X_train, seq_len)
    X_test_seq = create_sequences(X_test, seq_len)

    print(f"  Static features: {n_static}")
    print(f"  Temporal features: {n_temporal}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Train shape: {X_train_seq.shape}")

    tft = TFTDelinquencyModel(
        n_temporal_features=n_temporal,
        n_static_features=n_static,
        d_model=64, nhead=4, num_layers=2,
        dropout=0.2, seq_len=seq_len,
        epochs=30, batch_size=64, learning_rate=1e-3,
    )

    metrics = tft.train(
        X_train_seq, X_train_static, y_train.astype(np.float32),
        X_test_seq, X_test_static, y_test.astype(np.float32)
    )

    probs = tft.predict_proba(X_test_seq, X_test_static)
    auc = roc_auc_score(y_test, probs)
    print(f"\n  ✓ TFT Test AUC: {auc:.4f}")

    path = os.path.join(MODEL_DIR, "tft_model.pt")
    tft.save(path)
    print(f"  ✓ Saved: {path}")
    return tft, probs, auc


def create_ensemble(xgb_probs, lgb_probs, tft_probs, y_test):
    """Create weighted ensemble."""
    print("\n" + "="*60)
    print("  ENSEMBLE RESULTS")
    print("="*60)

    w_xgb, w_lgb, w_tft = 0.35, 0.40, 0.25
    ensemble_probs = w_xgb * xgb_probs + w_lgb * lgb_probs + w_tft * tft_probs

    auc = roc_auc_score(y_test, ensemble_probs)
    preds = (ensemble_probs >= 0.5).astype(int)
    f1 = f1_score(y_test, preds, zero_division=0)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    acc = accuracy_score(y_test, preds)

    print(f"  Weights: XGB={w_xgb}, LGB={w_lgb}, TFT={w_tft}")
    print(f"  Ensemble AUC: {auc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall: {rec:.4f}")
    print(f"  F1: {f1:.4f}")
    print(f"  Accuracy: {acc:.4f}")

    # Risk tiers
    def tier(s):
        if s >= 0.7: return 'critical'
        elif s >= 0.5: return 'watch'
        return 'stable'

    tiers = [tier(p) for p in ensemble_probs]
    for t in ['stable', 'watch', 'critical']:
        cnt = tiers.count(t)
        print(f"  {t}: {cnt} ({cnt/len(tiers)*100:.1f}%)")

    return ensemble_probs, auc


def log_to_mlflow(xgb_auc, lgb_auc, tft_auc, ens_auc):
    """Log metrics to MLflow."""
    try:
        import mlflow
        mlflow.set_tracking_uri("http://pdi-mlflow:5000")
        mlflow.set_experiment("PDI-Model-Training")
        with mlflow.start_run(run_name=f"train_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            mlflow.log_metric("xgboost_auc", xgb_auc)
            mlflow.log_metric("lightgbm_auc", lgb_auc)
            mlflow.log_metric("tft_auc", tft_auc)
            mlflow.log_metric("ensemble_auc", ens_auc)
            mlflow.log_param("ensemble_weights", "XGB=0.35,LGB=0.40,TFT=0.25")
            mlflow.log_param("models", "XGBoost+LightGBM+TFT")
            print("  ✓ Logged to MLflow")
    except Exception as e:
        print(f"  ⚠ MLflow logging failed: {e}")


def main():
    start = time.time()
    print("="*60)
    print("  PDI ENGINE — MODEL TRAINING PIPELINE")
    print(f"  3-Model Ensemble: XGBoost + LightGBM + TFT")
    print(f"  Timestamp: {datetime.now().isoformat()}")
    print("="*60)

    # 1. Load data
    merged = load_data()

    # 2. Prepare features
    X, y, feature_cols, cids = prepare_features(merged)

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"\n  Train: {len(X_train)} ({y_train.mean()*100:.1f}% positive)")
    print(f"  Test: {len(X_test)} ({y_test.mean()*100:.1f}% positive)")

    # 4. Train models
    xgb_model, xgb_probs, xgb_auc = train_xgboost(X_train, y_train, X_test, y_test)
    lgb_model, lgb_probs, lgb_auc = train_lightgbm(X_train, y_train, X_test, y_test)
    tft_model, tft_probs, tft_auc = train_tft(X_train, y_train, X_test, y_test, len(feature_cols))

    # 5. Ensemble
    ens_probs, ens_auc = create_ensemble(xgb_probs, lgb_probs, tft_probs, y_test)

    # 6. Save feature names
    joblib.dump(feature_cols, os.path.join(MODEL_DIR, "feature_names.joblib"))

    # 7. Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "dataset_size": len(X),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "n_features": len(feature_cols),
        "xgboost_auc": round(xgb_auc, 4),
        "lightgbm_auc": round(lgb_auc, 4),
        "tft_auc": round(tft_auc, 4),
        "ensemble_auc": round(ens_auc, 4),
        "elapsed_seconds": round(time.time() - start, 1),
    }
    with open(os.path.join(MODEL_DIR, "training_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # 8. MLflow
    log_to_mlflow(xgb_auc, lgb_auc, tft_auc, ens_auc)

    # Summary
    elapsed = time.time() - start
    print("\n" + "="*60)
    print("  TRAINING COMPLETE")
    print("="*60)
    print(f"  XGBoost AUC:  {xgb_auc:.4f}")
    print(f"  LightGBM AUC: {lgb_auc:.4f}")
    print(f"  TFT AUC:      {tft_auc:.4f}")
    print(f"  Ensemble AUC: {ens_auc:.4f}  ← FINAL")
    print(f"  Elapsed:      {elapsed:.0f}s")
    print(f"  Models saved: {MODEL_DIR}/")
    print("="*60)


if __name__ == "__main__":
    main()
