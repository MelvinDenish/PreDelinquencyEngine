"""
ML Training Pipeline
Orchestrates the full model training workflow:
1. Build datasets from PostgreSQL
2. Train XGBoost on tabular features
3. Train LightGBM on tabular features
4. Train LSTM on temporal sequences
5. Evaluate 3-model ensemble
6. SHAP + LIME explainability
7. Fairness audit (Fairlearn + AIF360)
8. Register models in MLflow
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config.settings import ModelConfig, MLflowConfig, PostgresConfig
from ml.dataset_builder import build_training_dataset, build_temporal_dataset
from ml.xgboost_model import XGBoostDelinquencyModel
from ml.lightgbm_model import LightGBMDelinquencyModel
from ml.lstm_model import LSTMDelinquencyModel
from ml.ensemble import EnsembleScorer
from ml.explainability import SHAPExplainer
from ml.fairness import run_bias_audit

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)


def train_pipeline():
    """Run the complete training pipeline."""
    print("=" * 70)
    print("Pre-Delinquency Engine - ML Training Pipeline")
    print("  Models: XGBoost + LightGBM + LSTM")
    print("  Explainability: SHAP + LIME")
    print("  Fairness: Fairlearn + AIF360")
    print("  Registry: MLflow")
    print("=" * 70)

    # ─────────────────────────────────────────────
    # Step 1: Build datasets
    # ─────────────────────────────────────────────
    print("\n[1/8] Building training datasets...")
    X_tab, y_tab, feature_names, customer_ids = build_training_dataset()
    if X_tab is None:
        print("ERROR: Could not build training dataset.")
        return

    X_seq, y_seq, cids_seq = build_temporal_dataset()

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_tab, y_tab, np.arange(len(y_tab)),
        test_size=0.2, stratify=y_tab, random_state=42,
    )
    print(f"  Train: {len(X_train)} samples, Test: {len(X_test)} samples")

    # ─────────────────────────────────────────────
    # Step 2: Train XGBoost
    # ─────────────────────────────────────────────
    print("\n[2/8] Training XGBoost model...")
    xgb_model = XGBoostDelinquencyModel()
    xgb_metrics = xgb_model.train(X_train, y_train, feature_names, X_test, y_test)
    xgb_model.save()
    print(f"  -> Train AUC: {xgb_metrics['train_auc']:.4f}")
    print(f"  -> CV AUC: {xgb_metrics['cv_auc_mean']:.4f} +/- {xgb_metrics['cv_auc_std']:.4f}")
    if xgb_metrics.get("val_auc"):
        print(f"  -> Test AUC: {xgb_metrics['val_auc']:.4f}")

    if xgb_metrics.get("feature_importance"):
        print("  -> Top features:")
        for feat, imp in list(xgb_metrics["feature_importance"].items())[:5]:
            print(f"     {feat}: {imp:.4f}")

    # ─────────────────────────────────────────────
    # Step 3: Train LightGBM
    # ─────────────────────────────────────────────
    print("\n[3/8] Training LightGBM model...")
    lgb_model = LightGBMDelinquencyModel()
    lgb_metrics = lgb_model.train(X_train, y_train, feature_names, X_test, y_test)
    lgb_model.save()
    print(f"  -> Train AUC: {lgb_metrics['train_auc']:.4f}")
    print(f"  -> CV AUC: {lgb_metrics['cv_auc_mean']:.4f} +/- {lgb_metrics['cv_auc_std']:.4f}")
    if lgb_metrics.get("val_auc"):
        print(f"  -> Test AUC: {lgb_metrics['val_auc']:.4f}")
    print(f"  -> Trees: {lgb_metrics.get('num_trees', 'N/A')}")

    # ─────────────────────────────────────────────
    # Step 4: Train LSTM
    # ─────────────────────────────────────────────
    lstm_model = None
    lstm_metrics = {}
    if X_seq is not None and len(X_seq) > 50:
        print("\n[4/8] Training LSTM model...")
        seq_train_idx, seq_test_idx = train_test_split(
            np.arange(len(X_seq)), test_size=0.2, stratify=y_seq, random_state=42,
        )
        lstm_model = LSTMDelinquencyModel(
            input_size=X_seq.shape[2], hidden_size=64, num_layers=2,
            epochs=30, batch_size=64,
        )
        lstm_metrics = lstm_model.train(
            X_seq[seq_train_idx], y_seq[seq_train_idx],
            X_seq[seq_test_idx], y_seq[seq_test_idx],
        )
        lstm_model.save()
        print(f"  -> Train AUC: {lstm_metrics.get('train_auc', 'N/A')}")
        print(f"  -> Best Val AUC: {lstm_metrics.get('best_val_auc', 'N/A')}")
    else:
        print("\n[4/8] Skipping LSTM (insufficient temporal data)")

    # ─────────────────────────────────────────────
    # Step 5: Evaluate 3-Model Ensemble
    # ─────────────────────────────────────────────
    print("\n[5/8] Evaluating 3-model ensemble (XGBoost + LightGBM + LSTM)...")
    ensemble = EnsembleScorer()

    xgb_test_probs = xgb_model.predict_proba(X_test)
    lgb_test_probs = lgb_model.predict_proba(X_test)

    lstm_test_probs = None
    if lstm_model is not None and X_seq is not None:
        test_customer_ids = customer_ids[idx_test]
        lstm_test_probs = np.zeros(len(X_test))
        for i, cid in enumerate(test_customer_ids):
            seq_idx = np.where(cids_seq == cid)[0]
            if len(seq_idx) > 0:
                lstm_test_probs[i] = lstm_model.predict_proba(X_seq[seq_idx[0]:seq_idx[0]+1])[0]

    ensemble_probs = ensemble.combine_batch(xgb_test_probs, lgb_test_probs, lstm_test_probs)
    ensemble_auc = roc_auc_score(y_test, ensemble_probs)
    ensemble_preds = (ensemble_probs >= 0.5).astype(int)

    print(f"  -> Ensemble Test AUC: {ensemble_auc:.4f}")
    print(f"  -> Risk tier distribution:")
    tiers = [ensemble.score_to_risk_tier(p) for p in ensemble_probs]
    for tier in ["critical", "watch", "stable"]:
        count = tiers.count(tier)
        print(f"     {tier}: {count} ({count/len(tiers)*100:.1f}%)")

    # ─────────────────────────────────────────────
    # Step 6: SHAP + LIME Explainability
    # ─────────────────────────────────────────────
    print("\n[6/8] Computing SHAP + LIME explanations...")

    # SHAP
    try:
        shap_explainer = SHAPExplainer(xgb_model.get_booster(), feature_names)
        sample_explanations = shap_explainer.explain_batch(X_test[:3])
        print("  [SHAP] Explanations:")
        for i, exp in enumerate(sample_explanations):
            print(f"    Sample {i+1}: {exp['explanation']}")
    except Exception as e:
        print(f"  [SHAP] Error (non-fatal): {e}")

    # LIME
    try:
        from ml.lime_explainer import LIMEExplainer
        lime_explainer = LIMEExplainer(
            predict_fn=xgb_model.predict_proba,
            feature_names=feature_names,
            training_data=X_train,
        )
        lime_explanations = lime_explainer.explain_batch(X_test[:3], num_samples=200)
        print("  [LIME] Explanations:")
        for i, exp in enumerate(lime_explanations):
            if "explanation" in exp:
                print(f"    Sample {i+1}: {exp['explanation']}")
    except Exception as e:
        print(f"  [LIME] Error (non-fatal): {e}")

    # ─────────────────────────────────────────────
    # Step 7: Fairness Audit (Fairlearn + AIF360)
    # ─────────────────────────────────────────────
    print("\n[7/8] Running fairness audit (Fairlearn + AIF360)...")
    fairness_results = {}
    try:
        from sqlalchemy import create_engine
        engine = create_engine(PostgresConfig.get_url())
        demo_df = pd.read_sql(
            "SELECT customer_id, age, gender, region, income_bracket FROM customers",
            engine,
        )
        test_cids = customer_ids[idx_test]
        demo_test = demo_df[demo_df["customer_id"].isin(test_cids)].reset_index(drop=True)

        if len(demo_test) >= len(X_test):
            demo_test = demo_test.iloc[:len(X_test)]
            fairness_results = run_bias_audit(xgb_model, X_test, y_test, demo_test)
            print(f"  -> Verdict: {fairness_results.get('verdict', 'N/A')}")
            print(f"  -> Frameworks: {fairness_results.get('frameworks_used', [])}")
        else:
            print("  -> Insufficient demographic data")
    except Exception as e:
        print(f"  -> Fairness error (non-fatal): {e}")

    # ─────────────────────────────────────────────
    # Step 8: Register with MLflow
    # ─────────────────────────────────────────────
    print("\n[8/8] Registering models with MLflow...")
    try:
        from ml.mlflow_registry import log_ensemble_run, log_training_run

        # Log individual model runs
        xgb_run_id = log_training_run("xgboost", xgb_metrics, xgb_metrics)
        lgb_run_id = log_training_run("lightgbm", lgb_metrics, lgb_metrics)
        if lstm_metrics:
            lstm_run_id = log_training_run("lstm", lstm_metrics, lstm_metrics)

        # Log ensemble run
        ensemble_metrics_dict = {"ensemble_auc": ensemble_auc}
        ensemble_run_id = log_ensemble_run(
            xgb_metrics, lgb_metrics, lstm_metrics, ensemble_metrics_dict, fairness_results
        )
        print(f"  -> MLflow runs logged: XGBoost={xgb_run_id}, LightGBM={lgb_run_id}")
        print(f"  -> Ensemble run: {ensemble_run_id}")
    except Exception as e:
        print(f"  -> MLflow error (non-fatal): {e}")

    # ─────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TRAINING PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  XGBoost AUC:  {xgb_metrics['train_auc']:.4f} (CV: {xgb_metrics['cv_auc_mean']:.4f})")
    print(f"  LightGBM AUC: {lgb_metrics['train_auc']:.4f} (CV: {lgb_metrics['cv_auc_mean']:.4f})")
    if lstm_metrics:
        print(f"  LSTM AUC:     {lstm_metrics.get('train_auc', 'N/A')}")
    print(f"  Ensemble AUC: {ensemble_auc:.4f}")
    print(f"  Models saved: {MODEL_DIR}")
    print(f"  Explainers:   SHAP + LIME")
    print(f"  Fairness:     Fairlearn + AIF360")
    print("=" * 70)

    return {
        "xgboost_metrics": xgb_metrics,
        "lightgbm_metrics": lgb_metrics,
        "lstm_metrics": lstm_metrics,
        "ensemble_auc": ensemble_auc,
        "fairness": fairness_results,
    }


if __name__ == "__main__":
    train_pipeline()
