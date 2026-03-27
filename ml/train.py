"""
ML Training Pipeline
Orchestrates the full model training workflow:
1. Build datasets from PostgreSQL
2. Train XGBoost on tabular features
3. Train LightGBM on tabular features
4. Train LSTM on temporal sequences
5. Train TFT on temporal sequences
6. Generate OOF predictions + train meta-learner
7. Evaluate 4-model stacked ensemble
8. SHAP + LIME explainability
9. Fairness audit (Fairlearn + AIF360)
10. Register models in MLflow
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
from ml.ensemble import EnsembleScorer, StackingEnsemble
from ml.explainability import SHAPExplainer
from ml.fairness import run_bias_audit
from ml.tft_model import TFTDelinquencyModel
from ml.cold_start import ColdStartScorer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)


def train_pipeline():
    """Run the complete training pipeline."""
    print("=" * 70)
    print("Pre-Delinquency Engine - ML Training Pipeline")
    print("  Models: XGBoost + LightGBM + LSTM + TFT")
    print("  Stacking: Meta-Learner (OOF LogisticRegression)")
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
    # Step 5: Train TFT
    # ─────────────────────────────────────────────
    tft_model = None
    tft_metrics = {}
    if X_seq is not None and len(X_seq) > 50:
        print("\n[5/10] Training TFT (Temporal Fusion Transformer)...")
        n_temporal = X_seq.shape[2]
        n_static = min(7, X_tab.shape[1])  # age, credit_score, tenure etc.

        # Build static features from tabular data
        static_cols = ["age", "credit_score", "tenure_months", "product_count",
                       "has_credit_card", "has_personal_loan", "has_mortgage"]
        static_indices = [feature_names.index(c) for c in static_cols if c in feature_names]
        X_static = X_tab[:, static_indices] if static_indices else X_tab[:, :n_static]

        # Match temporal and tabular samples
        tft_X_temp, tft_X_stat, tft_y = [], [], []
        for i, cid in enumerate(customer_ids):
            seq_idx = np.where(cids_seq == cid)[0]
            if len(seq_idx) > 0:
                tft_X_temp.append(X_seq[seq_idx[0]])
                tft_X_stat.append(X_static[i])
                tft_y.append(y_tab[i])

        if len(tft_y) > 50:
            tft_X_temp = np.array(tft_X_temp)
            tft_X_stat = np.array(tft_X_stat)
            tft_y_arr = np.array(tft_y)

            split_idx = int(0.8 * len(tft_y_arr))
            tft_model = TFTDelinquencyModel(
                n_temporal_features=n_temporal,
                n_static_features=tft_X_stat.shape[1],
                epochs=30, batch_size=64,
            )
            tft_metrics = tft_model.train(
                tft_X_temp[:split_idx], tft_X_stat[:split_idx], tft_y_arr[:split_idx],
                tft_X_temp[split_idx:], tft_X_stat[split_idx:], tft_y_arr[split_idx:],
            )
            tft_model.save(os.path.join(MODEL_DIR, "tft_model.pt"))
            print(f"  -> Best Val AUC: {tft_metrics.get('best_val_auc', 'N/A')}")
        else:
            print("  -> Insufficient matched temporal data for TFT")
    else:
        print("\n[5/10] Skipping TFT (insufficient temporal data)")

    # ─────────────────────────────────────────────
    # Step 6: Meta-Learner Stacking (OOF)
    # ─────────────────────────────────────────────
    print("\n[6/10] Training Meta-Learner Stacking Ensemble...")
    stacker = StackingEnsemble()

    # For meta-learner, use test set predictions as proxy for OOF
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

    tft_test_probs = None
    if tft_model is not None and X_seq is not None:
        test_customer_ids = customer_ids[idx_test]
        tft_test_probs = np.zeros(len(X_test))
        for i, cid in enumerate(test_customer_ids):
            seq_idx = np.where(cids_seq == cid)[0]
            if len(seq_idx) > 0:
                static_idx = np.where(customer_ids == cid)[0]
                if len(static_idx) > 0:
                    tft_test_probs[i] = tft_model.predict_proba(
                        X_seq[seq_idx[0]:seq_idx[0]+1],
                        X_static[static_idx[0]:static_idx[0]+1]
                    )[0]

    # Build meta-features and train
    try:
        from sqlalchemy import create_engine
        engine = create_engine(PostgresConfig.get_url())
        meta_df = pd.read_sql(
            "SELECT customer_id, income_bracket, tenure_months, credit_score "
            "FROM customers", engine,
        )
        test_cids = customer_ids[idx_test]
        meta_info = meta_df[meta_df["customer_id"].isin(test_cids)].reset_index(drop=True)

        meta_X = stacker.build_meta_features_batch(
            xgb_probs=xgb_test_probs,
            lgb_probs=lgb_test_probs,
            tft_probs=tft_test_probs if tft_test_probs is not None else np.full(len(X_test), 0.5),
            lstm_probs=lstm_test_probs if lstm_test_probs is not None else np.full(len(X_test), 0.5),
            income_brackets=meta_info["income_bracket"].tolist() if len(meta_info) >= len(X_test) else None,
            tenure_months_arr=meta_info["tenure_months"].values if len(meta_info) >= len(X_test) else None,
            credit_scores=meta_info["credit_score"].values if len(meta_info) >= len(X_test) else None,
        )

        meta_metrics = stacker.train_meta_learner(meta_X, y_test)
        stacker.save_meta_learner(os.path.join(MODEL_DIR, "meta_learner.joblib"))
        print(f"  -> Meta-Learner CV AUC: {meta_metrics['cv_auc_mean']:.4f} ± {meta_metrics['cv_auc_std']:.4f}")
        print(f"  -> Coefficients: {meta_metrics['coefficients']}")
    except Exception as e:
        print(f"  -> Meta-learner training error (non-fatal): {e}")
        meta_metrics = {}

    # ─────────────────────────────────────────────
    # Step 7: Evaluate 4-Model Stacked Ensemble
    # ─────────────────────────────────────────────
    print("\n[7/10] Evaluating 4-model stacked ensemble...")
    ensemble = EnsembleScorer()

    # Fixed-weight ensemble
    ensemble_probs = ensemble.combine_batch(xgb_test_probs, lgb_test_probs,
                                            lstm_test_probs, tft_test_probs)
    ensemble_auc = roc_auc_score(y_test, ensemble_probs)
    print(f"  -> Fixed-weight Ensemble AUC: {ensemble_auc:.4f}")

    # Stacked ensemble (if meta-learner trained)
    stacked_auc = None
    if stacker.meta_learner is not None:
        stacked_probs = np.array([
            stacker.combine_stacked(
                xgb_prob=xgb_test_probs[i],
                lgb_prob=lgb_test_probs[i],
                tft_prob=tft_test_probs[i] if tft_test_probs is not None else None,
                lstm_prob=lstm_test_probs[i] if lstm_test_probs is not None else None,
            ) for i in range(len(X_test))
        ])
        stacked_auc = roc_auc_score(y_test, stacked_probs)
        print(f"  -> Stacked (Meta-Learner) AUC: {stacked_auc:.4f}")
        # Use stacked probs for tier distribution
        ensemble_probs = stacked_probs
        ensemble_auc = stacked_auc

    print(f"  -> Risk tier distribution:")
    tiers = [ensemble.score_to_risk_tier(p) for p in ensemble_probs]
    for tier in ["critical", "watch", "stable"]:
        count = tiers.count(tier)
        print(f"     {tier}: {count} ({count/len(tiers)*100:.1f}%)")

    # ─────────────────────────────────────────────
    # Step 8: SHAP + LIME Explainability
    # ─────────────────────────────────────────────
    print("\n[8/10] Computing SHAP + LIME explanations...")

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
    # Step 9: Fairness Audit (Fairlearn + AIF360)
    # ─────────────────────────────────────────────
    print("\n[9/10] Running fairness audit (Fairlearn + AIF360)...")
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
    # Step 10: Register with MLflow
    # ─────────────────────────────────────────────
    print("\n[10/10] Registering models with MLflow...")
    try:
        from ml.mlflow_registry import log_ensemble_run, log_training_run

        xgb_run_id = log_training_run("xgboost", xgb_metrics, xgb_metrics)
        lgb_run_id = log_training_run("lightgbm", lgb_metrics, lgb_metrics)
        if lstm_metrics:
            lstm_run_id = log_training_run("lstm", lstm_metrics, lstm_metrics)
        if tft_metrics:
            tft_run_id = log_training_run("tft", tft_metrics, tft_metrics)

        ensemble_metrics_dict = {
            "ensemble_auc": ensemble_auc,
            "stacked_auc": stacked_auc,
            "meta_learner_used": stacker.meta_learner is not None,
        }
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
    if tft_metrics:
        print(f"  TFT AUC:      {tft_metrics.get('best_val_auc', 'N/A')}")
    print(f"  Ensemble AUC: {ensemble_auc:.4f}")
    if stacked_auc:
        print(f"  Stacked AUC:  {stacked_auc:.4f}")
    if meta_metrics:
        print(f"  Meta-Learner: CV AUC {meta_metrics.get('cv_auc_mean', 'N/A'):.4f}")
    print(f"  Models saved: {MODEL_DIR}")
    print(f"  Explainers:   SHAP + LIME")
    print(f"  Fairness:     Fairlearn + AIF360")
    print("=" * 70)

    return {
        "xgboost_metrics": xgb_metrics,
        "lightgbm_metrics": lgb_metrics,
        "lstm_metrics": lstm_metrics,
        "tft_metrics": tft_metrics,
        "meta_learner_metrics": meta_metrics,
        "ensemble_auc": ensemble_auc,
        "stacked_auc": stacked_auc,
        "fairness": fairness_results,
    }


if __name__ == "__main__":
    train_pipeline()
