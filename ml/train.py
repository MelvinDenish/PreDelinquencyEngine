"""
ML Training Pipeline
Orchestrates the full model training workflow:
1. Build datasets from PostgreSQL
2. Train XGBoost on tabular features
3. Train LSTM on temporal sequences
4. Evaluate ensemble
5. Run SHAP explainability
6. Run fairness audit
7. Register models in MLflow
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config.settings import ModelConfig, MLflowConfig, PostgresConfig
from ml.dataset_builder import build_training_dataset, build_temporal_dataset
from ml.xgboost_model import XGBoostDelinquencyModel
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
    print("=" * 70)

    # ─────────────────────────────────────────────
    # Step 1: Build datasets
    # ─────────────────────────────────────────────
    print("\n[1/7] Building training datasets...")
    X_tab, y_tab, feature_names, customer_ids = build_training_dataset()
    if X_tab is None:
        print("ERROR: Could not build training dataset. Ensure data is generated and features computed.")
        return

    X_seq, y_seq, cids_seq = build_temporal_dataset()

    # Train/test split
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_tab, y_tab, np.arange(len(y_tab)),
        test_size=0.2, stratify=y_tab, random_state=42,
    )

    print(f"  Train: {len(X_train)} samples, Test: {len(X_test)} samples")

    # ─────────────────────────────────────────────
    # Step 2: Train XGBoost
    # ─────────────────────────────────────────────
    print("\n[2/7] Training XGBoost model...")
    xgb_model = XGBoostDelinquencyModel()
    xgb_metrics = xgb_model.train(X_train, y_train, feature_names, X_test, y_test)
    xgb_model.save()
    print(f"  -> Train AUC: {xgb_metrics['train_auc']:.4f}")
    print(f"  -> CV AUC: {xgb_metrics['cv_auc_mean']:.4f} +/- {xgb_metrics['cv_auc_std']:.4f}")
    if "val_auc" in xgb_metrics:
        print(f"  -> Test AUC: {xgb_metrics['val_auc']:.4f}")

    if xgb_metrics.get("feature_importance"):
        print("  -> Top features:")
        for feat, imp in list(xgb_metrics["feature_importance"].items())[:5]:
            print(f"     {feat}: {imp:.4f}")

    # ─────────────────────────────────────────────
    # Step 3: Train LSTM
    # ─────────────────────────────────────────────
    lstm_model = None
    lstm_metrics = {}
    if X_seq is not None and len(X_seq) > 50:
        print("\n[3/7] Training LSTM model...")

        # Split temporal data
        seq_train_idx, seq_test_idx = train_test_split(
            np.arange(len(X_seq)), test_size=0.2, stratify=y_seq, random_state=42,
        )

        lstm_model = LSTMDelinquencyModel(
            input_size=X_seq.shape[2],
            hidden_size=64,
            num_layers=2,
            epochs=30,
            batch_size=64,
        )
        lstm_metrics = lstm_model.train(
            X_seq[seq_train_idx], y_seq[seq_train_idx],
            X_seq[seq_test_idx], y_seq[seq_test_idx],
        )
        lstm_model.save()
        print(f"  -> Train AUC: {lstm_metrics.get('train_auc', 'N/A')}")
        print(f"  -> Best Val AUC: {lstm_metrics.get('best_val_auc', 'N/A')}")
    else:
        print("\n[3/7] Skipping LSTM (insufficient temporal data)")

    # ─────────────────────────────────────────────
    # Step 4: Evaluate Ensemble
    # ─────────────────────────────────────────────
    print("\n[4/7] Evaluating ensemble model...")
    ensemble = EnsembleScorer()

    xgb_test_probs = xgb_model.predict_proba(X_test)

    if lstm_model is not None and X_seq is not None:
        # Map test customers to their temporal sequences
        test_customer_ids = customer_ids[idx_test]
        seq_customer_ids = cids_seq

        lstm_test_probs = np.zeros(len(X_test))
        has_lstm = np.zeros(len(X_test), dtype=bool)

        for i, cid in enumerate(test_customer_ids):
            seq_idx = np.where(seq_customer_ids == cid)[0]
            if len(seq_idx) > 0:
                lstm_test_probs[i] = lstm_model.predict_proba(X_seq[seq_idx[0]:seq_idx[0]+1])[0]
                has_lstm[i] = True

        ensemble_probs = ensemble.combine_batch(xgb_test_probs, lstm_test_probs)
    else:
        ensemble_probs = xgb_test_probs

    from sklearn.metrics import roc_auc_score, classification_report
    ensemble_auc = roc_auc_score(y_test, ensemble_probs)
    ensemble_preds = (ensemble_probs >= 0.5).astype(int)
    print(f"  -> Ensemble Test AUC: {ensemble_auc:.4f}")
    print(f"  -> Risk tier distribution:")
    tiers = [ensemble.score_to_risk_tier(p) for p in ensemble_probs]
    for tier in ["critical", "watch", "stable"]:
        count = tiers.count(tier)
        print(f"     {tier}: {count} ({count/len(tiers)*100:.1f}%)")

    # ─────────────────────────────────────────────
    # Step 5: SHAP Explainability
    # ─────────────────────────────────────────────
    print("\n[5/7] Computing SHAP explanations...")
    try:
        explainer = SHAPExplainer(xgb_model.get_booster(), feature_names)
        sample_explanations = explainer.explain_batch(X_test[:5])
        for i, exp in enumerate(sample_explanations):
            print(f"  -> Sample {i+1}: {exp['explanation']}")
    except Exception as e:
        print(f"  -> SHAP error (non-fatal): {e}")

    # ─────────────────────────────────────────────
    # Step 6: Fairness Audit
    # ─────────────────────────────────────────────
    print("\n[6/7] Running fairness audit...")
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
            print(f"  -> Fairness verdict: {fairness_results.get('verdict', 'N/A')}")
        else:
            print("  -> Insufficient demographic data for fairness audit")
            fairness_results = {}
    except Exception as e:
        print(f"  -> Fairness audit error (non-fatal): {e}")
        fairness_results = {}

    # ─────────────────────────────────────────────
    # Step 7: Register with MLflow
    # ─────────────────────────────────────────────
    print("\n[7/7] Registering models with MLflow...")
    try:
        import mlflow

        mlflow.set_tracking_uri(MLflowConfig.TRACKING_URI)
        mlflow.set_experiment(MLflowConfig.EXPERIMENT_NAME)

        with mlflow.start_run(run_name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log params
            mlflow.log_params({
                "xgboost_weight": ensemble.xgboost_weight,
                "lstm_weight": ensemble.lstm_weight,
                "n_customers": len(customer_ids),
                "n_features": len(feature_names),
                "train_size": len(X_train),
                "test_size": len(X_test),
            })

            # Log metrics
            mlflow.log_metrics({
                "xgboost_train_auc": xgb_metrics["train_auc"],
                "xgboost_cv_auc": xgb_metrics["cv_auc_mean"],
                "ensemble_test_auc": ensemble_auc,
            })

            if lstm_metrics:
                mlflow.log_metrics({
                    "lstm_train_auc": lstm_metrics.get("train_auc", 0),
                    "lstm_val_auc": lstm_metrics.get("best_val_auc", 0),
                })

            # Log model artifacts
            mlflow.log_artifact(os.path.join(MODEL_DIR, "xgboost_model.joblib"))
            if lstm_model:
                mlflow.log_artifact(os.path.join(MODEL_DIR, "lstm_model.pt"))

            run_id = mlflow.active_run().info.run_id
            print(f"  -> MLflow run ID: {run_id}")

    except Exception as e:
        print(f"  -> MLflow registration error (non-fatal): {e}")

    # ─────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TRAINING PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  XGBoost AUC:  {xgb_metrics['train_auc']:.4f}")
    if lstm_metrics:
        print(f"  LSTM AUC:     {lstm_metrics.get('train_auc', 'N/A')}")
    print(f"  Ensemble AUC: {ensemble_auc:.4f}")
    print(f"  Models saved: {MODEL_DIR}")
    print("=" * 70)

    return {
        "xgboost_metrics": xgb_metrics,
        "lstm_metrics": lstm_metrics,
        "ensemble_auc": ensemble_auc,
    }


if __name__ == "__main__":
    train_pipeline()


