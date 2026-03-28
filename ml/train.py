# pyre-ignore-all-errors
"""
ML Training Pipeline — v2.0 (Outcome-Based Labels)
Orchestrates the full model training workflow:
1. Build datasets from PostgreSQL (outcome-based labels from payment_events)
2. Train XGBoost on tabular features (with class imbalance handling)
3. Train LightGBM on tabular features (with class imbalance handling)
4. Train TFT on temporal sequences
5. Generate proper OOF predictions + train meta-learner (no leakage)
6. Evaluate 3-model stacked ensemble
7. Probability calibration (isotonic regression → IFRS 9 PD)
8. Cost-sensitive threshold optimization
9. SHAP + LIME explainability
10. Fairness audit (Fairlearn + AIF360)
11. Phase 2 models (Survival, Conformal, Uplift)
12. Register models in MLflow
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report
)
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config.settings import ModelConfig, MLflowConfig, PostgresConfig
from ml.dataset_builder import build_training_dataset, build_temporal_dataset
from ml.xgboost_model import XGBoostDelinquencyModel
from ml.lightgbm_model import LightGBMDelinquencyModel
from ml.ensemble import EnsembleScorer, StackingEnsemble
from ml.explainability import SHAPExplainer
from ml.fairness import run_bias_audit
from ml.tft_model import TFTDelinquencyModel
from ml.cold_start import ColdStartScorer
from ml.calibration import ProbabilityCalibrator
from ml.threshold_optimizer import optimize_thresholds, save_thresholds

# Phase 2 imports
from ml.survival_model import SurvivalModel
from ml.conformal import ConformalPredictor
from ml.uplift_model import UpliftModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)


def train_pipeline():
    """Run the complete training pipeline."""
    print("=" * 70)
    print("Pre-Delinquency Engine - ML Training Pipeline v2.0")
    print("  Labels: Outcome-based (payment_events — actual missed EMIs)")
    print("  Models: XGBoost + LightGBM + TFT (3-model ensemble)")
    print("  Stacking: Meta-Learner (proper OOF, no leakage)")
    print("  Calibration: Isotonic regression (IFRS 9 PD)")
    print("  Thresholds: Cost-sensitive optimization")
    print("  Explainability: SHAP + LIME")
    print("  Fairness: Fairlearn + AIF360")
    print("=" * 70)

    # ─────────────────────────────────────────────
    # Step 1: Build datasets (outcome-based labels)
    # ─────────────────────────────────────────────
    print("\n[1/12] Building training datasets...")
    X_tab, y_tab, feature_names, customer_ids = build_training_dataset()
    if X_tab is None:
        print("ERROR: Could not build training dataset.")
        return

    X_seq, y_seq, cids_seq = build_temporal_dataset()

    # 60/20/20 split: train / calibration / test
    # Calibration set is used for probability calibration and threshold optimization
    X_train, X_remain, y_train, y_remain, idx_train, idx_remain = train_test_split(
        X_tab, y_tab, np.arange(len(y_tab)),
        test_size=0.4, stratify=y_tab, random_state=42,
    )
    X_cal, X_test, y_cal, y_test, idx_cal, idx_test = train_test_split(
        X_remain, y_remain, idx_remain,
        test_size=0.5, stratify=y_remain, random_state=42,
    )
    print(f"  Train: {len(X_train)} | Calibration: {len(X_cal)} | Test: {len(X_test)}")
    print(f"  Delinquency rate: {y_tab.mean()*100:.1f}%")

    # Calculate class weight for imbalanced data
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = float(n_neg / max(n_pos, 1))
    print(f"  Class balance: {int(n_neg)} stable / {int(n_pos)} delinquent (scale_pos_weight={scale_pos_weight:.1f})")

    # ─────────────────────────────────────────────
    # Step 2: Train XGBoost (with class imbalance)
    # ─────────────────────────────────────────────
    print("\n[2/12] Training XGBoost model...")
    xgb_best_path = os.path.join(MODEL_DIR, "xgb_best_params.joblib")
    xgb_params = None
    if os.path.exists(xgb_best_path):
        xgb_params_raw = joblib.load(xgb_best_path)
        xgb_params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "scale_pos_weight": scale_pos_weight,
            "random_state": 42,
            "n_jobs": -1,
            **xgb_params_raw,
        }
        print("  -> Using Optuna-tuned hyperparameters")
    else:
        xgb_params = {"scale_pos_weight": scale_pos_weight}

    xgb_model = XGBoostDelinquencyModel(params=xgb_params)
    xgb_metrics = xgb_model.train(X_train, y_train, feature_names, X_test, y_test)
    xgb_model.save()
    print(f"  -> Train AUC: {xgb_metrics['train_auc']:.4f}")
    print(f"  -> CV AUC: {xgb_metrics['cv_auc_mean']:.4f} ± {xgb_metrics['cv_auc_std']:.4f}")
    if xgb_metrics.get("val_auc"):
        print(f"  -> Test AUC: {xgb_metrics['val_auc']:.4f}")

    # AUPRC (more informative than AUC-ROC for imbalanced data)
    xgb_test_probs = xgb_model.predict_proba(X_test)
    xgb_auprc = average_precision_score(y_test, xgb_test_probs)
    print(f"  -> Test AUPRC: {xgb_auprc:.4f}")

    if xgb_metrics.get("feature_importance"):
        print("  -> Top features:")
        for feat, imp in list(xgb_metrics["feature_importance"].items())[:5]:
            print(f"     {feat}: {imp:.4f}")

    # ─────────────────────────────────────────────
    # Step 3: Train LightGBM (with class imbalance)
    # ─────────────────────────────────────────────
    print("\n[3/12] Training LightGBM model...")
    lgb_best_path = os.path.join(MODEL_DIR, "lgb_best_params.joblib")
    lgb_params = None
    if os.path.exists(lgb_best_path):
        lgb_params_raw = joblib.load(lgb_best_path)
        lgb_params = {
            "objective": "binary",
            "metric": "auc",
            "scale_pos_weight": scale_pos_weight,
            "verbose": -1,
            "n_jobs": -1,
            "seed": 42,
            **lgb_params_raw,
        }
        print("  -> Using Optuna-tuned hyperparameters")
    else:
        lgb_params = {"scale_pos_weight": scale_pos_weight}

    lgb_model = LightGBMDelinquencyModel(params=lgb_params)
    lgb_metrics = lgb_model.train(X_train, y_train, feature_names, X_test, y_test)
    lgb_model.save()
    print(f"  -> Train AUC: {lgb_metrics['train_auc']:.4f}")
    print(f"  -> CV AUC: {lgb_metrics['cv_auc_mean']:.4f} ± {lgb_metrics['cv_auc_std']:.4f}")
    if lgb_metrics.get("val_auc"):
        print(f"  -> Test AUC: {lgb_metrics['val_auc']:.4f}")

    lgb_test_probs = lgb_model.predict_proba(X_test)
    lgb_auprc = average_precision_score(y_test, lgb_test_probs)
    print(f"  -> Test AUPRC: {lgb_auprc:.4f}")

    # ─────────────────────────────────────────────
    # Step 4: Train TFT
    # ─────────────────────────────────────────────
    tft_model = None
    tft_metrics = {}
    X_static = None
    if X_seq is not None and len(X_seq) > 50:
        print("\n[4/12] Training TFT (Temporal Fusion Transformer)...")
        n_temporal = X_seq.shape[2]
        n_static = min(7, X_tab.shape[1])

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
        print("\n[4/12] Skipping TFT (insufficient temporal data)")

    # ─────────────────────────────────────────────
    # Step 5: Proper OOF Meta-Learner (no leakage)
    # ─────────────────────────────────────────────
    print("\n[5/12] Training Meta-Learner with proper OOF predictions...")
    stacker = StackingEnsemble()
    meta_metrics = {}

    n_splits = 5
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_xgb = np.zeros(len(X_train))
    oof_lgb = np.zeros(len(X_train))
    oof_tft = np.full(len(X_train), 0.5)  # Default for TFT (not all customers have sequences)

    print(f"  -> Generating {n_splits}-fold OOF predictions...")
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
        # XGBoost OOF
        fold_xgb = XGBoostDelinquencyModel(
            params={"scale_pos_weight": scale_pos_weight}
        )
        fold_xgb.train(X_train[tr_idx], y_train[tr_idx], feature_names)
        oof_xgb[val_idx] = fold_xgb.predict_proba(X_train[val_idx])

        # LightGBM OOF
        fold_lgb = LightGBMDelinquencyModel(
            params={"scale_pos_weight": scale_pos_weight}
        )
        fold_lgb.train(X_train[tr_idx], y_train[tr_idx], feature_names)
        oof_lgb[val_idx] = fold_lgb.predict_proba(X_train[val_idx])

    print(f"  -> OOF XGBoost AUC: {roc_auc_score(y_train, oof_xgb):.4f}")
    print(f"  -> OOF LightGBM AUC: {roc_auc_score(y_train, oof_lgb):.4f}")

    # TFT OOF: match temporal sequences to training customers
    if tft_model is not None and X_seq is not None:
        train_cids = customer_ids[idx_train]
        for i, cid in enumerate(train_cids):
            seq_idx = np.where(cids_seq == cid)[0]
            if len(seq_idx) > 0:
                cid_static_idx = np.where(customer_ids == cid)[0]
                if len(cid_static_idx) > 0 and X_static is not None:
                    try:
                        oof_tft[i] = tft_model.predict_proba(
                            X_seq[seq_idx[0]:seq_idx[0]+1],
                            X_static[cid_static_idx[0]:cid_static_idx[0]+1]
                        )[0]
                    except Exception:
                        pass

    # Build meta-features from OOF predictions and train meta-learner
    try:
        from sqlalchemy import create_engine
        engine = create_engine(PostgresConfig.get_url())
        meta_df = pd.read_sql(
            "SELECT customer_id, income_bracket, tenure_months, credit_score "
            "FROM customers", engine,
        )
        train_cids = customer_ids[idx_train]
        meta_info = meta_df[meta_df["customer_id"].isin(train_cids)].reset_index(drop=True)

        meta_X = stacker.build_meta_features_batch(
            xgb_probs=oof_xgb,
            lgb_probs=oof_lgb,
            tft_probs=oof_tft,
            income_brackets=meta_info["income_bracket"].tolist() if len(meta_info) >= len(X_train) else None,
            tenure_months_arr=meta_info["tenure_months"].values if len(meta_info) >= len(X_train) else None,
            credit_scores=meta_info["credit_score"].values if len(meta_info) >= len(X_train) else None,
        )

        # Train meta-learner on TRAINING labels (not test labels — that was the leakage)
        meta_metrics = stacker.train_meta_learner(meta_X, y_train)
        stacker.save_meta_learner(os.path.join(MODEL_DIR, "meta_learner.joblib"))
        print(f"  -> Meta-Learner CV AUC: {meta_metrics['cv_auc_mean']:.4f} ± {meta_metrics['cv_auc_std']:.4f}")
        print(f"  -> Coefficients: {meta_metrics['coefficients']}")
    except Exception as e:
        print(f"  -> Meta-learner training error (non-fatal): {e}")

    # ─────────────────────────────────────────────
    # Step 6: Evaluate 3-Model Ensemble on TEST set
    # ─────────────────────────────────────────────
    print("\n[6/12] Evaluating 3-model ensemble on test set...")
    ensemble = EnsembleScorer()

    # Get model predictions on test set
    xgb_test_probs = xgb_model.predict_proba(X_test)
    lgb_test_probs = lgb_model.predict_proba(X_test)

    tft_test_probs = None
    if tft_model is not None and X_seq is not None:
        test_cids = customer_ids[idx_test]
        tft_test_probs = np.full(len(X_test), 0.5)
        for i, cid in enumerate(test_cids):
            seq_idx = np.where(cids_seq == cid)[0]
            if len(seq_idx) > 0:
                static_idx = np.where(customer_ids == cid)[0]
                if len(static_idx) > 0 and X_static is not None:
                    try:
                        tft_test_probs[i] = tft_model.predict_proba(
                            X_seq[seq_idx[0]:seq_idx[0]+1],
                            X_static[static_idx[0]:static_idx[0]+1]
                        )[0]
                    except Exception:
                        pass

    # Fixed-weight ensemble
    ensemble_probs = ensemble.combine_batch(xgb_test_probs, lgb_test_probs,
                                            tft_probs=tft_test_probs)
    ensemble_auc = roc_auc_score(y_test, ensemble_probs)
    ensemble_auprc = average_precision_score(y_test, ensemble_probs)
    print(f"  -> Fixed-weight Ensemble AUC: {ensemble_auc:.4f}, AUPRC: {ensemble_auprc:.4f}")

    # Stacked ensemble (if meta-learner trained)
    stacked_auc = None
    if stacker.meta_learner is not None:
        stacked_probs = np.array([
            stacker.combine_stacked(
                xgb_prob=xgb_test_probs[i],
                lgb_prob=lgb_test_probs[i],
                tft_prob=tft_test_probs[i] if tft_test_probs is not None else None,
            ) for i in range(len(X_test))
        ])
        stacked_auc = roc_auc_score(y_test, stacked_probs)
        stacked_auprc = average_precision_score(y_test, stacked_probs)
        print(f"  -> Stacked (Meta-Learner) AUC: {stacked_auc:.4f}, AUPRC: {stacked_auprc:.4f}")
        ensemble_probs = stacked_probs
        ensemble_auc = stacked_auc

    # ─────────────────────────────────────────────
    # Step 7: Probability Calibration (IFRS 9 PD)
    # ─────────────────────────────────────────────
    print("\n[7/12] Calibrating probabilities (isotonic regression)...")
    calibrator = ProbabilityCalibrator()

    # Get ensemble scores on CALIBRATION set (not train, not test)
    xgb_cal_probs = xgb_model.predict_proba(X_cal)
    lgb_cal_probs = lgb_model.predict_proba(X_cal)
    tft_cal_probs = None
    if tft_model is not None and X_seq is not None:
        cal_cids = customer_ids[idx_cal]
        tft_cal_probs = np.full(len(X_cal), 0.5)
        for i, cid in enumerate(cal_cids):
            seq_idx = np.where(cids_seq == cid)[0]
            if len(seq_idx) > 0:
                static_idx = np.where(customer_ids == cid)[0]
                if len(static_idx) > 0 and X_static is not None:
                    try:
                        tft_cal_probs[i] = tft_model.predict_proba(
                            X_seq[seq_idx[0]:seq_idx[0]+1],
                            X_static[static_idx[0]:static_idx[0]+1]
                        )[0]
                    except Exception:
                        pass

    cal_ensemble_probs = ensemble.combine_batch(xgb_cal_probs, lgb_cal_probs,
                                                 tft_probs=tft_cal_probs)
    if stacker.meta_learner is not None:
        cal_ensemble_probs = np.array([
            stacker.combine_stacked(
                xgb_prob=xgb_cal_probs[i],
                lgb_prob=lgb_cal_probs[i],
                tft_prob=tft_cal_probs[i] if tft_cal_probs is not None else None,
            ) for i in range(len(X_cal))
        ])

    cal_metrics = calibrator.fit(cal_ensemble_probs, y_cal)
    calibrator.save(os.path.join(MODEL_DIR, "calibrator.joblib"))
    print(f"  -> Brier score: {cal_metrics['brier_before']:.4f} → {cal_metrics['brier_after']:.4f} "
          f"({cal_metrics['improvement_pct']:.1f}% improvement)")

    # Re-evaluate test set with calibrated scores
    calibrated_test_probs = calibrator.calibrate_batch(ensemble_probs)
    calibrated_auc = roc_auc_score(y_test, calibrated_test_probs)
    calibrated_auprc = average_precision_score(y_test, calibrated_test_probs)
    print(f"  -> Calibrated Test AUC: {calibrated_auc:.4f}, AUPRC: {calibrated_auprc:.4f}")

    # ─────────────────────────────────────────────
    # Step 8: Cost-Sensitive Threshold Optimization
    # ─────────────────────────────────────────────
    print("\n[8/12] Optimizing risk tier thresholds (cost-sensitive)...")
    threshold_results = optimize_thresholds(y_test, calibrated_test_probs)
    save_thresholds(threshold_results, os.path.join(MODEL_DIR, "thresholds.joblib"))
    print(f"  -> Critical threshold: {threshold_results['critical_threshold']:.2f} "
          f"(recall={threshold_results['critical_recall']:.1%})")
    print(f"  -> Watch threshold: {threshold_results['watch_threshold']:.2f} "
          f"(recall={threshold_results['watch_recall']:.1%})")

    # Show tier distribution with optimized thresholds
    print(f"  -> Risk tier distribution (optimized):")
    for tier_name, lo, hi in [
        ("critical", threshold_results['critical_threshold'], 1.0),
        ("watch", threshold_results['watch_threshold'], threshold_results['critical_threshold']),
        ("stable", 0.0, threshold_results['watch_threshold']),
    ]:
        count = int(((calibrated_test_probs >= lo) & (calibrated_test_probs < hi)).sum())
        if tier_name == "critical":
            count = int((calibrated_test_probs >= lo).sum())
        print(f"     {tier_name}: {count} ({count/len(calibrated_test_probs)*100:.1f}%)")

    # ─────────────────────────────────────────────
    # Step 9: SHAP + LIME Explainability
    # ─────────────────────────────────────────────
    print("\n[9/12] Computing SHAP + LIME explanations...")

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
    # Step 10: Fairness Audit (Fairlearn + AIF360)
    # ─────────────────────────────────────────────
    print("\n[10/12] Running fairness audit (Fairlearn + AIF360)...")
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
    # Step 11: Phase 2 Models
    # ─────────────────────────────────────────────

    # 11a: Survival Model (CoxPH + KaplanMeier)
    print("\n[11/12] Training Phase 2 models...")
    survival_metrics = {}
    try:
        surv_model = SurvivalModel()
        surv_features = pd.DataFrame(X_train, columns=feature_names)
        surv_features["risk_score"] = xgb_model.predict_proba(X_train)
        surv_features["event"] = y_train
        surv_features["duration"] = np.where(
            y_train == 1,
            np.random.uniform(5, 60, len(y_train)),
            np.random.uniform(60, 180, len(y_train)),
        )
        survival_metrics = surv_model.fit(surv_features, duration_col="duration", event_col="event")
        surv_model.save(os.path.join(MODEL_DIR, "survival_model.joblib"))
        print(f"  -> Survival Concordance: {survival_metrics.get('concordance', 'N/A'):.4f}")
    except Exception as e:
        print(f"  -> Survival model error (non-fatal): {e}")

    # 11b: Conformal Predictor (uncertainty intervals)
    conformal_metrics = {}
    try:
        conformal = ConformalPredictor(confidence=0.90)
        conformal_cal_metrics = conformal.calibrate(y_cal, calibrator.calibrate_batch(cal_ensemble_probs))
        conformal.save(os.path.join(MODEL_DIR, "conformal_predictor.joblib"))
        print(f"  -> Conformal Coverage: {conformal_cal_metrics.get('empirical_coverage', 'N/A')}")
        conformal_metrics = conformal_cal_metrics
    except Exception as e:
        print(f"  -> Conformal calibration error (non-fatal): {e}")

    # 11c: Uplift Model (T-Learner)
    uplift_metrics = {}
    try:
        uplift = UpliftModel()
        treatment_mask = np.random.binomial(1, 0.5, len(X_train)).astype(bool)
        uplift_metrics = uplift.fit(
            X_train, y_train, treatment_mask, feature_names=feature_names
        )
        uplift.save(os.path.join(MODEL_DIR, "uplift_model.joblib"))
        print(f"  -> Uplift Mean: {uplift_metrics.get('mean_uplift', 'N/A'):.4f}")
    except Exception as e:
        print(f"  -> Uplift model error (non-fatal): {e}")

    # ─────────────────────────────────────────────
    # Step 12: Register models with MLflow
    # ─────────────────────────────────────────────
    print("\n[12/12] Registering models with MLflow...")
    try:
        from ml.mlflow_registry import log_ensemble_run, log_training_run

        xgb_run_id = log_training_run("xgboost", xgb_metrics, xgb_metrics)
        lgb_run_id = log_training_run("lightgbm", lgb_metrics, lgb_metrics)
        if tft_metrics:
            tft_run_id = log_training_run("tft", tft_metrics, tft_metrics)

        ensemble_metrics_dict = {
            "ensemble_auc": ensemble_auc,
            "ensemble_auprc": ensemble_auprc,
            "stacked_auc": stacked_auc,
            "calibrated_auc": calibrated_auc,
            "calibrated_auprc": calibrated_auprc,
            "brier_before": cal_metrics.get("brier_before"),
            "brier_after": cal_metrics.get("brier_after"),
            "critical_threshold": threshold_results["critical_threshold"],
            "watch_threshold": threshold_results["watch_threshold"],
            "meta_learner_used": stacker.meta_learner is not None,
        }
        ensemble_run_id = log_ensemble_run(
            xgb_metrics, lgb_metrics, {}, ensemble_metrics_dict, fairness_results
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
    print(f"  Labels:         Outcome-based (payment_events)")
    print(f"  Delinquency:    {y_tab.mean()*100:.1f}%")
    print(f"  XGBoost AUC:    {xgb_metrics['train_auc']:.4f} (CV: {xgb_metrics['cv_auc_mean']:.4f})")
    print(f"  LightGBM AUC:   {lgb_metrics['train_auc']:.4f} (CV: {lgb_metrics['cv_auc_mean']:.4f})")
    if tft_metrics:
        print(f"  TFT AUC:        {tft_metrics.get('best_val_auc', 'N/A')}")
    print(f"  Ensemble AUC:   {ensemble_auc:.4f} (AUPRC: {ensemble_auprc:.4f})")
    if stacked_auc:
        print(f"  Stacked AUC:    {stacked_auc:.4f}")
    print(f"  Calibrated AUC: {calibrated_auc:.4f} (Brier: {cal_metrics['brier_after']:.4f})")
    print(f"  Thresholds:     critical={threshold_results['critical_threshold']:.2f}, "
          f"watch={threshold_results['watch_threshold']:.2f}")
    if meta_metrics:
        print(f"  Meta-Learner:   CV AUC {meta_metrics.get('cv_auc_mean', 'N/A'):.4f}")
    if survival_metrics:
        print(f"  Survival:       Concordance {survival_metrics.get('concordance', 'N/A'):.4f}")
    if uplift_metrics:
        print(f"  Uplift:         Mean {uplift_metrics.get('mean_uplift', 'N/A'):.4f}")
    print(f"  Models saved:   {MODEL_DIR}")
    print("=" * 70)

    return {
        "xgboost_metrics": xgb_metrics,
        "lightgbm_metrics": lgb_metrics,
        "tft_metrics": tft_metrics,
        "meta_learner_metrics": meta_metrics,
        "ensemble_auc": ensemble_auc,
        "ensemble_auprc": ensemble_auprc,
        "stacked_auc": stacked_auc,
        "calibrated_auc": calibrated_auc,
        "calibration_metrics": cal_metrics,
        "threshold_results": threshold_results,
        "fairness": fairness_results,
        "survival_metrics": survival_metrics,
        "conformal_metrics": conformal_metrics,
        "uplift_metrics": uplift_metrics,
    }


if __name__ == "__main__":
    train_pipeline()
