"""
FastAPI Scoring Service
Provides REST endpoints for real-time delinquency risk scoring.
Integrates Feast feature retrieval, model inference, SHAP explainability,
and risk score storage.
"""
import os
import sys
import json
import logging
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd
import redis as redis_lib
import psycopg2
from psycopg2.extras import execute_values
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config.settings import (
    PostgresConfig, RedisConfig, ModelConfig, ScoringConfig, FeastConfig,
)
from ml.xgboost_model import XGBoostDelinquencyModel
from ml.lstm_model import LSTMDelinquencyModel
from ml.ensemble import EnsembleScorer
from ml.explainability import SHAPExplainer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ─────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────
app = FastAPI(
    title="Pre-Delinquency Intervention Engine - Scoring Service",
    description="Real-time delinquency risk scoring with SHAP explainability",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Global model instances (loaded on startup)
# ─────────────────────────────────────────────
xgb_model: Optional[XGBoostDelinquencyModel] = None
lstm_model: Optional[LSTMDelinquencyModel] = None
ensemble: Optional[EnsembleScorer] = None
shap_explainer: Optional[SHAPExplainer] = None
redis_client: Optional[redis_lib.Redis] = None


# ─────────────────────────────────────────────
# Request/Response Models
# ─────────────────────────────────────────────
class ScoreRequest(BaseModel):
    customer_id: str

class BatchScoreRequest(BaseModel):
    customer_ids: List[str]

class ScoreResponse(BaseModel):
    customer_id: str
    risk_score: float
    risk_tier: str
    credit_score_mapped: int
    xgboost_score: float
    lstm_score: Optional[float] = None
    ensemble_score: float
    top_shap_features: Optional[list] = None
    explanation: Optional[str] = None
    scored_at: str

class HealthResponse(BaseModel):
    status: str
    models_loaded: dict
    timestamp: str


# ─────────────────────────────────────────────
# Feature Retrieval
# ─────────────────────────────────────────────
def get_features_from_redis(customer_id: str) -> dict:
    """Retrieve features from Redis (streaming features)."""
    key = f"features:streaming:{customer_id}"
    data = redis_client.hgetall(key)
    if data:
        return {k.decode() if isinstance(k, bytes) else k:
                float(v.decode() if isinstance(v, bytes) else v)
                for k, v in data.items()}
    return {}


def get_features_from_db(customer_id: str) -> dict:
    """Retrieve features from PostgreSQL (batch + streaming)."""
    conn = psycopg2.connect(
        host=PostgresConfig.HOST, port=PostgresConfig.PORT,
        user=PostgresConfig.USER, password=PostgresConfig.PASSWORD,
        dbname=PostgresConfig.DB,
    )
    cursor = conn.cursor()

    features = {}

    # Streaming features
    cursor.execute("SELECT * FROM streaming_features WHERE customer_id = %s", (customer_id,))
    row = cursor.fetchone()
    if row:
        cols = [desc[0] for desc in cursor.description]
        for col, val in zip(cols, row):
            if col != "customer_id" and col != "updated_at" and val is not None:
                features[col] = float(val)

    # Batch features
    cursor.execute("SELECT * FROM batch_features WHERE customer_id = %s", (customer_id,))
    row = cursor.fetchone()
    if row:
        cols = [desc[0] for desc in cursor.description]
        for col, val in zip(cols, row):
            if col != "customer_id" and col != "updated_at" and val is not None:
                if isinstance(val, bool):
                    features[col] = 1.0 if val else 0.0
                else:
                    try:
                        features[col] = float(val)
                    except (ValueError, TypeError):
                        pass

    cursor.close()
    conn.close()
    return features


def assemble_feature_vector(customer_id: str) -> np.ndarray:
    """Assemble ordered feature vector for model inference."""
    # Try Redis first (faster), fall back to PostgreSQL
    features = get_features_from_redis(customer_id)
    db_features = get_features_from_db(customer_id)

    # Merge: Redis takes priority for streaming, DB for batch
    merged = {**db_features, **features}

    # Build ordered vector matching model's expected feature order
    vector = []
    for col in ModelConfig.FEATURE_COLUMNS:
        vector.append(merged.get(col, 0.0))

    return np.array(vector, dtype=np.float32)


# ─────────────────────────────────────────────
# Score Storage
# ─────────────────────────────────────────────
def store_risk_score(score_data: dict):
    """Store risk score in both Redis and PostgreSQL."""
    customer_id = score_data["customer_id"]

    # Redis (for fast access)
    redis_client.hset(f"risk_score:{customer_id}", mapping={
        "risk_score": str(score_data["risk_score"]),
        "risk_tier": score_data["risk_tier"],
        "credit_score": str(score_data["credit_score_mapped"]),
        "scored_at": score_data["scored_at"],
    })
    redis_client.expire(f"risk_score:{customer_id}", 86400)

    # PostgreSQL (for historical tracking)
    conn = psycopg2.connect(
        host=PostgresConfig.HOST, port=PostgresConfig.PORT,
        user=PostgresConfig.USER, password=PostgresConfig.PASSWORD,
        dbname=PostgresConfig.DB,
    )
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO risk_scores
        (customer_id, risk_score, risk_tier, credit_score_mapped,
         xgboost_score, lstm_score, ensemble_score, top_shap_features, model_version)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
        (
            customer_id, score_data["risk_score"], score_data["risk_tier"],
            score_data["credit_score_mapped"], score_data.get("xgboost_score"),
            score_data.get("lstm_score"), score_data["ensemble_score"],
            json.dumps(score_data.get("top_shap_features", [])),
            "v1.0",
        )
    )
    conn.commit()
    cursor.close()
    conn.close()


# ─────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────
@app.on_event("startup")
async def load_models():
    """Load models on startup."""
    global xgb_model, lstm_model, ensemble, shap_explainer, redis_client

    redis_client = redis_lib.Redis(
        host=RedisConfig.HOST, port=RedisConfig.PORT, db=RedisConfig.DB,
    )

    MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

    # Load XGBoost
    try:
        xgb_path = os.path.join(MODEL_DIR, "xgboost_model.joblib")
        if os.path.exists(xgb_path):
            xgb_model = XGBoostDelinquencyModel()
            xgb_model.load(xgb_path)
            logger.info("XGBoost model loaded")
    except Exception as e:
        logger.warning(f"XGBoost load failed (service will start without it): {e}")

    # Initialize SHAP
    if xgb_model is not None:
        try:
            shap_explainer = SHAPExplainer(xgb_model.get_booster(), xgb_model.feature_names)
            logger.info("SHAP explainer initialized")
        except Exception as e:
            logger.warning(f"SHAP initialization failed (non-fatal): {e}")

    # Load LSTM
    try:
        lstm_path = os.path.join(MODEL_DIR, "lstm_model.pt")
        if os.path.exists(lstm_path):
            lstm_model = LSTMDelinquencyModel()
            lstm_model.load(lstm_path)
            logger.info("LSTM model loaded")
    except Exception as e:
        logger.warning(f"LSTM load failed (non-fatal): {e}")

    # Initialize ensemble
    ensemble = EnsembleScorer()
    logger.info("Scoring service ready")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        models_loaded={
            "xgboost": xgb_model is not None,
            "lstm": lstm_model is not None,
            "shap": shap_explainer is not None,
        },
        timestamp=datetime.now().isoformat(),
    )


@app.post("/score", response_model=ScoreResponse)
async def score_customer(request: ScoreRequest):
    """Score a single customer."""
    if xgb_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    customer_id = request.customer_id

    # Step 1: Assemble feature vector
    features = assemble_feature_vector(customer_id)
    features_2d = features.reshape(1, -1)

    # Step 2: XGBoost inference
    xgb_prob = float(xgb_model.predict_proba(features_2d)[0])

    # Step 3: LSTM inference (if available)
    lstm_prob = None
    # LSTM requires temporal sequences - skip for now in single-score mode

    # Step 4: Ensemble
    ensemble_score = ensemble.combine(xgb_prob, lstm_prob)
    risk_tier = ensemble.score_to_risk_tier(ensemble_score)
    credit_score = ensemble.score_to_credit_score(ensemble_score)

    # Step 5: SHAP explanation
    top_shap = None
    explanation = None
    if shap_explainer:
        try:
            shap_result = shap_explainer.explain_single(features_2d)
            top_shap = shap_result["top_drivers"]
            explanation = shap_result["explanation"]
        except Exception as e:
            logger.warning(f"SHAP failed for {customer_id}: {e}")

    scored_at = datetime.now().isoformat()

    # Step 6: Store results
    score_data = {
        "customer_id": customer_id,
        "risk_score": ensemble_score,
        "risk_tier": risk_tier,
        "credit_score_mapped": credit_score,
        "xgboost_score": xgb_prob,
        "lstm_score": lstm_prob,
        "ensemble_score": ensemble_score,
        "top_shap_features": top_shap,
        "scored_at": scored_at,
    }
    try:
        store_risk_score(score_data)
    except Exception as e:
        logger.warning(f"Score storage failed: {e}")

    return ScoreResponse(
        customer_id=customer_id,
        risk_score=ensemble_score,
        risk_tier=risk_tier,
        credit_score_mapped=credit_score,
        xgboost_score=xgb_prob,
        lstm_score=lstm_prob,
        ensemble_score=ensemble_score,
        top_shap_features=top_shap,
        explanation=explanation,
        scored_at=scored_at,
    )


@app.post("/score/batch")
async def score_batch(request: BatchScoreRequest):
    """Score multiple customers."""
    results = []
    for customer_id in request.customer_ids:
        try:
            req = ScoreRequest(customer_id=customer_id)
            result = await score_customer(req)
            results.append(result.dict())
        except Exception as e:
            results.append({"customer_id": customer_id, "error": str(e)})
    return {"results": results}


@app.get("/score/{customer_id}")
async def get_score(customer_id: str):
    """Get latest stored risk score for a customer."""
    conn = psycopg2.connect(
        host=PostgresConfig.HOST, port=PostgresConfig.PORT,
        user=PostgresConfig.USER, password=PostgresConfig.PASSWORD,
        dbname=PostgresConfig.DB,
    )
    cursor = conn.cursor()
    cursor.execute(
        """SELECT risk_score, risk_tier, credit_score_mapped, xgboost_score,
                  lstm_score, ensemble_score, top_shap_features, scored_at
           FROM risk_scores WHERE customer_id = %s
           ORDER BY scored_at DESC LIMIT 1""",
        (customer_id,)
    )
    row = cursor.fetchone()
    cursor.close()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="No score found for customer")

    return {
        "customer_id": customer_id,
        "risk_score": float(row[0]),
        "risk_tier": row[1],
        "credit_score_mapped": row[2],
        "xgboost_score": float(row[3]) if row[3] else None,
        "lstm_score": float(row[4]) if row[4] else None,
        "ensemble_score": float(row[5]) if row[5] else None,
        "top_shap_features": row[6],
        "scored_at": row[7].isoformat() if row[7] else None,
    }


if __name__ == "__main__":
    uvicorn.run(app, host=ScoringConfig.HOST, port=ScoringConfig.PORT)
