"""
Celery Application Configuration
Async task processing for intervention dispatch.
"""
import os
import sys
from celery import Celery

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config.settings import CeleryConfig

celery_app = Celery(
    "pdi_intervention",
    broker=CeleryConfig.BROKER_URL,
    backend=CeleryConfig.RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Kolkata",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_routes={
        "intervention.tasks.dispatch_intervention": {"queue": "interventions"},
        "intervention.tasks.process_scoring_batch": {"queue": "scoring"},
    },
)
