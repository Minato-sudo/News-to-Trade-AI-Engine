"""
api/routes/admin.py
====================
Admin endpoints — model status, outcome labeling, retraining.

These endpoints expose the Phase 1 pipeline controls:
  GET  /api/admin/status      — model info + labeled sample count + accuracy
  POST /api/admin/label       — trigger outcome labeling for unlabeled signals
  POST /api/admin/retrain     — trigger full Phase 1 retraining pipeline
  GET  /api/admin/accuracy    — signal prediction accuracy over time
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from datetime import datetime
from fastapi import APIRouter, Depends, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_

from db.database import get_db
from db.models import Signal
from api.auth import get_current_user, User

router = APIRouter(prefix="/api/admin", tags=["admin"])


# ── Status endpoint ──────────────────────────────────────────────────────────
@router.get("/status")
async def get_model_status(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Returns the current model phase, accuracy metrics, and labeled sample counts.
    """
    from training.retrain_pipeline import load_metrics, MIN_SAMPLES
    from pathlib import Path
    from config import settings

    metrics = load_metrics()

    # Count labeled vs total signals (globally, not per-user)
    total_res = await db.execute(select(func.count()).select_from(Signal))
    total_signals = total_res.scalar() or 0

    labeled_res = await db.execute(
        select(func.count()).select_from(Signal)
        .where(Signal.actual_direction.isnot(None))
    )
    labeled_signals = labeled_res.scalar() or 0

    # Accuracy of predictions vs real outcomes
    correct_res = await db.execute(
        select(func.count()).select_from(Signal)
        .where(
            and_(
                Signal.actual_direction.isnot(None),
                Signal.direction == Signal.actual_direction,
            )
        )
    )
    correct = correct_res.scalar() or 0
    live_accuracy = round(correct / labeled_signals, 4) if labeled_signals > 0 else None

    # Ready for retraining?
    needs_retrain = labeled_signals >= MIN_SAMPLES

    # Model file info
    model_path = Path(settings.RESULTS_DIR) / "quant_fusion_model.pkl"
    model_file_exists = model_path.exists()
    model_size_kb = round(model_path.stat().st_size / 1024, 1) if model_file_exists else 0

    return {
        "model_phase":        metrics.get("phase", "Phase 0 — Synthetic Data"),
        "trained_at":         metrics.get("trained_at"),
        "val_accuracy":       metrics.get("val_accuracy"),
        "val_f1_macro":       metrics.get("val_f1_macro"),
        "training_samples":   metrics.get("samples", 0),
        "live_prediction_accuracy": live_accuracy,
        "total_signals":      total_signals,
        "labeled_signals":    labeled_signals,
        "unlabeled_signals":  total_signals - labeled_signals,
        "ready_for_retrain":  needs_retrain,
        "min_samples_needed": MIN_SAMPLES,
        "model_file_kb":      model_size_kb,
        "distribution":       metrics.get("distribution", {}),
    }


# ── Label outcomes ────────────────────────────────────────────────────────────
@router.post("/label")
async def trigger_labeling(
    max_signals: int = 100,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Fetch real stock prices for unlabeled signals and write actual_direction + actual_return.
    Runs synchronously (may take 1–2 min for large batches).
    """
    from training.label_outcomes import run_labeling
    result = await run_labeling(db, max_signals=max_signals)
    return {"status": "done", **result}


# ── Retrain model ─────────────────────────────────────────────────────────────
@router.post("/retrain")
async def trigger_retraining(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Run the full Phase 1 retraining pipeline:
    1. Collect labeled signals from DB
    2. Rebuild feature vectors
    3. Train XGBoost with TimeSeriesSplit cross-validation
    4. Save model to disk
    5. Hot-swap the live prediction singleton
    """
    from training.retrain_pipeline import run_retraining_pipeline
    result = await run_retraining_pipeline(db)
    return result


# ── Per-ticker accuracy breakdown ─────────────────────────────────────────────
@router.get("/accuracy")
async def get_accuracy_breakdown(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Show prediction accuracy broken down by ticker and direction.
    Only includes signals that have been labeled with real outcomes.
    """
    result = await db.execute(
        select(Signal)
        .where(Signal.actual_direction.isnot(None))
        .order_by(Signal.generated_at.desc())
        .limit(500)
    )
    signals = result.scalars().all()

    if not signals:
        return {"accuracy_by_ticker": {}, "confusion": {}, "total_labeled": 0}

    # Per-ticker accuracy
    ticker_stats: dict = {}
    for sig in signals:
        t = sig.ticker
        if t not in ticker_stats:
            ticker_stats[t] = {"correct": 0, "total": 0, "returns": []}
        ticker_stats[t]["total"] += 1
        if sig.direction == sig.actual_direction:
            ticker_stats[t]["correct"] += 1
        if sig.actual_return is not None:
            ticker_stats[t]["returns"].append(sig.actual_return)

    accuracy_by_ticker = {}
    for t, s in ticker_stats.items():
        avg_ret = round(float(sum(s["returns"]) / len(s["returns"])), 4) if s["returns"] else 0.0
        accuracy_by_ticker[t] = {
            "accuracy":    round(s["correct"] / s["total"], 3) if s["total"] else 0,
            "correct":     s["correct"],
            "total":       s["total"],
            "avg_return":  avg_ret,
        }

    # Simple 3×3 confusion matrix: predicted vs actual
    confusion = {
        "up":   {"up": 0, "flat": 0, "down": 0},
        "flat": {"up": 0, "flat": 0, "down": 0},
        "down": {"up": 0, "flat": 0, "down": 0},
    }
    for sig in signals:
        p = sig.direction or "flat"
        a = sig.actual_direction or "flat"
        if p in confusion and a in confusion[p]:
            confusion[p][a] += 1

    overall_correct = sum(1 for s in signals if s.direction == s.actual_direction)

    return {
        "total_labeled":     len(signals),
        "overall_accuracy":  round(overall_correct / len(signals), 4),
        "accuracy_by_ticker": accuracy_by_ticker,
        "confusion_matrix":  confusion,
    }
