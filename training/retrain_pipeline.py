"""
training/retrain_pipeline.py
=============================
Phase 1 Real-Data Retraining Pipeline.

Collects all labeled signals from the database, reconstructs their feature
vectors using the stored FinBERT + quant data, and retrains the XGBoost
model — replacing the Phase 0 synthetic model with one trained on actual
market outcomes.

Minimum 30 labeled samples are required before retraining is attempted.
After successful retraining, the live `_model` singleton in quant_fusion.py
is hot-swapped so future predictions use the new model without a restart.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report

from config import settings

logger = logging.getLogger(__name__)

MODEL_PATH   = Path(settings.RESULTS_DIR) / "quant_fusion_model.pkl"
METRICS_PATH = Path(settings.RESULTS_DIR) / "model_metrics.json"

DIRECTION_MAP = {"up": 2, "flat": 1, "down": 0}   # shifted for XGBoost multiclass
REVERSE_MAP   = {2: "up", 1: "flat", 0: "down"}

MIN_SAMPLES = 30   # won't retrain below this threshold


async def build_training_dataset(db) -> Optional[pd.DataFrame]:
    """
    Pulls all labeled signals from the DB and reconstructs feature vectors.

    Returns a DataFrame with columns f0..f13 + label, or None if insufficient data.
    """
    from sqlalchemy import select, and_
    from db.models import Signal

    result = await db.execute(
        select(Signal)
        .where(
            and_(
                Signal.actual_direction.isnot(None),
                Signal.impact_score.isnot(None),
                Signal.confidence.isnot(None),
            )
        )
        .order_by(Signal.generated_at.asc())
    )
    signals = result.scalars().all()

    if len(signals) < MIN_SAMPLES:
        logger.info(f"[RETRAIN] Only {len(signals)} labeled samples — need {MIN_SAMPLES}. Skipping.")
        return None

    from intelligence.quant_fusion import (
        build_feature_vector, DIRECTION_MAP as DM, get_market_features
    )

    rows = []
    for sig in signals:
        try:
            # Reconstruct market features at signal time (use defaults if unavailable)
            market = get_market_features(sig.ticker, sig.generated_at)

            # Analog features stored in rag_analog JSON field
            analog = sig.rag_analog or {}
            analog_dir  = analog.get("direction", "flat")
            analog_conf = float(analog.get("confidence", 0.5))
            analog_ret  = float(analog.get("return_pct", 0.0))
            n_analogs   = int(analog.get("n_analogs", 0))

            feat = build_feature_vector(
                impact_score      = float(sig.impact_score),
                confidence        = float(sig.confidence),
                analog_direction  = analog_dir,
                analog_confidence = analog_conf,
                analog_return     = analog_ret,
                n_analogs         = n_analogs,
                market_features   = market,
            )

            label = DIRECTION_MAP.get(sig.actual_direction, 1)
            row   = {f"f{i}": float(v) for i, v in enumerate(feat)}
            row["label"] = label
            row["generated_at"] = sig.generated_at
            rows.append(row)

        except Exception as e:
            logger.debug(f"[RETRAIN] Skipping signal {sig.id}: {e}")
            continue

    if len(rows) < MIN_SAMPLES:
        logger.warning(f"[RETRAIN] Only {len(rows)} valid feature rows after processing. Need {MIN_SAMPLES}.")
        return None

    df = pd.DataFrame(rows).sort_values("generated_at").reset_index(drop=True)
    logger.info(f"[RETRAIN] Dataset ready: {len(df)} samples  "
                f"up={sum(df.label==2)} flat={sum(df.label==1)} down={sum(df.label==0)}")
    return df


def train_on_real_data(df: pd.DataFrame) -> tuple:
    """
    Train XGBoost on real labeled data with TimeSeriesSplit cross-validation.
    Returns (model, metrics_dict).
    """
    feature_cols = [c for c in df.columns if c.startswith("f")]
    X = df[feature_cols].values
    y = df["label"].values  # 0 (down), 1 (flat), 2 (up)

    tscv = TimeSeriesSplit(n_splits=min(5, max(2, len(df) // 10)))
    val_accuracies = []
    val_f1s        = []

    model = xgb.XGBClassifier(
        n_estimators      = 300,
        max_depth         = 5,
        learning_rate     = 0.03,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        min_child_weight  = 3,
        gamma             = 0.1,
        use_label_encoder = False,
        eval_metric       = "mlogloss",
        random_state      = 42,
        n_jobs            = -1,
    )

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)
        f1  = f1_score(y_val, preds, average="macro", zero_division=0)
        val_accuracies.append(acc)
        val_f1s.append(f1)
        logger.info(f"[RETRAIN] Fold {fold+1}: acc={acc:.3f}  macro-f1={f1:.3f}")

    # Final fit on all data
    model.fit(X, y)

    metrics = {
        "phase":          "Phase 1 — Real Data",
        "samples":        int(len(df)),
        "val_accuracy":   round(float(np.mean(val_accuracies)), 4),
        "val_f1_macro":   round(float(np.mean(val_f1s)), 4),
        "trained_at":     datetime.utcnow().isoformat(),
        "distribution":   {
            "up":   int(sum(y == 2)),
            "flat": int(sum(y == 1)),
            "down": int(sum(y == 0)),
        },
    }

    logger.info(f"[RETRAIN] ✅ Real-data model trained — "
                f"val_acc={metrics['val_accuracy']:.3f}  f1={metrics['val_f1_macro']:.3f}")
    return model, metrics


def save_model_and_metrics(model, metrics: dict):
    """Persist model + metrics JSON to disk."""
    import json
    MODEL_PATH.parent.mkdir(exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"[RETRAIN] Model saved → {MODEL_PATH}")


def hot_swap_model(new_model):
    """Replace the in-memory XGBoost singleton without restarting the server."""
    import intelligence.quant_fusion as qf
    qf._model = new_model
    logger.info("[RETRAIN] 🔄 Live model hot-swapped — all new predictions use Phase 1 model.")


async def run_retraining_pipeline(db) -> dict:
    """
    Full pipeline: build dataset → train → save → hot-swap.
    Returns a result dict suitable for the API response.
    """
    logger.info("[RETRAIN] Starting Phase 1 retraining pipeline...")

    df = await build_training_dataset(db)
    if df is None:
        return {
            "status":  "skipped",
            "reason":  f"Fewer than {MIN_SAMPLES} labeled samples available.",
            "samples": 0,
        }

    model, metrics = train_on_real_data(df)
    save_model_and_metrics(model, metrics)
    hot_swap_model(model)

    return {
        "status":  "success",
        "metrics": metrics,
    }


def load_metrics() -> dict:
    """Load the most recent model metrics from disk."""
    import json
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            return json.load(f)
    return {
        "phase":        "Phase 0 — Synthetic Data",
        "val_accuracy": None,
        "val_f1_macro": None,
        "trained_at":   None,
        "samples":      0,
    }


# ── CLI runner ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import asyncio
    from db.database import async_session

    async def _main():
        async with async_session() as db:
            result = await run_retraining_pipeline(db)
            print(result)

    asyncio.run(_main())
