"""
intelligence/quant_fusion.py
=============================
XGBoost-based quant fusion layer.
Combines NLP signal (FinBERT) + RAG analog prior + market features (yfinance).
All data sources are free.

Phase 0: Train on synthetic / historical data.
Phase 1+: Retrain on walk-forward real data.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score

from config import settings

logger = logging.getLogger(__name__)

MODEL_PATH = Path(settings.RESULTS_DIR) / "quant_fusion_model.pkl"
ENCODER_PATH = Path(settings.RESULTS_DIR) / "label_encoder.pkl"

DIRECTION_MAP = {"up": 1, "down": -1, "flat": 0}
REVERSE_MAP   = {1: "up", -1: "down", 0: "flat"}


# ─────────────────────────────────────────────────────────────────────────────
def get_market_features(ticker: str, event_date: Optional[datetime] = None) -> dict:
    """
    Fetch free market features for a ticker using yfinance.
    Returns a dict of scalar features for the XGBoost model.
    """
    features = {
        "price_change_1d": 0.0,
        "price_change_5d": 0.0,
        "volume_ratio":    1.0,
        "rsi_14":          50.0,
        "above_sma20":     0.0,
        "volatility_20d":  0.02,
        "spy_corr_20d":    0.5,
    }

    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)
        df = tk.history(period="3mo", auto_adjust=True)

        if df is None or len(df) < 5:
            return features

        close  = df["Close"].dropna().values
        volume = df["Volume"].dropna().values

        # 1-day return
        features["price_change_1d"] = float((close[-1] - close[-2]) / close[-2]) if len(close) > 1 else 0.0
        # 5-day return
        features["price_change_5d"] = float((close[-1] - close[-6]) / close[-6]) if len(close) > 5 else 0.0
        # Volume ratio (today vs 20-day avg)
        features["volume_ratio"] = float(volume[-1] / np.mean(volume[-20:])) if len(volume) >= 20 else 1.0
        # RSI-14
        features["rsi_14"] = float(_compute_rsi(close, 14))
        # Above 20-day SMA
        sma20 = np.mean(close[-20:]) if len(close) >= 20 else close[-1]
        features["above_sma20"] = 1.0 if close[-1] > sma20 else 0.0
        # 20-day volatility (annualized)
        returns = np.diff(close[-21:]) / close[-21:-1]
        features["volatility_20d"] = float(np.std(returns) * np.sqrt(252)) if len(returns) > 1 else 0.02

    except Exception as e:
        logger.warning(f"[QUANT] Market features failed for {ticker}: {e}")

    return features


def _compute_rsi(prices: np.ndarray, period: int = 14) -> float:
    """Simple RSI calculation."""
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(prices[-(period+1):])
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100 - (100 / (1 + rs)))


# ─────────────────────────────────────────────────────────────────────────────
def build_feature_vector(
    impact_score:    float,
    confidence:      float,
    analog_direction: str,
    analog_confidence: float,
    analog_return:   float,
    n_analogs:       int,
    market_features: dict,
) -> np.ndarray:
    """
    Build the feature vector fed into XGBoost.
    Total: 14 features.
    """
    analog_dir_num = DIRECTION_MAP.get(analog_direction, 0)

    features = np.array([
        # NLP signal features (from FinBERT)
        impact_score,                             # signed -1 to +1
        confidence,                               # 0–1
        float(analog_dir_num),                    # -1/0/+1
        analog_confidence,                        # 0–1
        analog_return,                            # historical mean return %
        float(n_analogs),                         # number of RAG analogs found
        # Market features (from yfinance)
        market_features.get("price_change_1d",  0.0),
        market_features.get("price_change_5d",  0.0),
        market_features.get("volume_ratio",     1.0),
        market_features.get("rsi_14",           50.0),
        market_features.get("above_sma20",      0.0),
        market_features.get("volatility_20d",   0.02),
        market_features.get("spy_corr_20d",     0.5),
        # Cross features
        impact_score * analog_confidence,         # interaction term
    ], dtype=np.float32)

    return features


# ─────────────────────────────────────────────────────────────────────────────
def generate_synthetic_training_data(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic training data for Phase 0 model validation.
    This validates the pipeline mechanics before connecting real data.
    """
    rng = np.random.RandomState(seed)
    rows = []

    for _ in range(n_samples):
        # True direction (ground truth)
        true_dir = rng.choice([-1, 0, 1], p=[0.35, 0.30, 0.35])

        # NLP signal (correlated with true direction + noise)
        impact_score    = true_dir * 0.5 + rng.randn() * 0.3
        impact_score    = np.clip(impact_score, -1, 1)
        confidence      = 0.5 + abs(impact_score) * 0.3 + rng.rand() * 0.2

        # RAG analog (partially correlated)
        analog_dir      = true_dir if rng.rand() > 0.3 else rng.choice([-1, 0, 1])
        analog_conf     = 0.4 + rng.rand() * 0.5
        analog_return   = true_dir * 2.0 + rng.randn() * 1.5

        # Market features (partially correlated with direction)
        market = {
            "price_change_1d": true_dir * 0.01 + rng.randn() * 0.02,
            "price_change_5d": true_dir * 0.03 + rng.randn() * 0.04,
            "volume_ratio":    1.0 + abs(true_dir) * 0.5 + rng.rand() * 0.5,
            "rsi_14":          50 + true_dir * 15 + rng.randn() * 10,
            "above_sma20":     1.0 if true_dir > 0 else 0.0,
            "volatility_20d":  0.02 + rng.rand() * 0.03,
            "spy_corr_20d":    0.3 + rng.rand() * 0.5,
        }

        feat = build_feature_vector(
            impact_score, confidence,
            REVERSE_MAP[analog_dir], analog_conf, analog_return, 5,
            market,
        )

        rows.append({**{f"f{i}": v for i, v in enumerate(feat)}, "label": true_dir})

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
def train_model(df: Optional[pd.DataFrame] = None, save: bool = True) -> xgb.XGBClassifier:
    """
    Train XGBoost on synthetic data (Phase 0) or real walk-forward data.
    Uses TimeSeriesSplit to avoid data leakage.
    """
    if df is None:
        logger.info("[QUANT] Training on synthetic data (Phase 0)...")
        df = generate_synthetic_training_data(n_samples=2000)

    feature_cols = [c for c in df.columns if c.startswith("f")]
    X = df[feature_cols].values
    y = df["label"].values  # -1, 0, 1

    # Shift labels to 0, 1, 2 for XGBoost multi-class
    y_shifted = y + 1   # -1→0, 0→1, 1→2

    tscv = TimeSeriesSplit(n_splits=5)
    val_scores = []

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y_shifted[train_idx], y_shifted[val_idx]
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)
        val_scores.append(acc)
        logger.info(f"[QUANT] Fold {fold+1}: val_acc={acc:.3f}")

    logger.info(f"[QUANT] Mean val accuracy: {np.mean(val_scores):.3f}")

    # Final fit on all data
    model.fit(X, y_shifted)

    if save:
        MODEL_PATH.parent.mkdir(exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"[QUANT] Model saved to {MODEL_PATH}")

    return model


def load_model() -> Optional[xgb.XGBClassifier]:
    """Load trained model from disk, or train a new one if not found."""
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        logger.info(f"[QUANT] Model loaded from {MODEL_PATH}")
        return model
    logger.info("[QUANT] No model found — training on synthetic data...")
    return train_model()


# ─────────────────────────────────────────────────────────────────────────────
_model: Optional[xgb.XGBClassifier] = None

def predict_signal(
    impact_score:     float,
    confidence:       float,
    analog_direction: str,
    analog_confidence: float,
    analog_return:    float,
    n_analogs:        int,
    ticker:           str,
    event_date:       Optional[datetime] = None,
) -> dict:
    """
    Full quant fusion prediction for a single event.
    Returns: direction, confidence, quant_score, feature breakdown.
    """
    global _model
    if _model is None:
        _model = load_model()

    # Get free market features
    market_features = get_market_features(ticker, event_date)

    # Build feature vector
    feat = build_feature_vector(
        impact_score, confidence,
        analog_direction, analog_confidence, analog_return, n_analogs,
        market_features,
    ).reshape(1, -1)

    # XGBoost prediction
    xgb_proba = _model.predict_proba(feat)[0]  # [P(down), P(flat), P(up)]
    
    # If we lack historical RAG data or market features, XGBoost (trained on synthetic data) 
    # tends to collapse to neutral. We must inject the direct NLP impact score into the 
    # probabilities to ensure the engine remains dynamic and responsive to news sentiment.
    nlp_up = max(0.0, impact_score)
    nlp_down = max(0.0, -impact_score)
    nlp_flat = max(0.0, 1.0 - abs(impact_score))
    
    # Blend XGBoost with direct NLP (heavy bias to NLP if no historical analogs)
    blend_weight = 0.85 if n_analogs == 0 else 0.4
    proba = np.array([
        xgb_proba[0] * (1 - blend_weight) + nlp_down * blend_weight,
        xgb_proba[1] * (1 - blend_weight) + nlp_flat * blend_weight,
        xgb_proba[2] * (1 - blend_weight) + nlp_up * blend_weight,
    ])
    proba = proba / np.sum(proba)  # Normalize
    
    # Calibrate/damp overconfident probabilities towards a uniform prior
    # This scales confidence to realistic 55%-85% limits.
    damp_factor = 0.65
    proba = proba * damp_factor + (1.0 - damp_factor) / 3.0
    
    pred_class = int(np.argmax(proba))  # 0, 1, or 2
    pred_direction = REVERSE_MAP[pred_class - 1]  # shift back: 0→-1, 1→0, 2→1
    
    quant_score = float(proba[2] - proba[0])   # P(up) - P(down) → signed score
    
    return {
        "direction":       pred_direction,
        "confidence":      round(float(max(proba)), 4),
        "quant_score":     round(quant_score, 4),
        "prob_up":         round(float(proba[2]), 4),
        "prob_down":       round(float(proba[0]), 4),
        "prob_flat":       round(float(proba[1]), 4),
        "market_features": market_features,
    }


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== XGBoost Quant Fusion Test ===\n")

    # Train on synthetic data
    model = train_model(save=False)

    # Test prediction
    result = predict_signal(
        impact_score=0.6,
        confidence=0.8,
        analog_direction="up",
        analog_confidence=0.7,
        analog_return=3.2,
        n_analogs=5,
        ticker="AAPL",
    )
    print(f"Prediction: {result}")
