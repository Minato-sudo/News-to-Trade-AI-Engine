"""
intelligence/impact_classifier.py
==================================
FinBERT-based financial news impact classifier.
Uses ProsusAI/finbert — 100% free, open weights on HuggingFace.

Classifies each headline/text into:
  - direction: 'up' | 'down' | 'flat'  (maps from positive/negative/neutral)
  - confidence: 0.0–1.0
  - impact_score: signed score (-1.0 to +1.0)
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import logging
from typing import Optional
from functools import lru_cache

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

from config import settings

logger = logging.getLogger(__name__)

# FinBERT label → trading direction mapping
FINBERT_TO_DIRECTION = {
    "positive": "up",
    "negative": "down",
    "neutral":  "flat",
}

# ─────────────────────────────────────────────────────────────────────────────
_classifier_instance = None

def get_classifier():
    """
    Singleton: load FinBERT once, reuse across all requests.
    Uses CUDA if available (RTX 5050), else CPU.
    """
    global _classifier_instance
    if _classifier_instance is None:
        device = 0 if torch.cuda.is_available() else -1
        device_name = "GPU (CUDA)" if device == 0 else "CPU"
        logger.info(f"[IMPACT] Loading FinBERT on {device_name}...")

        _classifier_instance = pipeline(
            "text-classification",
            model=settings.IMPACT_MODEL,     # ProsusAI/finbert
            device=device,
            top_k=None,                       # return all label scores
            truncation=True,
            max_length=512,
        )
        logger.info("[IMPACT] FinBERT loaded.")
    return _classifier_instance


# ─────────────────────────────────────────────────────────────────────────────
def classify_headline(text: str) -> dict:
    """
    Classify a single headline/text.

    Returns:
    {
        "direction":      "up" | "down" | "flat",
        "confidence":     float (0–1),
        "impact_score":   float (-1 to +1),  # +1 = strongly up, -1 = strongly down
        "raw_scores":     {"positive": float, "negative": float, "neutral": float}
    }
    """
    if not text or len(text.strip()) < 5:
        return {
            "direction": "flat", "confidence": 0.0,
            "impact_score": 0.0, "raw_scores": {}
        }

    classifier = get_classifier()

    try:
        results = classifier(text[:512])[0]   # top_k=None returns list of all labels
        raw_scores = {r["label"].lower(): r["score"] for r in results}

        pos = raw_scores.get("positive", 0.0)
        neg = raw_scores.get("negative", 0.0)
        neu = raw_scores.get("neutral",  0.0)

        # Best label
        best_label = max(raw_scores, key=raw_scores.get)
        direction  = FINBERT_TO_DIRECTION.get(best_label, "flat")
        confidence = raw_scores[best_label]

        # Signed impact score: positive pulls toward +1, negative toward -1
        impact_score = pos - neg   # range: -1 to +1

        return {
            "direction":    direction,
            "confidence":   round(float(confidence), 4),
            "impact_score": round(float(impact_score), 4),
            "raw_scores":   {k: round(float(v), 4) for k, v in raw_scores.items()},
        }

    except Exception as e:
        logger.warning(f"[IMPACT] Classification failed: {e}")
        return {
            "direction": "flat", "confidence": 0.0,
            "impact_score": 0.0, "raw_scores": {}
        }


# ─────────────────────────────────────────────────────────────────────────────
def classify_batch(texts: list[str], batch_size: int = 32) -> list[dict]:
    """
    Classify a batch of headlines efficiently.
    """
    results = []
    classifier = get_classifier()

    for i in range(0, len(texts), batch_size):
        batch = [t[:512] for t in texts[i:i+batch_size] if t and len(t.strip()) > 5]
        if not batch:
            results.extend([{"direction": "flat", "confidence": 0.0, "impact_score": 0.0, "raw_scores": {}}] * len(texts[i:i+batch_size]))
            continue

        try:
            batch_results = classifier(batch)
            for item_results in batch_results:
                raw_scores = {r["label"].lower(): r["score"] for r in item_results}
                pos = raw_scores.get("positive", 0.0)
                neg = raw_scores.get("negative", 0.0)
                best_label = max(raw_scores, key=raw_scores.get)
                results.append({
                    "direction":    FINBERT_TO_DIRECTION.get(best_label, "flat"),
                    "confidence":   round(float(raw_scores[best_label]), 4),
                    "impact_score": round(float(pos - neg), 4),
                    "raw_scores":   {k: round(float(v), 4) for k, v in raw_scores.items()},
                })
        except Exception as e:
            logger.warning(f"[IMPACT] Batch failed: {e}")
            results.extend([{"direction": "flat", "confidence": 0.0, "impact_score": 0.0, "raw_scores": {}}] * len(batch))

    return results


# ─────────────────────────────────────────────────────────────────────────────
def score_cluster(
    headlines: list[str],
    aggregate: str = "mean",
) -> dict:
    """
    Aggregate impact scores across all headlines in a cluster.
    aggregate: 'mean' | 'max_abs' | 'weighted' (weighted by confidence)
    """
    if not headlines:
        return {"direction": "flat", "confidence": 0.0, "impact_score": 0.0}

    scores = classify_batch(headlines)

    impact_scores = [s["impact_score"] for s in scores]
    confidences   = [s["confidence"]   for s in scores]

    if aggregate == "weighted":
        total_conf = sum(confidences) or 1.0
        agg_impact = sum(i * c for i, c in zip(impact_scores, confidences)) / total_conf
        agg_conf   = sum(confidences) / len(confidences)
    elif aggregate == "max_abs":
        idx = int(np.argmax(np.abs(impact_scores)))
        agg_impact = impact_scores[idx]
        agg_conf   = confidences[idx]
    else:  # mean
        agg_impact = float(np.mean(impact_scores))
        agg_conf   = float(np.mean(confidences))

    # Derive direction from aggregated score
    if agg_impact > settings.SIGNAL_DIRECTION_THRESHOLD - 0.5:
        direction = "up"
    elif agg_impact < -(settings.SIGNAL_DIRECTION_THRESHOLD - 0.5):
        direction = "down"
    else:
        direction = "flat"

    return {
        "direction":    direction,
        "confidence":   round(agg_conf, 4),
        "impact_score": round(agg_impact, 4),
        "n_headlines":  len(headlines),
    }


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== FinBERT Impact Classifier Test ===\n")

    test_headlines = [
        "Apple reports record quarterly earnings, beats Wall Street estimates",
        "Tesla stock crashes after CEO tweets controversy",
        "Federal Reserve holds interest rates steady",
        "Microsoft acquires gaming company for $68 billion",
        "Recession fears grow as manufacturing index drops",
    ]

    for h in test_headlines:
        result = classify_headline(h)
        print(f"Headline: {h[:60]}")
        print(f"  Direction: {result['direction']}  "
              f"Confidence: {result['confidence']:.2f}  "
              f"Impact: {result['impact_score']:+.2f}")
        print()
