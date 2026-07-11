"""
intelligence/rag_retrieval.py
==============================
Qdrant-based RAG (Retrieval-Augmented Generation) for historical event analogs.
Uses Qdrant in-memory or local mode — 100% free, no cloud account needed.

For each new cluster, retrieves the most similar past events and their
realized outcomes (direction + return) to provide context for signal generation.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import logging
from typing import Optional
from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
)

from config import settings

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
_qdrant_client: Optional[QdrantClient] = None

def get_qdrant_client() -> QdrantClient:
    """Singleton Qdrant client. In-memory by default (free, no setup)."""
    global _qdrant_client
    if _qdrant_client is None:
        if settings.QDRANT_MODE == "memory":
            _qdrant_client = QdrantClient(":memory:")
            logger.info("[RAG] Qdrant started in-memory mode.")
        else:
            os.makedirs(settings.QDRANT_PATH, exist_ok=True)
            _qdrant_client = QdrantClient(path=settings.QDRANT_PATH)
            logger.info(f"[RAG] Qdrant started in local disk mode: {settings.QDRANT_PATH}")

        # Create collection if it doesn't exist
        existing = [c.name for c in _qdrant_client.get_collections().collections]
        if settings.QDRANT_COLLECTION not in existing:
            _qdrant_client.create_collection(
                collection_name=settings.QDRANT_COLLECTION,
                vectors_config=VectorParams(
                    size=settings.QDRANT_VECTOR_SIZE,   # 384 (MiniLM)
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"[RAG] Created Qdrant collection: {settings.QDRANT_COLLECTION}")

    return _qdrant_client


# ─────────────────────────────────────────────────────────────────────────────
def store_event(
    event_id: int,
    embedding: list[float],
    headline: str,
    summary: str,
    ticker: str,
    direction: str,
    return_pct: float,
    event_date: Optional[str] = None,
) -> bool:
    """
    Store a historical event in Qdrant for future RAG retrieval.

    Parameters
    ----------
    event_id    : Unique integer ID (matches DB primary key)
    embedding   : 384-dim MiniLM vector
    headline    : Article headline
    summary     : BART-generated summary
    ticker      : Associated ticker
    direction   : Realized direction ('up'/'down'/'flat')
    return_pct  : Realized forward return (%)
    """
    client = get_qdrant_client()
    try:
        client.upsert(
            collection_name=settings.QDRANT_COLLECTION,
            points=[
                PointStruct(
                    id=event_id,
                    vector=embedding,
                    payload={
                        "headline":   headline[:300],
                        "summary":    summary[:500] if summary else "",
                        "ticker":     ticker,
                        "direction":  direction,
                        "return_pct": return_pct,
                        "event_date": event_date or str(datetime.utcnow().date()),
                    },
                )
            ],
        )
        return True
    except Exception as e:
        logger.error(f"[RAG] Store failed: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
def retrieve_analogs(
    query_embedding: list[float],
    top_k: int = 5,
    ticker_filter: Optional[str] = None,
) -> list[dict]:
    """
    Retrieve top-k most similar historical events.

    Parameters
    ----------
    query_embedding : 384-dim vector of the current event
    top_k           : Number of analogs to return
    ticker_filter   : Optional — filter by ticker

    Returns list of dicts with analog details + similarity score.
    """
    client = get_qdrant_client()

    # Check if collection has any points
    try:
        count = client.count(collection_name=settings.QDRANT_COLLECTION).count
        if count == 0:
            logger.info("[RAG] No historical events stored yet — skipping retrieval.")
            return []
    except Exception:
        return []

    try:
        search_filter = None
        if ticker_filter:
            search_filter = Filter(
                must=[FieldCondition(key="ticker", match=MatchValue(value=ticker_filter))]
            )

        results = client.search(
            collection_name=settings.QDRANT_COLLECTION,
            query_vector=query_embedding,
            query_filter=search_filter,
            limit=top_k,
            with_payload=True,
        )

        analogs = []
        for hit in results:
            analogs.append({
                "similarity":  round(float(hit.score), 4),
                "headline":    hit.payload.get("headline", ""),
                "summary":     hit.payload.get("summary", ""),
                "ticker":      hit.payload.get("ticker", ""),
                "direction":   hit.payload.get("direction", "flat"),
                "return_pct":  hit.payload.get("return_pct", 0.0),
                "event_date":  hit.payload.get("event_date", ""),
                "event_id":    hit.id,
            })

        logger.info(f"[RAG] Retrieved {len(analogs)} analogs (top_k={top_k})")
        return analogs

    except Exception as e:
        logger.error(f"[RAG] Retrieval failed: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
def analog_signal_prior(analogs: list[dict]) -> dict:
    """
    Derive a signal prior from retrieved analogs.
    Weights analogs by their cosine similarity score.

    Returns:
    {
        "direction":   "up"|"down"|"flat",
        "confidence":  float,
        "mean_return": float,
        "n_analogs":   int,
    }
    """
    if not analogs:
        return {"direction": "flat", "confidence": 0.0, "mean_return": 0.0, "n_analogs": 0}

    weights = np.array([a["similarity"] for a in analogs])
    weights = weights / weights.sum()   # normalize

    # Weighted return
    returns = np.array([a["return_pct"] for a in analogs])
    mean_return = float(np.dot(weights, returns))

    # Direction vote (weighted)
    direction_scores = {"up": 0.0, "down": 0.0, "flat": 0.0}
    for a, w in zip(analogs, weights):
        direction_scores[a.get("direction", "flat")] += float(w)

    best_direction = max(direction_scores, key=direction_scores.get)
    confidence = direction_scores[best_direction]

    return {
        "direction":   best_direction,
        "confidence":  round(confidence, 4),
        "mean_return": round(mean_return, 4),
        "n_analogs":   len(analogs),
        "top_analog":  analogs[0] if analogs else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
def seed_with_sample_events():
    """
    Seed Qdrant with a small set of synthetic historical events.
    Used for demo / Phase 0 testing without real historical data.
    """
    import random
    random.seed(42)
    np.random.seed(42)

    sample_events = [
        {"headline": "Apple reports record quarterly profit, beats estimates by 20%",
         "ticker": "AAPL", "direction": "up",   "return_pct": 4.5},
        {"headline": "Fed raises interest rates by 50bps, signals more hikes ahead",
         "ticker": "SPY",  "direction": "down",  "return_pct": -2.1},
        {"headline": "Tesla recalls 1.2M vehicles over autopilot safety concerns",
         "ticker": "TSLA", "direction": "down",  "return_pct": -6.3},
        {"headline": "Microsoft Azure cloud revenue grows 29% year over year",
         "ticker": "MSFT", "direction": "up",   "return_pct": 3.2},
        {"headline": "Bank of America beats earnings, raises dividend",
         "ticker": "BAC",  "direction": "up",   "return_pct": 2.8},
        {"headline": "NVIDIA announces next-gen GPU with 3x performance improvement",
         "ticker": "NVDA", "direction": "up",   "return_pct": 8.1},
        {"headline": "Amazon announces major layoffs, cuts 18000 jobs",
         "ticker": "AMZN", "direction": "down",  "return_pct": -1.5},
        {"headline": "Google parent Alphabet misses revenue estimates for first time",
         "ticker": "GOOGL","direction": "down",  "return_pct": -4.7},
        {"headline": "JPMorgan reports solid quarterly results amid banking turmoil",
         "ticker": "JPM",  "direction": "flat",  "return_pct": 0.3},
        {"headline": "S&P 500 enters bear market territory on recession fears",
         "ticker": "SPY",  "direction": "down",  "return_pct": -3.4},
    ]

    logger.info("[RAG] Seeding Qdrant with sample historical events...")
    for i, event in enumerate(sample_events):
        # Random 384-dim embedding (placeholder — real system uses MiniLM)
        embedding = np.random.randn(384).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)   # normalize
        store_event(
            event_id=i + 1,
            embedding=embedding.tolist(),
            headline=event["headline"],
            summary=event["headline"],
            ticker=event["ticker"],
            direction=event["direction"],
            return_pct=event["return_pct"],
            event_date="2024-01-01",
        )
    logger.info(f"[RAG] Seeded {len(sample_events)} historical events.")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== Qdrant RAG Test ===\n")

    # Seed with sample events
    seed_with_sample_events()

    # Query with a random embedding
    np.random.seed(99)
    query = np.random.randn(384).astype(np.float32)
    query = query / np.linalg.norm(query)

    analogs = retrieve_analogs(query.tolist(), top_k=3)
    print(f"\nTop {len(analogs)} analogs:")
    for a in analogs:
        print(f"  [{a['ticker']}] {a['headline'][:60]}")
        print(f"    Similarity={a['similarity']:.3f}  Direction={a['direction']}  Return={a['return_pct']:+.1f}%")

    prior = analog_signal_prior(analogs)
    print(f"\nAnalog prior: {prior}")
