"""
training/label_outcomes.py
===========================
Phase 1 Outcome Labeler.

For every Signal in the DB that is:
  - older than MIN_AGE_DAYS (so the outcome has had time to materialise)
  - not yet labeled (actual_direction IS NULL)

We fetch the stock price at the time the signal was generated (T0)
and the price 3 trading days later (T+3), compute the forward return,
and write the actual_direction + actual_return back to the Signal row.

This transforms Phase 0 synthetic predictions into Phase 1 real labels
that can be used to retrain the XGBoost model on real data.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import logging
from datetime import datetime, timedelta
from typing import List, Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Signals must be at least this old before we can check the outcome
MIN_AGE_DAYS = 3
# Return threshold to call a signal "up" or "down" (otherwise flat)
RETURN_THRESHOLD = 0.003   # 0.3 %


def _get_price_on_date(ticker: str, target_date: datetime) -> Optional[float]:
    """
    Fetch the closing price of `ticker` on or near `target_date` using yfinance.
    Looks up to 5 trading days forward to handle weekends/holidays.
    Returns None if unavailable.
    """
    try:
        import yfinance as yf
        start = target_date.strftime("%Y-%m-%d")
        end   = (target_date + timedelta(days=8)).strftime("%Y-%m-%d")
        df    = yf.download(ticker, start=start, end=end,
                            auto_adjust=True, progress=False)
        if df is None or df.empty:
            return None
        close = df["Close"].dropna()
        if close.empty:
            return None
        return float(close.iloc[0])
    except Exception as e:
        logger.debug(f"[LABEL] Price fetch failed {ticker} @ {target_date}: {e}")
        return None


def label_signal(signal_row) -> Tuple[Optional[str], Optional[float]]:
    """
    Given a Signal ORM row, return (actual_direction, actual_return_pct).
    Returns (None, None) if prices are unavailable.
    """
    generated_at = signal_row.generated_at
    ticker       = signal_row.ticker

    # Price at signal generation time (T0)
    price_t0 = _get_price_on_date(ticker, generated_at)
    if price_t0 is None or price_t0 <= 0:
        return None, None

    # Price 3 trading days later (T+3)
    t3 = generated_at + timedelta(days=5)   # use +5 calendar days ≈ +3 trading days
    price_t3 = _get_price_on_date(ticker, t3)
    if price_t3 is None or price_t3 <= 0:
        return None, None

    ret = (price_t3 - price_t0) / price_t0

    if ret > RETURN_THRESHOLD:
        direction = "up"
    elif ret < -RETURN_THRESHOLD:
        direction = "down"
    else:
        direction = "flat"

    return direction, round(float(ret), 6)


async def run_labeling(db, max_signals: int = 200) -> dict:
    """
    Async function — label up to `max_signals` unlabeled signals.
    Designed to be called from a FastAPI background task or scheduler.

    Returns a summary dict with counts.
    """
    from sqlalchemy import select, and_
    from db.models import Signal

    cutoff = datetime.utcnow() - timedelta(days=MIN_AGE_DAYS)

    # Fetch unlabeled signals older than MIN_AGE_DAYS
    result = await db.execute(
        select(Signal)
        .where(
            and_(
                Signal.generated_at <= cutoff,
                Signal.actual_direction.is_(None),
            )
        )
        .order_by(Signal.generated_at.asc())
        .limit(max_signals)
    )
    signals: List = result.scalars().all()

    if not signals:
        logger.info("[LABEL] No unlabeled signals eligible for labeling.")
        return {"labeled": 0, "skipped": 0, "total_eligible": 0}

    labeled = 0
    skipped = 0

    for sig in signals:
        direction, ret = label_signal(sig)
        if direction is not None:
            sig.actual_direction  = direction
            sig.actual_return     = ret
            sig.outcome_labeled_at = datetime.utcnow()
            labeled += 1
        else:
            skipped += 1

    await db.commit()

    logger.info(f"[LABEL] Done — labeled={labeled}, skipped={skipped}")
    return {
        "labeled":         labeled,
        "skipped":         skipped,
        "total_eligible":  len(signals),
    }


# ── CLI runner ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import asyncio
    from db.database import async_session

    async def _main():
        async with async_session() as db:
            result = await run_labeling(db, max_signals=500)
            print(result)

    asyncio.run(_main())
