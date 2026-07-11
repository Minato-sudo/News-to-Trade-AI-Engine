"""
api/routes/signals.py — Financial signal feed endpoints
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from datetime import datetime, timedelta
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, func

from db.database import get_db
from db.models import Signal, User, Cluster, NewsArticle
from api.auth import get_current_user
import yfinance as yf

router = APIRouter(prefix="/api/signals", tags=["signals"])


# ── Output Models ──────────────────────────────────────────────────────────────
class SignalOut(BaseModel):
    id: int
    ticker: str
    direction: str
    confidence: float
    impact_score: float
    quant_score: Optional[float]
    headline: str
    generated_at: datetime
    acted_on: bool
    cluster_id: Optional[int] = None
    plain_english: Optional[str] = None
    action_hint: Optional[str] = None
    prob_up: Optional[float] = None
    prob_down: Optional[float] = None
    prob_flat: Optional[float] = None

    class Config:
        from_attributes = True


class GenerateSignalRequest(BaseModel):
    ticker: str
    headline: str
    text: Optional[str] = ""


class TickerAnalysis(BaseModel):
    ticker: str
    signal_count: int
    overall_direction: str
    overall_confidence: float
    up_count: int
    down_count: int
    flat_count: int
    avg_impact: float
    plain_english: str
    price_history: List[dict]   # [{date, price}]
    projected_prices: List[dict]  # [{date, price}] — 3-day projection


# ── Helper: Plain-English Generator ──────────────────────────────────────────
def _generate_plain_english(
    direction: str,
    confidence: float,
    impact_score: float,
    ticker: str,
    prob_up: float,
    prob_down: float,
    prob_flat: float,
    n_analogs: int = 0,
) -> tuple[str, str]:
    """
    Returns (plain_english_paragraph, action_hint) based purely on signal data.
    No models — pure rules. Readable, honest, jargon-free.
    """
    conf_pct = round(confidence * 100)
    impact_pct = round(abs(impact_score) * 100)
    sign = "positive" if impact_score > 0 else "negative" if impact_score < 0 else "neutral"

    # Confidence level label
    if conf_pct >= 70:
        conf_label = "strong"
    elif conf_pct >= 58:
        conf_label = "moderate"
    else:
        conf_label = "weak"

    # Build the story
    if direction == "up":
        sentiment_line = (
            f"The AI found {sign} sentiment in this news ({impact_pct}pts above neutral), "
            f"and similar past events moved {ticker} upward in price. "
            f"The model sees a {conf_pct}% ({conf_label}) probability that this news "
            f"pushes the stock price higher in the near term."
        )
        if conf_pct >= 65:
            what_to_do = (
                f"This looks like a potentially favourable news event for {ticker}. "
                f"If you already hold {ticker}, this supports keeping it. "
                f"If you're looking to enter, this adds some support — but always check the broader market first."
            )
            action_hint = "Consider Holding / Bullish Lean"
        else:
            what_to_do = (
                f"The signal leans upward but with only {conf_pct}% confidence — "
                f"meaning the system isn't very sure. "
                f"It's safer to watch for a day or two before acting on this."
            )
            action_hint = "Hold — Watch and Wait"

    elif direction == "down":
        sentiment_line = (
            f"The AI found {sign} sentiment in this news ({impact_pct}pts below neutral), "
            f"and similar past events moved {ticker} downward in price. "
            f"The model sees a {conf_pct}% ({conf_label}) probability that this news "
            f"puts pressure on the stock price in the near term."
        )
        if conf_pct >= 65:
            what_to_do = (
                f"This news looks unfavourable for {ticker}. "
                f"If you hold {ticker}, review whether your position is within your risk limits. "
                f"If you were planning to buy, it may be worth waiting a few days to see if this plays out."
            )
            action_hint = "Caution — Bearish Lean"
        else:
            what_to_do = (
                f"The signal leans downward but confidence is only {conf_pct}% — "
                f"the system is uncertain. "
                f"Avoid making big moves on this signal alone. Wait for more news or price confirmation."
            )
            action_hint = "Hold — Low Confidence"

    else:  # flat
        sentiment_line = (
            f"The AI found mostly neutral sentiment in this news for {ticker}. "
            f"Past similar events did not produce a clear directional move in the stock. "
            f"The model gives roughly equal odds to upward ({round(prob_up*100)}%) "
            f"and downward ({round(prob_down*100)}%) outcomes."
        )
        what_to_do = (
            f"This news doesn't seem to be a strong catalyst in either direction for {ticker} right now. "
            f"It's a good time to hold your current position and let other news or earnings reports "
            f"give a clearer picture."
        )
        action_hint = "Hold — Neutral Signal"

    note = ""
    if n_analogs > 0:
        note = f" (Based on {n_analogs} similar past events found in our historical database.)"

    plain_english = f"{sentiment_line} {what_to_do}{note}"
    return plain_english, action_hint


def _generate_ticker_plain_english(
    ticker: str,
    overall_direction: str,
    overall_confidence: float,
    up_count: int,
    down_count: int,
    flat_count: int,
    signal_count: int,
) -> str:
    """Plain-English summary for the overall ticker picture across all signals."""
    if signal_count == 0:
        return f"No recent news signals found for {ticker}."

    bull_pct = round(up_count / signal_count * 100) if signal_count else 0
    bear_pct = round(down_count / signal_count * 100) if signal_count else 0
    conf_pct = round(overall_confidence * 100)

    if overall_direction == "up":
        return (
            f"Across {signal_count} recent news articles about {ticker}, "
            f"{bull_pct}% were bullish vs {bear_pct}% bearish. "
            f"The overall news environment for {ticker} is currently positive, "
            f"with {conf_pct}% average model confidence. "
            f"This does not guarantee a price rise — markets can move against the news — "
            f"but the current news flow is leaning in {ticker}'s favour."
        )
    elif overall_direction == "down":
        return (
            f"Across {signal_count} recent news articles about {ticker}, "
            f"{bear_pct}% were bearish vs {bull_pct}% bullish. "
            f"The overall news environment for {ticker} is currently under pressure, "
            f"with {conf_pct}% average model confidence. "
            f"It may be worth being cautious until the news flow stabilises."
        )
    else:
        return (
            f"Across {signal_count} recent news articles about {ticker}, "
            f"signals are mixed: {bull_pct}% bullish, {bear_pct}% bearish, "
            f"and the rest neutral. The overall news picture is unclear right now. "
            f"No strong directional call can be made from news alone."
        )


# ── Endpoints ─────────────────────────────────────────────────────────────────
@router.get("/", response_model=list[SignalOut])
async def get_signals(
    ticker: Optional[str] = Query(None, description="Filter by ticker"),
    direction: Optional[str] = Query(None, description="up | down | flat"),
    limit: int = Query(50, le=200),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get recent signals for the authenticated user."""
    query = (
        select(Signal)
        .where(Signal.user_id == current_user.id)
        .order_by(desc(Signal.generated_at))
        .limit(limit)
    )
    if ticker:
        query = query.where(Signal.ticker == ticker.upper())
    if direction:
        query = query.where(Signal.direction == direction.lower())

    result = await db.execute(query)
    return result.scalars().all()


@router.get("/ticker-analysis/{ticker}", response_model=TickerAnalysis)
async def get_ticker_analysis(
    ticker: str,
    days: int = Query(30, description="Look-back window in days"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Holistic view: aggregate of ALL recent signals for this ticker (all users),
    plus 30-day price history and 3-day projection.
    """
    import yfinance as yf
    import numpy as np

    ticker = ticker.upper()
    cutoff = datetime.utcnow() - timedelta(days=days)

    # Pull all signals for this ticker (across all users for broader picture)
    result = await db.execute(
        select(Signal)
        .where(Signal.ticker == ticker, Signal.generated_at >= cutoff)
        .order_by(desc(Signal.generated_at))
        .limit(100)
    )
    sigs = result.scalars().all()

    # Aggregate
    up_count   = sum(1 for s in sigs if s.direction == "up")
    down_count = sum(1 for s in sigs if s.direction == "down")
    flat_count = len(sigs) - up_count - down_count
    avg_conf   = float(np.mean([s.confidence for s in sigs])) if sigs else 0.5
    avg_impact = float(np.mean([s.impact_score for s in sigs])) if sigs else 0.0

    if up_count > down_count and up_count > flat_count:
        overall_direction = "up"
    elif down_count > up_count and down_count > flat_count:
        overall_direction = "down"
    else:
        overall_direction = "flat"

    # Price history from yfinance
    price_history = []
    projected_prices = []
    try:
        tk_obj = yf.Ticker(ticker)
        df = tk_obj.history(period=f"{days + 5}d", auto_adjust=True)
        if df is not None and len(df) > 0:
            close_series = df["Close"].dropna()
            for dt_idx, price_val in close_series.items():
                dt = dt_idx if isinstance(dt_idx, datetime) else dt_idx.to_pydatetime()
                price_history.append({
                    "date": dt.strftime("%Y-%m-%d"),
                    "price": round(float(price_val), 2)
                })

            # 3-day projection: simple linear extrapolation + signal nudge
            if len(price_history) >= 5:
                last_prices = [p["price"] for p in price_history[-5:]]
                last_price  = last_prices[-1]
                trend       = (last_prices[-1] - last_prices[0]) / 4

                nudge        = avg_impact * last_price * 0.005
                daily_change = trend + nudge

                last_dt = datetime.strptime(price_history[-1]["date"], "%Y-%m-%d")
                for i in range(1, 4):
                    proj_dt = last_dt + timedelta(days=i)
                    while proj_dt.weekday() >= 5:
                        proj_dt += timedelta(days=1)
                    projected_prices.append({
                        "date":  proj_dt.strftime("%Y-%m-%d"),
                        "price": round(last_price + daily_change * i, 2),
                        "projected": True,
                    })
    except Exception as e:
        import logging
        logging.getLogger(__name__).debug(f"Price chart fetch failed for {ticker}: {e}")

    plain_english = _generate_ticker_plain_english(
        ticker, overall_direction, avg_conf,
        up_count, down_count, flat_count, len(sigs)
    )

    return TickerAnalysis(
        ticker=ticker,
        signal_count=len(sigs),
        overall_direction=overall_direction,
        overall_confidence=round(avg_conf, 4),
        up_count=up_count,
        down_count=down_count,
        flat_count=flat_count,
        avg_impact=round(avg_impact, 4),
        plain_english=plain_english,
        price_history=price_history,
        projected_prices=projected_prices,
    )


@router.post("/generate", response_model=SignalOut, status_code=201)
async def generate_signal(
    payload: GenerateSignalRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Generate a financial signal for a given ticker + headline.
    Runs FinBERT → RAG → XGBoost fusion pipeline.
    Returns plain-English description and action hint.
    """
    from intelligence.impact_classifier import classify_headline
    from intelligence.rag_retrieval import retrieve_analogs, analog_signal_prior, get_qdrant_client
    from intelligence.quant_fusion import predict_signal
    from core.stage1_embedding import embed_parallel
    import numpy as np

    text = payload.text or payload.headline

    # Step 1: FinBERT impact classification
    impact = classify_headline(text)

    # Step 2: Get embedding for RAG query
    try:
        emb, _ = embed_parallel([text], k=1)
        query_emb = emb[0].tolist()
    except Exception:
        query_emb = [0.0] * 384

    # Step 3: RAG analog retrieval
    analogs = retrieve_analogs(query_emb, top_k=5, ticker_filter=payload.ticker)
    prior   = analog_signal_prior(analogs)

    # Step 4: XGBoost quant fusion
    quant = predict_signal(
        impact_score=impact["impact_score"],
        confidence=impact["confidence"],
        analog_direction=prior["direction"],
        analog_confidence=prior["confidence"],
        analog_return=prior["mean_return"],
        n_analogs=prior["n_analogs"],
        ticker=payload.ticker,
    )

    # Step 5: Find the closest cluster by embedding cosine similarity
    related_cluster_id = None
    try:
        cluster_result = await db.execute(
            select(Cluster).order_by(desc(Cluster.updated_at)).limit(20)
        )
        clusters = cluster_result.scalars().all()
        best_sim  = -1.0
        best_cid  = None
        qv = np.array(query_emb)
        for cl in clusters:
            if cl.centroid:
                cv = np.array(cl.centroid[:len(query_emb)])
                if cv.shape == qv.shape and np.linalg.norm(cv) > 0 and np.linalg.norm(qv) > 0:
                    sim = float(np.dot(qv, cv) / (np.linalg.norm(qv) * np.linalg.norm(cv)))
                    if sim > best_sim:
                        best_sim = sim
                        best_cid = cl.id
        if best_sim > 0.3:
            related_cluster_id = best_cid
    except Exception:
        pass

    # Step 6: Generate plain-English explanation
    plain_english, action_hint = _generate_plain_english(
        direction=quant["direction"],
        confidence=quant["confidence"],
        impact_score=impact["impact_score"],
        ticker=payload.ticker.upper(),
        prob_up=quant.get("prob_up", 0.33),
        prob_down=quant.get("prob_down", 0.33),
        prob_flat=quant.get("prob_flat", 0.33),
        n_analogs=prior["n_analogs"],
    )

    # Step 7: Store signal
    signal = Signal(
        user_id=current_user.id,
        ticker=payload.ticker.upper(),
        direction=quant["direction"],
        confidence=quant["confidence"],
        impact_score=impact["impact_score"],
        quant_score=quant["quant_score"],
        headline=payload.headline[:500],
        raw_text=text[:2000],
        rag_analog=analogs[0] if analogs else None,
        cluster_id=related_cluster_id,
    )
    db.add(signal)
    await db.flush()
    await db.refresh(signal)

    # Attach non-DB fields to the ORM object for response serialisation
    signal.__dict__["plain_english"]     = plain_english
    signal.__dict__["action_hint"]       = action_hint
    signal.__dict__["prob_up"]           = quant.get("prob_up")
    signal.__dict__["prob_down"]         = quant.get("prob_down")
    signal.__dict__["prob_flat"]         = quant.get("prob_flat")

    return signal


@router.get("/{signal_id}", response_model=SignalOut)
async def get_signal(
    signal_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Signal).where(Signal.id == signal_id, Signal.user_id == current_user.id)
    )
    signal = result.scalar_one_or_none()
    if not signal:
        raise HTTPException(status_code=404, detail="Signal not found")
    return signal
