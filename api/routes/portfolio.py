"""
api/routes/portfolio.py — Paper trading portfolio endpoints
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

from db.database import get_db
from db.models import Trade, Signal, User
from api.auth import get_current_user

router = APIRouter(prefix="/api/portfolio", tags=["portfolio"])


class TradeOut(BaseModel):
    id: int
    ticker: str
    direction: str
    quantity: Optional[float]
    entry_price: Optional[float]
    exit_price: Optional[float]
    pnl: Optional[float]
    status: str
    opened_at: datetime
    closed_at: Optional[datetime]
    class Config:
        from_attributes = True


class TradeRequest(BaseModel):
    signal_id: int
    auto_size: bool = True   # use Kelly position sizing


@router.get("/account")
async def get_account(current_user: User = Depends(get_current_user)):
    """Get paper trading account summary."""
    from trading.paper_trader import get_account_info, get_open_positions
    account = get_account_info()
    positions = get_open_positions()
    return {
        "account": account,
        "open_positions": positions,
        "n_positions": len(positions),
    }


@router.post("/trade", response_model=TradeOut, status_code=201)
async def execute_trade(
    payload: TradeRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Execute a paper trade based on an existing signal."""
    from trading.paper_trader import place_paper_order

    # Load signal
    result = await db.execute(
        select(Signal).where(Signal.id == payload.signal_id, Signal.user_id == current_user.id)
    )
    signal = result.scalar_one_or_none()
    if not signal:
        raise HTTPException(status_code=404, detail="Signal not found")
    if signal.acted_on:
        raise HTTPException(status_code=400, detail="Signal already acted on")

    # Execute paper trade
    order = place_paper_order(
        ticker=signal.ticker,
        direction=signal.direction,
        signal_confidence=signal.confidence or 0.5,
    )

    if order.get("status") in ("skipped", "failed"):
        raise HTTPException(status_code=400, detail=order.get("reason", order.get("error", "trade failed")))

    # Store trade
    trade = Trade(
        user_id=current_user.id,
        signal_id=signal.id,
        ticker=signal.ticker,
        direction="buy" if signal.direction == "up" else "sell",
        quantity=order.get("quantity", 0),
        entry_price=order.get("price", 0.0),
        alpaca_order_id=order.get("order_id"),
        status="open",
    )
    db.add(trade)

    # Mark signal as acted on
    signal.acted_on = True
    await db.flush()
    await db.refresh(trade)
    return trade


@router.get("/trades", response_model=list[TradeOut])
async def get_trades(
    status: Optional[str] = Query(None, description="open | closed"),
    limit: int = Query(50, le=200),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get trade history for current user."""
    query = (
        select(Trade)
        .where(Trade.user_id == current_user.id)
        .order_by(desc(Trade.opened_at))
        .limit(limit)
    )
    if status:
        query = query.where(Trade.status == status)
    result = await db.execute(query)
    return result.scalars().all()


@router.get("/summary")
async def get_portfolio_summary(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get P&L summary for current user."""
    result = await db.execute(
        select(Trade).where(Trade.user_id == current_user.id, Trade.status == "closed")
    )
    closed_trades = result.scalars().all()

    total_pnl = sum(t.pnl or 0 for t in closed_trades)
    wins  = sum(1 for t in closed_trades if (t.pnl or 0) > 0)
    total = len(closed_trades)
    win_rate = (wins / total * 100) if total > 0 else 0.0

    return {
        "total_closed_trades": total,
        "total_pnl":           round(total_pnl, 2),
        "win_rate_pct":        round(win_rate, 1),
        "wins":                wins,
        "losses":              total - wins,
    }
