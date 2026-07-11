"""
trading/paper_trader.py
=======================
Alpaca paper trading integration (free account, no real capital).
All operations target Alpaca's paper trading endpoint only.

Setup: Create a free account at https://alpaca.markets/
       Add ALPACA_API_KEY and ALPACA_SECRET_KEY to your .env file.
       Paper trading is completely free — no real money involved.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import logging
from datetime import datetime, timedelta
from typing import Optional
import numpy as np

from config import settings

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
_trading_client = None
_data_client = None


def get_trading_client():
    """Lazy-load Alpaca paper trading client."""
    global _trading_client
    if _trading_client is None:
        if not settings.ALPACA_API_KEY or not settings.ALPACA_SECRET_KEY:
            logger.warning("[TRADE] No Alpaca API keys set — running in simulation mode.")
            return None
        try:
            from alpaca.trading.client import TradingClient
            _trading_client = TradingClient(
                api_key=settings.ALPACA_API_KEY,
                secret_key=settings.ALPACA_SECRET_KEY,
                paper=True,   # ← always paper trading
            )
            logger.info("[TRADE] Alpaca paper trading client connected.")
        except Exception as e:
            logger.error(f"[TRADE] Alpaca connection failed: {e}")
            return None
    return _trading_client


# ─────────────────────────────────────────────────────────────────────────────
def get_account_info() -> dict:
    """Get paper trading account balance and equity."""
    client = get_trading_client()
    if client is None:
        # Simulation mode
        return {
            "equity":          10000.0,
            "cash":            10000.0,
            "buying_power":    10000.0,
            "portfolio_value": 10000.0,
            "mode":            "simulation",
        }
    try:
        account = client.get_account()
        return {
            "equity":          float(account.equity),
            "cash":            float(account.cash),
            "buying_power":    float(account.buying_power),
            "portfolio_value": float(account.portfolio_value),
            "mode":            "paper",
        }
    except Exception as e:
        logger.error(f"[TRADE] get_account failed: {e}")
        return {"equity": 0.0, "cash": 0.0, "buying_power": 0.0, "portfolio_value": 0.0, "mode": "error"}


# ─────────────────────────────────────────────────────────────────────────────
def place_paper_order(
    ticker:    str,
    direction: str,        # 'up' → buy, 'down' → sell/short
    signal_confidence: float = 0.6,
    max_position_pct:  float = 0.05,   # max 5% of portfolio per trade
) -> dict:
    """
    Place a paper trade based on a signal.

    Parameters
    ----------
    ticker             : Stock ticker symbol
    direction          : 'up' (buy) | 'down' (short sell) | 'flat' (skip)
    signal_confidence  : 0–1 confidence from XGBoost fusion
    max_position_pct   : Max portfolio % to risk per trade

    Returns order details dict.
    """
    if direction == "flat":
        logger.info(f"[TRADE] Signal is flat for {ticker} — no trade placed.")
        return {"status": "skipped", "reason": "flat signal", "ticker": ticker}

    if signal_confidence < settings.SIGNAL_CONFIDENCE_THRESHOLD:
        logger.info(f"[TRADE] Low confidence ({signal_confidence:.2f}) for {ticker} — skip.")
        return {"status": "skipped", "reason": "low confidence", "ticker": ticker}

    client = get_trading_client()

    # Get account equity to size position
    account = get_account_info()
    equity = account.get("equity", 10000.0)

    # Position sizing: Kelly-inspired, capped at max_position_pct
    position_value = equity * min(max_position_pct, signal_confidence * max_position_pct * 2)

    if client is None:
        # Simulation mode — return a fake order
        import random
        fake_price = round(random.uniform(50, 500), 2)
        qty = max(1, int(position_value / fake_price))
        order_id = f"sim-{ticker}-{int(datetime.utcnow().timestamp())}"
        logger.info(f"[TRADE][SIM] {direction.upper()} {qty} shares of {ticker} @ ~${fake_price:.2f}")
        return {
            "status":     "filled",
            "mode":       "simulation",
            "order_id":   order_id,
            "ticker":     ticker,
            "direction":  direction,
            "quantity":   qty,
            "price":      fake_price,
            "value":      round(qty * fake_price, 2),
            "placed_at":  datetime.utcnow().isoformat(),
        }

    # Real paper order via Alpaca
    try:
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        # Get current price to compute quantity
        import yfinance as yf
        ticker_data = yf.Ticker(ticker)
        hist = ticker_data.history(period="1d")
        current_price = float(hist["Close"].iloc[-1]) if not hist.empty else 100.0
        qty = max(1, int(position_value / current_price))

        side = OrderSide.BUY if direction == "up" else OrderSide.SELL

        order_request = MarketOrderRequest(
            symbol=ticker,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY,
        )
        order = client.submit_order(order_request)

        logger.info(f"[TRADE][PAPER] {side.value.upper()} {qty} {ticker} — order {order.id}")
        return {
            "status":    "submitted",
            "mode":      "paper",
            "order_id":  str(order.id),
            "ticker":    ticker,
            "direction": direction,
            "quantity":  qty,
            "price":     current_price,
            "value":     round(qty * current_price, 2),
            "placed_at": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"[TRADE] Order failed for {ticker}: {e}")
        return {"status": "failed", "error": str(e), "ticker": ticker}


# ─────────────────────────────────────────────────────────────────────────────
def get_open_positions() -> list[dict]:
    """Return all open paper positions."""
    client = get_trading_client()
    if client is None:
        return []   # simulation: no tracked positions
    try:
        positions = client.get_all_positions()
        return [
            {
                "ticker":     p.symbol,
                "qty":        float(p.qty),
                "avg_entry":  float(p.avg_entry_price),
                "current":    float(p.current_price),
                "market_val": float(p.market_value),
                "unrealized_pnl": float(p.unrealized_pl),
                "unrealized_pnl_pct": float(p.unrealized_plpc) * 100,
            }
            for p in positions
        ]
    except Exception as e:
        logger.error(f"[TRADE] get_positions failed: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
def close_position(ticker: str) -> dict:
    """Close a paper position in a specific ticker."""
    client = get_trading_client()
    if client is None:
        return {"status": "skipped", "mode": "simulation", "ticker": ticker}
    try:
        client.close_position(ticker)
        logger.info(f"[TRADE] Closed position: {ticker}")
        return {"status": "closed", "ticker": ticker}
    except Exception as e:
        logger.error(f"[TRADE] Close position failed for {ticker}: {e}")
        return {"status": "failed", "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== Paper Trading Test ===\n")

    # Test account info (simulation mode if no API keys)
    account = get_account_info()
    print(f"Account: {account}\n")

    # Test paper order (simulation)
    order = place_paper_order("AAPL", "up", signal_confidence=0.75)
    print(f"Order result: {order}\n")

    order2 = place_paper_order("TSLA", "flat", signal_confidence=0.9)
    print(f"Flat signal: {order2}")
