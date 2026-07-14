"""
db/models.py
============
SQLAlchemy ORM models for the multi-tenant platform.
"""

from datetime import datetime
from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey,
    Integer, JSON, String, Text
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


# ─────────────────────────────────────────────────────────────────────────────
class User(Base):
    """Multi-tenant user — each user has isolated signals + portfolio."""
    __tablename__ = "users"

    id              = Column(Integer, primary_key=True, index=True)
    username        = Column(String(50), unique=True, index=True, nullable=False)
    email           = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active       = Column(Boolean, default=True)
    created_at      = Column(DateTime, default=datetime.utcnow)

    signals         = relationship("Signal",   back_populates="user", cascade="all, delete-orphan")
    trades          = relationship("Trade",    back_populates="user", cascade="all, delete-orphan")
    subscriptions   = relationship("Subscription", back_populates="user", cascade="all, delete-orphan")


# ─────────────────────────────────────────────────────────────────────────────
class NewsArticle(Base):
    """Ingested news article from any free RSS/Yahoo feed."""
    __tablename__ = "news_articles"

    id          = Column(Integer, primary_key=True, index=True)
    title       = Column(String(500), nullable=False)
    text        = Column(Text)
    source      = Column(String(100))
    url         = Column(String(1000), unique=True)
    published   = Column(DateTime, index=True)
    fetched_at  = Column(DateTime, default=datetime.utcnow)
    cluster_id  = Column(Integer, ForeignKey("clusters.id"), nullable=True)
    embedding   = Column(JSON, nullable=True)   # stored as list[float]

    cluster     = relationship("Cluster", back_populates="articles")


# ─────────────────────────────────────────────────────────────────────────────
class Cluster(Base):
    """Event cluster — a group of thematically related news articles."""
    __tablename__ = "clusters"

    id            = Column(Integer, primary_key=True, index=True)
    label         = Column(String(200))          # auto-generated topic label
    summary       = Column(Text)                 # BART-generated storyline
    size          = Column(Integer, default=0)   # number of articles
    centroid      = Column(JSON, nullable=True)  # 384-dim centroid vector
    created_at    = Column(DateTime, default=datetime.utcnow)
    updated_at    = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    articles      = relationship("NewsArticle", back_populates="cluster")
    signals       = relationship("Signal",  back_populates="cluster")


# ─────────────────────────────────────────────────────────────────────────────
class Signal(Base):
    """
    Financial signal derived from a news cluster.
    direction: 'up' | 'down' | 'flat'
    """
    __tablename__ = "signals"

    id              = Column(Integer, primary_key=True, index=True)
    user_id         = Column(Integer, ForeignKey("users.id"), nullable=False)
    cluster_id      = Column(Integer, ForeignKey("clusters.id"), nullable=True)
    ticker          = Column(String(20), index=True, nullable=False)
    direction       = Column(String(10))         # 'up' | 'down' | 'flat' (predicted)
    confidence      = Column(Float)              # 0.0 – 1.0
    impact_score    = Column(Float)              # FinBERT magnitude estimate
    rag_analog      = Column(JSON, nullable=True) # closest historical event
    quant_score     = Column(Float, nullable=True) # XGBoost fusion score
    headline        = Column(String(500))
    raw_text        = Column(Text, nullable=True)
    generated_at    = Column(DateTime, default=datetime.utcnow, index=True)
    acted_on        = Column(Boolean, default=False)

    # ── Phase 1: Real outcome labels (populated by training/label_outcomes.py) ──
    actual_direction   = Column(String(10), nullable=True)  # 'up' | 'down' | 'flat' (real)
    actual_return      = Column(Float,      nullable=True)  # T+3 forward return %
    outcome_labeled_at = Column(DateTime,   nullable=True)  # when the label was written

    user            = relationship("User",    back_populates="signals")
    cluster         = relationship("Cluster", back_populates="signals")


# ─────────────────────────────────────────────────────────────────────────────
class Trade(Base):
    """Paper trade executed via Alpaca (free paper trading)."""
    __tablename__ = "trades"

    id              = Column(Integer, primary_key=True, index=True)
    user_id         = Column(Integer, ForeignKey("users.id"), nullable=False)
    signal_id       = Column(Integer, ForeignKey("signals.id"), nullable=True)
    ticker          = Column(String(20), index=True)
    direction       = Column(String(10))          # 'buy' | 'sell' | 'hold'
    quantity        = Column(Float)
    entry_price     = Column(Float)
    exit_price      = Column(Float, nullable=True)
    pnl             = Column(Float, nullable=True)
    status          = Column(String(20), default="open")  # open | closed | cancelled
    alpaca_order_id = Column(String(100), nullable=True)
    opened_at       = Column(DateTime, default=datetime.utcnow)
    closed_at       = Column(DateTime, nullable=True)

    user            = relationship("User", back_populates="trades")


# ─────────────────────────────────────────────────────────────────────────────
class Subscription(Base):
    """User subscription to a ticker for signal notifications."""
    __tablename__ = "subscriptions"

    id          = Column(Integer, primary_key=True, index=True)
    user_id     = Column(Integer, ForeignKey("users.id"), nullable=False)
    ticker      = Column(String(20), nullable=False)
    active      = Column(Boolean, default=True)
    created_at  = Column(DateTime, default=datetime.utcnow)

    user        = relationship("User", back_populates="subscriptions")


# ─────────────────────────────────────────────────────────────────────────────
class HistoricalEvent(Base):
    """Stored historical event for RAG retrieval (Qdrant analog)."""
    __tablename__ = "historical_events"

    id              = Column(Integer, primary_key=True, index=True)
    headline        = Column(String(500))
    summary         = Column(Text)
    ticker          = Column(String(20))
    direction       = Column(String(10))         # actual realized direction
    return_pct      = Column(Float)              # realized forward return
    event_date      = Column(DateTime)
    embedding       = Column(JSON)               # 384-dim vector
    stored_at       = Column(DateTime, default=datetime.utcnow)
