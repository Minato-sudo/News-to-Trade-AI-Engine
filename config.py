"""
config.py
=========
Central configuration for the From-Storyline-to-Signal platform.
All settings can be overridden via environment variables or a .env file.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── App ──────────────────────────────────────────────────────────────────
    APP_NAME: str = "Storyline-to-Signal"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # ── Auth (JWT) ────────────────────────────────────────────────────────────
    SECRET_KEY: str = "change-me-in-production-use-a-long-random-string"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 1 day

    # ── Database ──────────────────────────────────────────────────────────────
    DATABASE_URL: str = f"sqlite+aiosqlite:///{BASE_DIR}/db/app.db"

    # ── Paths ─────────────────────────────────────────────────────────────────
    RESULTS_DIR: Path = BASE_DIR / "results"
    DATA_DIR: Path = BASE_DIR / "data"
    CORE_DIR: Path = BASE_DIR / "core"

    # ── NLP Models (free HuggingFace) ─────────────────────────────────────────
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    SUMMARIZATION_MODEL: str = "facebook/bart-large-cnn"
    IMPACT_MODEL: str = "ProsusAI/finbert"          # Free FinBERT on HuggingFace
    EMBEDDING_WORKERS: int = 4                        # parallel workers for Stage 1
    EMBEDDING_BATCH_SIZE: int = 64

    # ── Clustering ────────────────────────────────────────────────────────────
    DEFAULT_N_CLUSTERS: int = 10
    TEMPORAL_ALPHA: float = 0.1                       # weight of timestamp feature

    # ── Qdrant (local, free) ──────────────────────────────────────────────────
    QDRANT_MODE: str = "memory"                       # "memory" or "local" or "cloud"
    QDRANT_PATH: str = str(BASE_DIR / "qdrant_storage")
    QDRANT_COLLECTION: str = "news_events"
    QDRANT_VECTOR_SIZE: int = 384                     # MiniLM output size

    # ── Free News Feeds (RSS) ─────────────────────────────────────────────────
    RSS_FEEDS: list = [
        "https://feeds.reuters.com/reuters/businessNews",
        "https://feeds.reuters.com/reuters/technologyNews",
        "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml",
        "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
        "https://finance.yahoo.com/news/rssindex",
        "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "https://feeds.content.dowjones.io/public/rss/mw_realtimeheadlines",
    ]
    FEED_POLL_INTERVAL_SECONDS: int = 300             # poll every 5 minutes

    # ── Market Data (Yahoo Finance — free) ────────────────────────────────────
    MARKET_DATA_SOURCE: str = "yfinance"
    DEFAULT_TICKERS: list = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
        "META", "NVDA", "JPM", "GS", "BAC",
        "SPY",  "QQQ",  "^GSPC",
    ]
    PRICE_HORIZON_DAYS: int = 3                       # forward return window

    # ── Alpaca Paper Trading (free account) ───────────────────────────────────
    ALPACA_API_KEY: str = ""                          # set in .env
    ALPACA_SECRET_KEY: str = ""                       # set in .env
    ALPACA_BASE_URL: str = "https://paper-api.alpaca.markets"  # paper trading
    ALPACA_DATA_URL: str = "https://data.alpaca.markets"

    # ── Email / SMTP (for password reset OTP) ─────────────────────────────────
    SMTP_EMAIL: str = ""                              # Gmail address
    SMTP_APP_PASSWORD: str = ""                       # Gmail App Password (not real password)

    # ── Signal Thresholds ─────────────────────────────────────────────────────
    SIGNAL_CONFIDENCE_THRESHOLD: float = 0.6          # min confidence to emit signal
    SIGNAL_DIRECTION_THRESHOLD: float = 0.55           # min prob for up/down call

    # ── API Rate Limiting ─────────────────────────────────────────────────────
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW_SECONDS: int = 60

    # ── CORS ─────────────────────────────────────────────────────────────────
    ALLOWED_ORIGINS: list = ["*"]


settings = Settings()
