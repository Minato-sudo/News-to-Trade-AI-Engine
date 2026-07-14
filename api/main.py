"""
api/main.py
===========
FastAPI application entry point.
Multi-tenant, async, production-ready.
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import asyncio

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from loguru import logger

from config import settings
from db.database import init_db
from api.routes import users, signals, clusters, portfolio, admin


# ── Background ingestion job ──────────────────────────────────────────────────
async def _run_ingestion_cycle():
    """
    Runs every FEED_POLL_INTERVAL_SECONDS.
    Fetches RSS + Yahoo Finance news and saves new articles to DB.
    """
    try:
        from ingestion.feed_poller import fetch_all_free_news
        from db.database import async_session
        from db.models import NewsArticle
        from sqlalchemy import select

        articles = fetch_all_free_news()
        if not articles:
            return

        async with async_session() as db:
            saved = 0
            for a in articles[:100]:
                exist = await db.execute(
                    select(NewsArticle).where(NewsArticle.url == a["url"])
                )
                if exist.scalar_one_or_none():
                    continue
                art = NewsArticle(
                    title=a["title"],
                    text=a["text"],
                    source=a["source"],
                    url=a["url"],
                    published=a["published_at"],
                )
                db.add(art)
                saved += 1
            await db.commit()
        logger.info(f"[SCHEDULER] Ingestion cycle complete — {saved} new articles saved.")
    except Exception as e:
        logger.warning(f"[SCHEDULER] Ingestion cycle failed: {e}")


async def _run_labeling_cycle():
    """Label unlabeled signals with real T+3 stock outcomes (runs every 6h)."""
    try:
        from training.label_outcomes import run_labeling
        from db.database import async_session
        async with async_session() as db:
            result = await run_labeling(db, max_signals=100)
        logger.info(f"[SCHEDULER] Labeling cycle done: {result}")
    except Exception as e:
        logger.warning(f"[SCHEDULER] Labeling cycle failed: {e}")


async def _run_retraining_cycle():
    """Auto-retrain XGBoost when enough labeled samples exist (runs every 24h)."""
    try:
        from training.retrain_pipeline import run_retraining_pipeline
        from db.database import async_session
        async with async_session() as db:
            result = await run_retraining_pipeline(db)
        logger.info(f"[SCHEDULER] Retraining cycle done: {result.get('status')}")
    except Exception as e:
        logger.warning(f"[SCHEDULER] Retraining cycle failed: {e}")


async def _migrate_new_columns():
    """Add new nullable Signal columns on existing SQLite DBs without Alembic."""
    try:
        from db.database import engine
        async with engine.begin() as conn:
            for col, col_type in [
                ("actual_direction",   "TEXT"),
                ("actual_return",      "REAL"),
                ("outcome_labeled_at", "DATETIME"),
            ]:
                try:
                    await conn.execute(
                        __import__("sqlalchemy").text(
                            f"ALTER TABLE signals ADD COLUMN {col} {col_type}"
                        )
                    )
                    logger.info(f"[MIGRATE] Added column: signals.{col}")
                except Exception:
                    pass  # column already exists
    except Exception as e:
        logger.debug(f"[MIGRATE] Migration skipped: {e}")


# ── Lifespan (startup / shutdown) ────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize DB, warm up ML models, and start background scheduler."""
    logger.info("🚀 Starting Storyline-to-Signal API...")

    # Init DB tables
    await init_db()
    logger.info("✅ Database initialized")

    # Migrate any new columns (idempotent — safe to run every startup)
    await _migrate_new_columns()
    logger.info("✅ DB schema up to date")

    # Warm up Qdrant
    try:
        from intelligence.rag_retrieval import get_qdrant_client, seed_with_sample_events
        get_qdrant_client()
        seed_with_sample_events()
        logger.info("✅ Qdrant vector DB ready")
    except Exception as e:
        logger.warning(f"⚠️  Qdrant init failed: {e}")

    # Pre-train XGBoost
    try:
        from intelligence.quant_fusion import load_model
        load_model()
        logger.info("✅ XGBoost model ready")
    except Exception as e:
        logger.warning(f"⚠️  XGBoost init failed: {e}")

    # ── Start background scheduler ─────────────────────────────────────────────
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    scheduler = AsyncIOScheduler()

    # News ingestion — every N seconds (configured in settings)
    scheduler.add_job(
        _run_ingestion_cycle,
        trigger="interval",
        seconds=settings.FEED_POLL_INTERVAL_SECONDS,
        id="news_ingestion",
        replace_existing=True,
        max_instances=1,
    )

    # Outcome labeling — every 6 hours
    scheduler.add_job(
        _run_labeling_cycle,
        trigger="interval",
        hours=6,
        id="outcome_labeling",
        replace_existing=True,
        max_instances=1,
    )

    # Model retraining — every 24 hours
    scheduler.add_job(
        _run_retraining_cycle,
        trigger="interval",
        hours=24,
        id="model_retraining",
        replace_existing=True,
        max_instances=1,
    )

    scheduler.start()
    logger.info(f"✅ Scheduler started — ingestion every {settings.FEED_POLL_INTERVAL_SECONDS}s, labeling every 6h, retraining every 24h")

    # Run first ingestion immediately
    asyncio.create_task(_run_ingestion_cycle())

    logger.info("✅ API ready — all systems go!")
    yield

    scheduler.shutdown(wait=False)
    logger.info("👋 Shutting down...")


# ── App Factory ───────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="From Storyline to Signal",
    description=(
        "Real-time financial signal intelligence platform. "
        "Built on the CLUST-MCMS-P parallel news clustering pipeline."
    ),
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS — allow Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(users.router)
app.include_router(signals.router)
app.include_router(clusters.router)
app.include_router(portfolio.router)
app.include_router(admin.router)


# ── Root endpoints ────────────────────────────────────────────────────────────
@app.get("/", tags=["health"])
async def root():
    return {
        "name":    settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status":  "running",
        "docs":    "/docs",
    }


@app.get("/health", tags=["health"])
async def health():
    return {"status": "ok", "timestamp": __import__("datetime").datetime.utcnow().isoformat()}


# ── Ingestion endpoint (public, no auth required) ────────────────────────────
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from db.database import get_db
from db.models import NewsArticle
from sqlalchemy import select

@app.post("/api/ingest/fetch", tags=["ingestion"])
@limiter.limit(f"{settings.RATE_LIMIT_REQUESTS}/minute")
async def trigger_ingestion(request: Request, db: AsyncSession = Depends(get_db), max_articles: int = 50):
    """
    Manually trigger a news ingestion cycle.
    Fetches from all free RSS feeds + Yahoo Finance.
    Saves articles to DB.
    """
    from ingestion.feed_poller import fetch_all_free_news
    articles = fetch_all_free_news()
    
    saved_count = 0
    for a in articles[:max_articles]:
        # Avoid duplicate URLs
        exist = await db.execute(select(NewsArticle).where(NewsArticle.url == a["url"]))
        if exist.scalar_one_or_none():
            continue
            
        art = NewsArticle(
            title=a["title"],
            text=a["text"],
            source=a["source"],
            url=a["url"],
            published=a["published_at"],
        )
        db.add(art)
        saved_count += 1
        
    await db.commit()
    
    return {
        "status": "fetched",
        "count": len(articles),
        "saved": saved_count,
        "sample": [
            {"title": a["title"][:80], "source": a["source"]}
            for a in articles[:5]
        ],
    }


# ── Dev server ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        workers=1,
    )
