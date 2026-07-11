"""
ingestion/feed_poller.py
========================
Polls free RSS news feeds + Yahoo Finance news.
100% free — no API keys required.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import feedparser
import yfinance as yf
import hashlib
from datetime import datetime, timezone
from typing import Optional
import logging

from config import settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
def _make_id(url: str) -> str:
    """Stable deterministic ID for deduplication."""
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def _parse_date(entry) -> datetime:
    """Parse RSS date → UTC datetime."""
    try:
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            import time
            return datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
    except Exception:
        pass
    return datetime.now(timezone.utc)


# ─────────────────────────────────────────────────────────────────────────────
def fetch_rss_articles(
    feeds: Optional[list] = None,
    max_per_feed: int = 20,
) -> list[dict]:
    """
    Fetch recent articles from free RSS feeds.
    Returns list of dicts: {id, title, text, source, url, published_at}
    """
    feeds = feeds or settings.RSS_FEEDS
    articles = []
    seen_urls = set()

    for feed_url in feeds:
        try:
            feed = feedparser.parse(feed_url)
            source = feed.feed.get("title", feed_url.split("/")[2])

            for entry in feed.entries[:max_per_feed]:
                url = entry.get("link", "")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)

                title = entry.get("title", "").strip()
                # Use summary/description as body text (full text requires scraping)
                text = entry.get("summary", entry.get("description", "")).strip()
                # Strip HTML tags
                import re
                text = re.sub(r"<[^>]+>", " ", text).strip()

                if not title or len(text) < 20:
                    continue

                articles.append({
                    "id":           _make_id(url),
                    "title":        title,
                    "text":         text,
                    "source":       source,
                    "url":          url,
                    "published_at": _parse_date(entry),
                })

            logger.info(f"[RSS] {source}: {len(feed.entries)} entries fetched")

        except Exception as e:
            logger.warning(f"[RSS] Feed failed {feed_url}: {e}")

    logger.info(f"[RSS] Total articles fetched: {len(articles)}")
    return articles


# ─────────────────────────────────────────────────────────────────────────────
def fetch_yahoo_finance_news(
    tickers: Optional[list] = None,
    max_per_ticker: int = 10,
) -> list[dict]:
    """
    Fetch news headlines from Yahoo Finance (free, no API key).
    Uses yfinance library.
    """
    tickers = tickers or settings.DEFAULT_TICKERS[:5]   # limit to avoid rate limits
    articles = []
    seen_urls = set()

    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            news = t.news or []

            for item in news[:max_per_ticker]:
                url = item.get("link", item.get("url", ""))
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)

                title = item.get("title", "").strip()
                text  = item.get("summary", title)

                if not title:
                    continue

                # Parse timestamp
                ts = item.get("providerPublishTime", 0)
                pub = datetime.fromtimestamp(ts, tz=timezone.utc) if ts else datetime.now(timezone.utc)

                articles.append({
                    "id":           _make_id(url),
                    "title":        title,
                    "text":         text,
                    "source":       item.get("publisher", "Yahoo Finance"),
                    "url":          url,
                    "published_at": pub,
                    "related_ticker": ticker,
                })

        except Exception as e:
            logger.warning(f"[YFinance] {ticker} news failed: {e}")

    logger.info(f"[YFinance] Total articles: {len(articles)}")
    return articles


# ─────────────────────────────────────────────────────────────────────────────
def fetch_all_free_news(
    rss_feeds: Optional[list] = None,
    tickers: Optional[list] = None,
    deduplicate: bool = True,
) -> list[dict]:
    """
    Master ingestion function — combines RSS + Yahoo Finance.
    Deduplicates by article ID (URL hash).
    """
    rss_articles   = fetch_rss_articles(feeds=rss_feeds)
    yahoo_articles = fetch_yahoo_finance_news(tickers=tickers)

    all_articles = rss_articles + yahoo_articles

    if deduplicate:
        seen = set()
        deduped = []
        for a in all_articles:
            if a["id"] not in seen:
                seen.add(a["id"])
                deduped.append(a)
        all_articles = deduped

    # Sort by publication time (newest first)
    all_articles.sort(key=lambda x: x["published_at"], reverse=True)

    logger.info(f"[INGESTION] Total unique articles: {len(all_articles)}")
    return all_articles


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)

    print("=== Testing RSS Feed Poller ===")
    articles = fetch_all_free_news(tickers=["AAPL", "MSFT"])
    print(f"\nFetched {len(articles)} articles total")
    for a in articles[:3]:
        print(f"\n[{a['source']}] {a['title'][:80]}")
        print(f"  Published: {a['published_at']}")
        print(f"  URL: {a['url'][:60]}...")
