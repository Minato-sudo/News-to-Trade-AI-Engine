# News-to-Trade AI Engine 🚀

> **Real-time financial signal intelligence powered by AI news clustering.**  
> Built on the CLUST-MCMS-P parallel event-centric news clustering pipeline.

## Quick Start (Local)

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Start the full stack (API + frontend)
bash run.sh
```

Then open:
- **API Docs**: http://localhost:8000/docs
- **Dashboard**: http://localhost:3000

---

## Deployment (Production)

This project is fully containerized and production-ready. 

**Backend (Hugging Face Spaces / Docker):**
1. Create a Docker Space on Hugging Face.
2. Link this GitHub repo. It will automatically build and expose port `7860`.
3. Add `DATABASE_URL` pointing to a PostgreSQL instance for infinite scaling.

**Frontend (Vercel):**
1. Import this repo into Vercel.
2. Select the `frontend` root directory.
3. Add the `NEXT_PUBLIC_API_URL` environment variable pointing to your Hugging Face space URL.

---

## Architecture

```
Stock_Project/
├── core/              ← NLP Pipeline (Stage 1/2/3 from your research)
│   ├── stage1_embedding.py      ← Parallel MiniLM embedding (all-MiniLM-L6-v2)
│   ├── stage2_clustering.py     ← Temporal-semantic K-Means
│   └── stage3_summarization.py  ← BART-large-CNN summarization
│
├── ingestion/         ← Free news feeds (RSS + Yahoo Finance)
│   └── feed_poller.py
│
├── intelligence/      ← AI signal generation
│   ├── impact_classifier.py     ← FinBERT (ProsusAI/finbert)
│   ├── rag_retrieval.py         ← Qdrant vector DB (local/in-memory)
│   └── quant_fusion.py          ← XGBoost fusion
│
├── trading/           ← Alpaca paper trading (free)
│   └── paper_trader.py
│
├── api/               ← FastAPI (multi-tenant, JWT auth)
│   ├── main.py
│   ├── auth.py
│   └── routes/
│
├── frontend/          ← Next.js dashboard
│   └── src/app/
│
└── db/                ← SQLAlchemy + SQLite (dev)
```

## Signal Generation Pipeline

```
News Headline
    │
    ▼
Stage 1: MiniLM Embedding (384-dim, parallel)
    │
    ├─► FinBERT Impact Score (positive/negative/neutral)
    │
    ├─► Qdrant RAG: retrieve top-5 historical analogs
    │
    └─► XGBoost Fusion: NLP + RAG + yfinance market features
            │
            ▼
        Signal: direction (up/down/flat) + confidence
            │
            ▼
        Alpaca Paper Trade (optional)
```

## Free Data Sources (No API Keys Required!)

| Source | What We Use | Cost |
|---|---|---|
| Reuters RSS | Business news | Free |
| NY Times RSS | Financial news | Free |
| Yahoo Finance RSS | Market news | Free |
| CNBC RSS | Breaking news | Free |
| yfinance | Price data, market features | Free |
| HuggingFace | FinBERT, BART, MiniLM models | Free |
| Qdrant (local) | Vector DB | Free |
| SQLite/PostgreSQL | Database (Local/Prod) | Free |
| Alpaca | Paper trading | Free (no real $) |

## Optional: Alpaca Paper Trading

1. Create a free account at https://alpaca.markets/
2. Get your paper trading API keys
3. Add to `.env`:
   ```
   ALPACA_API_KEY=your_key_here
   ALPACA_SECRET_KEY=your_secret_here
   ```
4. The system will automatically use real Alpaca paper orders instead of simulation

Without API keys, the system runs in **simulation mode** — identical behavior, just locally simulated orders.

## ML Metrics (from your research)

| Stage | Metric | Achieved | Target |
|---|---|---|---|
| Stage 1 (PDC) | Speedup S(4) | ~2.1x | ≥ 1.67x ✅ |
| Stage 2 (NLP) | NMI | 0.72+ | ≥ 0.70 ✅ |
| Stage 3 (NLP) | ROUGE-2 | 30.8+ | ≥ 28.0 ✅ |

## Development

```bash
# API only
source venv/bin/activate
python -m uvicorn api.main:app --reload --port 8000

# Frontend only
cd frontend && npm run dev

# Test feed poller
source venv/bin/activate && python ingestion/feed_poller.py

# Test FinBERT classifier
source venv/bin/activate && python intelligence/impact_classifier.py

# Test Qdrant RAG
source venv/bin/activate && python intelligence/rag_retrieval.py

# Test XGBoost fusion
source venv/bin/activate && python intelligence/quant_fusion.py
```
