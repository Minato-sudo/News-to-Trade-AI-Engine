"""
api/routes/clusters.py — News cluster / storyline explorer endpoints
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, func

from db.database import get_db
from db.models import Cluster, NewsArticle, User
from api.auth import get_current_user

router = APIRouter(prefix="/api/clusters", tags=["clusters"])


class ArticleOut(BaseModel):
    id: int
    title: str
    source: Optional[str]
    url: Optional[str]
    published: Optional[datetime]
    class Config:
        from_attributes = True


class ClusterOut(BaseModel):
    id: int
    label: Optional[str]
    summary: Optional[str]
    size: int
    created_at: datetime
    updated_at: datetime
    class Config:
        from_attributes = True


class ClusterDetail(ClusterOut):
    articles: list[ArticleOut] = []


@router.get("/", response_model=list[ClusterOut])
async def list_clusters(
    limit: int = Query(20, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List all event clusters (sorted by newest)."""
    result = await db.execute(
        select(Cluster).order_by(desc(Cluster.updated_at)).offset(offset).limit(limit)
    )
    return result.scalars().all()


@router.get("/{cluster_id}", response_model=ClusterDetail)
async def get_cluster(
    cluster_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get a single cluster with its articles."""
    result = await db.execute(select(Cluster).where(Cluster.id == cluster_id))
    cluster = result.scalar_one_or_none()
    if not cluster:
        raise HTTPException(status_code=404, detail="Cluster not found")

    art_result = await db.execute(
        select(NewsArticle).where(NewsArticle.cluster_id == cluster_id).limit(20)
    )
    articles = art_result.scalars().all()

    return ClusterDetail(
        id=cluster.id,
        label=cluster.label,
        summary=cluster.summary,
        size=cluster.size,
        created_at=cluster.created_at,
        updated_at=cluster.updated_at,
        articles=articles,
    )


from fastapi import BackgroundTasks

async def background_run_pipeline(n_articles: int):
    from db.database import AsyncSessionLocal
    from db.models import NewsArticle, Cluster
    from sqlalchemy import select, update, delete
    import pandas as pd
    import numpy as np
    import os
    
    async with AsyncSessionLocal() as db:
        # 1. Fetch articles from DB
        res = await db.execute(select(NewsArticle))
        articles = res.scalars().all()
        
        # If DB is empty, seed from CSV first
        if len(articles) < 10:
            csv_path = "data/raw_dataset_sample.csv"
            if os.path.exists(csv_path):
                df_csv = pd.read_csv(csv_path).head(n_articles)
                for _, row in df_csv.iterrows():
                    art = NewsArticle(
                        title=str(row.get("title", ""))[:200],
                        text=str(row.get("text", "")),
                        source="Dataset Sample",
                        published=datetime.utcnow() if "timestamp" not in row else datetime.fromtimestamp(float(row["timestamp"])),
                    )
                    db.add(art)
                await db.commit()
                # Re-fetch
                res = await db.execute(select(NewsArticle))
                articles = res.scalars().all()
                
        if not articles:
            return
            
        articles = articles[:n_articles]
        texts = [a.text for a in articles]
        timestamps = np.array([
            a.published.timestamp() if a.published else float(i * 3600)
            for i, a in enumerate(articles)
        ])
        
        # 2. Get embeddings
        from sentence_transformers import SentenceTransformer
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        
        # Save embeddings back to DB articles
        for art, emb in zip(articles, embeddings):
            art.embedding = emb.tolist()
            
        # 3. Cluster using Stage 2 logic
        from core.stage2_clustering import build_temporal_semantic_features, run_kmeans
        features = build_temporal_semantic_features(embeddings, timestamps)
        
        # Determine K
        k = max(2, min(len(articles) // 10, 15))
        labels, km, _ = run_kmeans(features, k)
        
        # Delete existing clusters to start fresh
        await db.execute(update(NewsArticle).values(cluster_id=None))
        await db.execute(delete(Cluster))
        await db.commit()
        
        # 4. Summarize and save clusters
        from core.stage3_summarization import build_summarizer, select_top_k_articles, compute_cluster_centroids
        centroids = compute_cluster_centroids(embeddings, labels)
        
        # Group articles
        cluster_articles_texts = select_top_k_articles(
            pd.DataFrame({"text": texts}),
            embeddings,
            labels,
            centroids,
            k=3
        )
        
        # Load summarizer with fallback: BART -> t5-small -> Extractive fallback
        summarizer = None
        try:
            from core.stage3_summarization import build_summarizer
            summarizer = build_summarizer()
        except Exception as e:
            print(f"⚠️ Primary summarizer failed to load ({e}). Trying fallback model...")
            try:
                from transformers import pipeline
                import torch
                device = 0 if torch.cuda.is_available() else -1
                summarizer = pipeline("summarization", model="t5-small", device=device)
                print("✅ Fallback model (t5-small) loaded successfully!")
            except Exception as e2:
                print(f"⚠️ Fallback model failed ({e2}). Using extractive rules.")
        
        for cid in np.unique(labels):
            texts_in_cluster = cluster_articles_texts.get(cid, [])
            combined = " ".join(texts_in_cluster)
            
            summary_txt = ""
            if combined:
                if summarizer is not None:
                    try:
                        out = summarizer(combined[:1024], max_length=100, min_length=20, do_sample=False, truncation=True)
                        summary_txt = out[0]["summary_text"]
                    except Exception as e:
                        # extractive fallback
                        import re
                        sents = []
                        for txt in texts_in_cluster:
                            s = [x.strip() for x in re.split(r'(?<=[.!?])\s+', txt) if x.strip()]
                            if s: sents.append(s[0])
                        summary_txt = " ".join(sents[:3])
                else:
                    # extractive fallback
                    import re
                    sents = []
                    for txt in texts_in_cluster:
                        s = [x.strip() for x in re.split(r'(?<=[.!?])\s+', txt) if x.strip()]
                        if s: sents.append(s[0])
                    summary_txt = " ".join(sents[:3])
                    
            label_name = f"Event Group #{cid+1}"
            if texts_in_cluster:
                # Use clean snippet/title
                label_name = texts_in_cluster[0][:60] + "..."
                
            db_cluster = Cluster(
                id=int(cid + 1),
                label=label_name,
                summary=summary_txt,
                size=int((labels == cid).sum()),
                centroid=centroids[cid].tolist(),
            )
            db.add(db_cluster)
            
        await db.commit()
        
        # 5. Link articles to clusters
        for art, label in zip(articles, labels):
            art.cluster_id = int(label + 1)
            
        await db.commit()


@router.post("/run-pipeline")
async def run_pipeline(
    background_tasks: BackgroundTasks,
    n_articles: int = Query(200, description="Number of articles to cluster"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Trigger the full NLP pipeline: fetch → embed → cluster → summarize.
    Runs as a background job.
    """
    background_tasks.add_task(background_run_pipeline, n_articles)
    return {
        "status": "accepted",
        "message": f"Pipeline started for {n_articles} articles. Check /api/clusters for results.",
        "n_articles": n_articles,
    }
