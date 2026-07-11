"""
data_loader.py
==============
Loads and prepares two datasets:
  1. CC-News  (HuggingFace) — used for PDC timing benchmarks
  2. All-the-News (built-in fallback or Kaggle) — used for NLP evaluation (NMI/ARI/ROUGE)

Outputs clean pandas DataFrames with columns:
  text      : article body (cleaned)
  title     : headline
  timestamp : Unix timestamp (float, used for temporal clustering)
  label     : ground-truth topic label (All-the-News only, for NMI/ARI)
"""

import os
import time
import numpy as np
import pandas as pd
from datetime import datetime
from datasets import load_dataset
from tqdm import tqdm

# ── reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ── sizes we benchmark at ────────────────────────────────────────────────────
SIZES = [1000, 5000, 10000]


# ─────────────────────────────────────────────────────────────────────────────
def _clean_text(text: str, max_words: int = 200) -> str:
    """Basic cleaning: strip whitespace, truncate to max_words."""
    if not isinstance(text, str):
        return ""
    tokens = text.split()
    return " ".join(tokens[:max_words])


# ─────────────────────────────────────────────────────────────────────────────
def load_cc_news(n: int = 10000, seed: int = SEED) -> pd.DataFrame:
    """
    Load n articles from CC-News (HuggingFace).
    Used ONLY for PDC speed benchmarks — no labels needed.

    Returns DataFrame with: text, title, timestamp
    """
    print(f"\n[DATA] Loading CC-News (n={n:,}) from HuggingFace...")
    t0 = time.time()

    ds = load_dataset("cc_news", split="train", trust_remote_code=True)

    # sample n rows reproducibly
    if len(ds) > n:
        indices = np.random.RandomState(seed).choice(len(ds), size=n, replace=False)
        ds = ds.select(indices.tolist())
    else:
        ds = ds.select(range(len(ds)))

    df = ds.to_pandas()

    # ── map columns ──────────────────────────────────────────────────────────
    # CC-News columns: title, text, domain, date, description, url, image_url
    text_col  = "text"   if "text"  in df.columns else df.columns[0]
    title_col = "title"  if "title" in df.columns else text_col
    date_col  = "date"   if "date"  in df.columns else None

    df["text"]  = df[text_col].apply(_clean_text)
    df["title"] = df[title_col].apply(lambda x: str(x)[:200] if isinstance(x, str) else "")

    # ── build timestamp ───────────────────────────────────────────────────────
    if date_col and date_col in df.columns:
        def parse_ts(d):
            try:
                if isinstance(d, (int, float)):
                    return float(d)
                return datetime.fromisoformat(str(d)[:10]).timestamp()
            except Exception:
                return float(datetime(2021, 1, 1).timestamp())
        df["timestamp"] = df[date_col].apply(parse_ts)
    else:
        # fallback: spread timestamps uniformly over 1 year
        base = datetime(2021, 1, 1).timestamp()
        df["timestamp"] = np.linspace(base, base + 365 * 86400, len(df))

    df["label"] = -1   # no ground-truth for CC-News
    result = df[["text", "title", "timestamp", "label"]].copy()
    result = result[result["text"].str.len() > 50].reset_index(drop=True)

    print(f"[DATA] CC-News loaded: {len(result):,} articles  ({time.time()-t0:.1f}s)")
    return result


# ─────────────────────────────────────────────────────────────────────────────
def load_all_the_news(n: int = 10000, n_topics: int = 20, seed: int = SEED) -> pd.DataFrame:
    """
    Load labeled news dataset for NLP evaluation (NMI/ARI/ROUGE).

    Priority:
      1. Local Kaggle CSV  (articles1.csv)
      2. AG News from HuggingFace  — 120k articles, 4 clean topics (best option)
      3. Labeled proxy from CC-News (last resort)

    Returns DataFrame with: text, title, timestamp, label (0..n_topics-1)
    """
    print(f"\n[DATA] Loading All-the-News (n={n:,}, topics={n_topics}) ...")
    t0 = time.time()

    # ── 1. Try local Kaggle CSV first ────────────────────────────────────────
    kaggle_paths = [
        os.path.join(os.path.dirname(__file__), "data", "articles1.csv"),
        os.path.join(os.path.expanduser("~"), "Documents", "NLP+PDC_Project", "data", "articles1.csv"),
    ]
    kaggle_csv = next((p for p in kaggle_paths if os.path.exists(p)), None)

    if kaggle_csv:
        print(f"[DATA] Loading real All-the-News CSV from {kaggle_csv} ...")
        df = pd.read_csv(kaggle_csv, usecols=lambda c: c in
                         ["content", "title", "section", "publication", "date", "url"])
        print(f"[DATA] Loaded {len(df):,} articles from Kaggle CSV.")
    else:
        # ── 2. AG News (HuggingFace) — clean 4-topic dataset ─────────────────
        try:
            print("[DATA] Loading AG News from HuggingFace (120k articles, 4 clean topics)...")
            ds = load_dataset("ag_news", split="train", trust_remote_code=True)
            df = ds.to_pandas()
            # AG News label map: 0=World, 1=Sports, 2=Business, 3=Sci/Tech
            label_names = {0: "world", 1: "sports", 2: "business", 3: "scitech"}
            df["section"] = df["label"].map(label_names)
            df["title"]   = df["text"].str[:80]   # use first 80 chars as title
            df["content"] = df["text"]
            df["date"]    = None
            n_topics = 4   # AG News has exactly 4 topics
            print(f"[DATA] AG News loaded: {len(df):,} articles, 4 topics.")
        except Exception as e:
            # ── 3. Last resort: keyword proxy ─────────────────────────────────
            print(f"[DATA] AG News unavailable ({e}) — building labeled proxy from CC-News...")
            df = _build_labeled_proxy(n=max(n * 3, 30000), n_topics=n_topics, seed=seed)

    # ── normalise columns ─────────────────────────────────────────────────────
    text_col    = next((c for c in ["content", "text", "article", "body"] if c in df.columns), df.columns[0])
    title_col   = next((c for c in ["title", "headline"]                  if c in df.columns), text_col)
    section_col = next((c for c in ["section", "topic", "category"]       if c in df.columns), None)
    date_col    = next((c for c in ["date", "published", "timestamp"]     if c in df.columns), None)

    df["text"]  = df[text_col].apply(_clean_text)
    df["title"] = df[title_col].apply(lambda x: str(x)[:200] if isinstance(x, str) else "")

    # ── timestamps ────────────────────────────────────────────────────────────
    if date_col and df[date_col].notna().any():
        def parse_ts(d):
            try:
                if isinstance(d, (int, float)):
                    return float(d)
                return datetime.fromisoformat(str(d)[:10]).timestamp()
            except Exception:
                return float(datetime(2017, 1, 1).timestamp())
        df["timestamp"] = df[date_col].apply(parse_ts)
    else:
        base = datetime(2017, 1, 1).timestamp()
        df["timestamp"] = np.linspace(base, base + 365 * 86400, len(df))

    # ── labels ────────────────────────────────────────────────────────────────
    if section_col and section_col in df.columns and "label" not in df.columns:
        top_sections = df[section_col].value_counts().head(n_topics).index.tolist()
        df = df[df[section_col].isin(top_sections)].copy()
        label_map = {s: i for i, s in enumerate(top_sections)}
        df["label"] = df[section_col].map(label_map)
    elif "label" not in df.columns:
        df["label"] = 0

    # ── keep n_topics categories, sample n rows ───────────────────────────────
    df = df[df["text"].str.len() > 50].dropna(subset=["label"]).copy()
    unique_labels = df["label"].nunique()
    actual_topics = min(n_topics, unique_labels)

    top_labels = df["label"].value_counts().head(actual_topics).index.tolist()
    df = df[df["label"].isin(top_labels)].copy()
    label_remap = {old: new for new, old in enumerate(top_labels)}
    df["label"] = df["label"].map(label_remap)

    # balanced sample: equal per class
    per_class = n // actual_topics
    sampled = (
        df.groupby("label", group_keys=False)
          .apply(lambda g: g.sample(min(len(g), per_class), random_state=seed))
    )
    sampled = sampled.sample(min(n, len(sampled)), random_state=seed).reset_index(drop=True)

    result = sampled[["text", "title", "timestamp", "label"]].copy()
    print(f"[DATA] All-the-News: {len(result):,} articles, {result['label'].nunique()} topics  ({time.time()-t0:.1f}s)")
    return result



# ─────────────────────────────────────────────────────────────────────────────
def _build_labeled_proxy(n: int = 30000, n_topics: int = 20, seed: int = SEED) -> pd.DataFrame:
    """
    Fallback: load CC-News and assign topic labels by keyword matching
    so we have a labeled dataset for NMI/ARI evaluation even without Kaggle.
    """
    TOPIC_KEYWORDS = {
        "politics":       ["president", "congress", "senate", "election", "vote", "government", "democrat", "republican"],
        "sports":         ["game", "player", "team", "score", "championship", "season", "coach", "league"],
        "technology":     ["ai", "tech", "software", "data", "cyber", "robot", "startup", "silicon"],
        "health":         ["vaccine", "hospital", "disease", "covid", "cancer", "health", "drug", "medical"],
        "business":       ["market", "stock", "economy", "bank", "trade", "company", "profit", "billion"],
        "entertainment":  ["movie", "music", "celebrity", "film", "actor", "award", "netflix", "album"],
        "science":        ["climate", "space", "research", "scientist", "nasa", "study", "experiment"],
        "crime":          ["police", "court", "arrest", "murder", "crime", "trial", "prison", "shooting"],
        "education":      ["school", "university", "student", "teacher", "class", "degree", "college"],
        "environment":    ["carbon", "emissions", "wildfire", "drought", "pollution", "renewable", "forest"],
        "international":  ["china", "russia", "ukraine", "europe", "nato", "war", "treaty", "sanctions"],
        "immigration":    ["border", "immigrant", "asylum", "migration", "refugee", "visa", "deportation"],
        "religion":       ["church", "faith", "prayer", "mosque", "christian", "muslim", "jewish", "bishop"],
        "finance":        ["interest", "inflation", "federal reserve", "mortgage", "budget", "tax", "debt"],
        "transport":      ["airline", "flight", "car", "rail", "highway", "accident", "tesla", "vehicle"],
        "food":           ["restaurant", "food", "diet", "nutrition", "cooking", "recipe", "organic"],
        "weather":        ["hurricane", "storm", "tornado", "flood", "earthquake", "snow", "temperature"],
        "legal":          ["lawsuit", "attorney", "judge", "supreme court", "settlement", "verdict"],
        "housing":        ["housing", "rent", "mortgage", "real estate", "apartment", "property"],
        "labor":          ["worker", "union", "strike", "wage", "unemployment", "job", "salary"],
    }
    TOPIC_KEYWORDS = dict(list(TOPIC_KEYWORDS.items())[:n_topics])

    print("[DATA] Building labeled proxy dataset from CC-News keywords...")
    ds = load_dataset("cc_news", split="train", trust_remote_code=True)
    indices = np.random.RandomState(seed).choice(len(ds), size=min(n, len(ds)), replace=False)
    df = ds.select(indices.tolist()).to_pandas()

    text_col = "text" if "text" in df.columns else df.columns[0]
    df["text"] = df[text_col].fillna("").str.lower()
    df["title"] = df["title"].fillna("") if "title" in df.columns else df["text"].str[:100]

    def assign_label(text):
        for topic, kws in TOPIC_KEYWORDS.items():
            if any(kw in text for kw in kws):
                return topic
        return None

    df["section"] = df["text"].apply(assign_label)
    df = df[df["section"].notna()].copy()

    # restore original case for text
    df["text"] = df[text_col].fillna("")
    return df


# ─────────────────────────────────────────────────────────────────────────────
def prepare_subsets(df: pd.DataFrame, sizes: list = SIZES, seed: int = SEED) -> dict:
    """
    Given a full DataFrame, return a dict {size: DataFrame} for each size in sizes.
    Ensures each subset is a random sample (with fixed seed).
    """
    subsets = {}
    for n in sizes:
        if n <= len(df):
            subsets[n] = df.sample(n=n, random_state=seed).reset_index(drop=True)
        else:
            subsets[n] = df.reset_index(drop=True)
            print(f"[DATA] Warning: requested {n:,} but only {len(df):,} available.")
    return subsets


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # quick smoke test
    print("=" * 60)
    print("SMOKE TEST — data_loader.py")
    print("=" * 60)

    cc = load_cc_news(n=1000)
    print(f"\nCC-News sample:\n{cc[['title','timestamp','label']].head(3)}")

    atn = load_all_the_news(n=1000, n_topics=10)
    print(f"\nAll-the-News sample:\n{atn[['title','timestamp','label']].head(3)}")
    print(f"\nLabel distribution:\n{atn['label'].value_counts().head(10)}")

    subsets = prepare_subsets(atn, sizes=[500, 1000])
    print(f"\nSubset sizes: { {k: len(v) for k,v in subsets.items()} }")
    print("\n[OK] data_loader.py passed smoke test.")
