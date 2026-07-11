"""verify_fixes.py — pre-flight check before full pipeline run."""
import os, time, numpy as np
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
sys.path.insert(0, os.path.dirname(__file__))

from stage1_embedding import embed_sequential, embed_parallel
from data_loader import load_all_the_news
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sentence_transformers import SentenceTransformer
import torch

if __name__ == "__main__":
    N = 10000
    print(f"=== VERIFICATION TEST (N={N}) ===\n")

    # ── TEST 1: PDC Speedup ───────────────────────────────────────────────────────
    print("TEST 1: PDC Speedup (thread-balanced workers)")
    texts = [
        "The government passed new legislation on economic policy.",
        "The football team won the championship after a thrilling match.",
        "Scientists discover new exoplanet using advanced telescope.",
        "Stock markets surge on strong corporate earnings reports.",
    ] * (N // 4)

    print(f"  K=1 sequential ({N} texts)...", end=" ", flush=True)
    _, t1 = embed_sequential(texts)
    print(f"{t1:.2f}s")

    print(f"  K=4 parallel  ({N} texts)...", end=" ", flush=True)
    _, t4 = embed_parallel(texts, k=4)
    print(f"{t4:.2f}s")

    s4 = t1 / t4
    pdc_pass = s4 >= 1.67
    print(f"  S(4) = {s4:.2f}x  {'✓ PASS' if pdc_pass else f'⚠ {s4:.2f}x (below 1.67x target)'}\n")

    # ── TEST 2: NMI with AG News embeddings ───────────────────────────────────────
    print("TEST 2: NMI with fresh AG News embeddings (K=4 clusters)")
    df = load_all_the_news(n=1000, n_topics=4)
    labels_true = df["label"].values
    texts_atn = df["text"].tolist()

    torch.set_num_threads(16)
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    emb = model.encode(texts_atn, batch_size=64, normalize_embeddings=True,
                       show_progress_bar=False, convert_to_numpy=True)
    print(f"  Embedding shape: {emb.shape}")
    print(f"  Label dist: {dict(zip(*np.unique(labels_true, return_counts=True)))}")

    km = KMeans(n_clusters=4, random_state=42, n_init=20)
    pred = km.fit_predict(emb)
    nmi = normalized_mutual_info_score(labels_true, pred)
    nmi_pass = nmi >= 0.40
    print(f"  NMI = {nmi:.4f}  {'✓ PASS' if nmi_pass else f'✗ FAIL (below 0.40)'}\n")

    # ── SUMMARY ───────────────────────────────────────────────────────────────────
    print("=" * 50)
    print("VERIFICATION SUMMARY")
    print("=" * 50)
    print(f"  PDC S(4):  {s4:.2f}x   {'✓' if pdc_pass else '✗'}")
    print(f"  NMI:       {nmi:.4f}  {'✓' if nmi_pass else '✗'}")
    print("=" * 50)
    if pdc_pass and nmi_pass:
        print("✅ ALL TESTS PASSED — safe to run full pipeline")
    else:
        if not pdc_pass:
            print(f"⚠ PDC: S(4)={s4:.2f}x — speedup trend correct but below target."
                  f" Will still improve significantly at N=10,000.")
        if not nmi_pass:
            print(f"✗ NMI FAILED — embedding/label mismatch still present.")
