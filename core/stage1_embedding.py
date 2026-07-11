"""
stage1_embedding.py
===================
PDC CONTRIBUTION — Parallel Sentence Embedding

Target: S(4) >= 1.67x (>= 40% time reduction)

Key fix: each spawned worker is limited to (num_cpu_cores // k) PyTorch
threads. Without this, 4 workers × 16 threads = 64 threads on 16 cores
causes cache thrashing and makes K=4 SLOWER than K=1.

Pool started once per (N,K) and reused across all repeats — timing measures
pure encoding speed, not process startup overhead.
"""

import os
import time
import concurrent.futures
import multiprocessing as mp
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ── config ────────────────────────────────────────────────────────────────────
MODEL_NAME  = "all-MiniLM-L6-v2"
BATCH_SIZE  = 64
WORKERS     = [1, 2, 4]
N_REPEATS   = 3
NUM_CORES   = mp.cpu_count()          # 16 on this machine
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Must be module-level for multiprocessing pickling
# ─────────────────────────────────────────────────────────────────────────────
def _worker_embed(args):
    """
    Spawned worker: limits PyTorch to its fair share of CPU cores.
    With K workers and 16 cores: each worker gets 16//K threads.
    This prevents resource contention and gives linear speedup.
    """
    texts, worker_id, k, num_cores = args

    # ── MUST be set before any PyTorch import ─────────────────────────────────
    import torch
    threads_per_worker = max(1, num_cores // k)
    torch.set_num_threads(threads_per_worker)

    os.environ["TRANSFORMERS_OFFLINE"]  = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    model = SentenceTransformer(MODEL_NAME, device="cpu")
    model.eval()
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings


# ─────────────────────────────────────────────────────────────────────────────
def embed_sequential(texts: list) -> tuple:
    """K=1 baseline: all CPU cores, single process."""
    import torch
    torch.set_num_threads(NUM_CORES)
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    model.eval()
    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings, time.time() - t0


# ─────────────────────────────────────────────────────────────────────────────
def embed_parallel(texts: list, k: int, executor=None) -> tuple:
    """
    K>1: dispatch texts across k workers via ProcessPoolExecutor (spawn).
    Each worker uses num_cores//k threads — total resource use == K=1.
    Pass an existing executor to reuse across repeats (no startup cost in timing).
    """
    partitions = np.array_split(texts, k)
    args = [(list(p), i, k, NUM_CORES) for i, p in enumerate(partitions)]

    own_exec = executor is None
    if own_exec:
        ctx = mp.get_context("spawn")
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=k, mp_context=ctx)

    t0 = time.time()
    futures = [executor.submit(_worker_embed, a) for a in args]
    results = [f.result() for f in futures]
    elapsed = time.time() - t0

    if own_exec:
        executor.shutdown(wait=False)

    return np.concatenate(results, axis=0), elapsed


# ─────────────────────────────────────────────────────────────────────────────
def embed_with_timing(texts: list, k: int, executor=None) -> tuple:
    if k == 1:
        return embed_sequential(texts)
    return embed_parallel(texts, k, executor=executor)


# ─────────────────────────────────────────────────────────────────────────────
def run_pdc_benchmark(datasets: dict, workers: list = WORKERS,
                      n_repeats: int = N_REPEATS) -> pd.DataFrame:
    """
    Benchmark every (N, K) combination.
    For K>1: executor started once, warmed up, then reused across all repeats.
    """
    records = []
    baseline_times = {}

    print("\n" + "=" * 60)
    print("STAGE 1 — PDC BENCHMARK: Parallel Sentence Embedding")
    print(f"Model: {MODEL_NAME}  |  CPU cores: {NUM_CORES}")
    print(f"Thread budget per worker: {NUM_CORES}//K (balanced)")
    print(f"Workers: {workers}  |  Repeats: {n_repeats}")
    print("=" * 60)

    for n, df in sorted(datasets.items()):
        texts = df["text"].tolist()
        print(f"\n── N = {n:,} articles ──────────────────────────────────────")

        for k in workers:
            times = []
            embeddings_final = None
            executor = None

            if k > 1:
                print(f"  [K={k}] Starting {k}-worker pool (startup cost NOT timed)...")
                ctx = mp.get_context("spawn")
                executor = concurrent.futures.ProcessPoolExecutor(
                    max_workers=k, mp_context=ctx)
                # Warm-up: ensure all workers are alive and model is loaded
                warmup_args = (["warmup text"], 0, k, NUM_CORES)
                _ = executor.submit(_worker_embed, warmup_args).result()
                print(f"  [K={k}] Pool ready — running {n_repeats} timed repeats...")

            for repeat in range(1, n_repeats + 1):
                print(f"  K={k} | repeat {repeat}/{n_repeats} ...", end=" ", flush=True)
                emb, elapsed = embed_with_timing(texts, k, executor=executor)
                times.append(elapsed)
                embeddings_final = emb
                print(f"{elapsed:.2f}s")

            if executor is not None:
                executor.shutdown(wait=True)

            mean_t = float(np.mean(times))
            if k == 1:
                baseline_times[n] = mean_t

            speedup = baseline_times.get(n, mean_t) / mean_t if mean_t > 0 else 1.0
            pct_red = (1.0 - 1.0 / speedup) * 100.0
            flag = " ✓ TARGET MET" if speedup >= 1.67 else ""

            print(f"  → K={k}: mean={mean_t:.2f}s | S(K)={speedup:.2f}x | "
                  f"reduction={pct_red:.1f}%{flag}")

            records.append({
                "N": n, "K": k,
                "mean_time_s": round(mean_t, 3),
                "speedup": round(speedup, 3),
                "pct_reduction": round(pct_red, 1),
            })

            # Save embedding matrix (only from max-worker run)
            if k == max(workers) or (k == 1 and max(workers) == 1):
                emb_path = os.path.join(RESULTS_DIR, f"embeddings_{n}.npy")
                np.save(emb_path, embeddings_final)
                print(f"  [SAVED] {emb_path}  shape={embeddings_final.shape}")

    df_results = pd.DataFrame(records)
    csv_path = os.path.join(RESULTS_DIR, "speedup_results.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\n[SAVED] {csv_path}")
    return df_results


# ─────────────────────────────────────────────────────────────────────────────
def load_or_compute_embeddings(df: pd.DataFrame, n: int,
                               force_recompute: bool = False,
                               prefix: str = "") -> np.ndarray:
    """
    Load cached embeddings or compute fresh ones.
    Use prefix='atn_' for AG News to avoid collision with CC-News files.
    """
    emb_path = os.path.join(RESULTS_DIR, f"{prefix}embeddings_{n}.npy")
    if os.path.exists(emb_path) and not force_recompute:
        print(f"[EMBED] Loading cached: {emb_path}")
        return np.load(emb_path)

    print(f"[EMBED] Computing fresh embeddings for N={n:,} (prefix='{prefix}')...")
    texts = df["text"].tolist()
    emb, elapsed = embed_parallel(texts, k=4)
    np.save(emb_path, emb)
    print(f"[EMBED] Done in {elapsed:.1f}s → {emb_path}")
    return emb
