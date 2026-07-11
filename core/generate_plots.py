"""
generate_plots.py
=================
Generates all figures and tables required for the Final Research Paper:

  Figure 1: Speedup Curve (S(K) vs K for N = 1k, 5k, 10k)
  Figure 2: Processing Time Bar Chart (sequential vs parallel)
  Figure 3: NMI Scores — Temporal-Semantic vs Semantic-Only (ablation)
  Figure 4: NMI Stability across dataset sizes
  Figure 5: ROUGE Scores Table (BART vs T5)
  Figure 6: Elbow Method plot (WCSS vs K)

All figures saved to results/plots/
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                # non-interactive backend for server
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

PLOTS_DIR   = os.path.join(os.path.dirname(__file__), "..", "results", "plots")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":    "Times New Roman",
    "font.size":      12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi":     150,
})
COLORS = {
    "pdc":  "#1A56CC",   # blue  — PDC
    "nlp":  "#1A7A3C",   # green — NLP
    "seq":  "#CC3333",   # red   — sequential baseline
    "ts":   "#1A56CC",   # temporal-semantic
    "sem":  "#888888",   # semantic-only
    "bart": "#1A56CC",
    "t5":   "#1A7A3C",
}


# ─────────────────────────────────────────────────────────────────────────────
def plot_speedup_curve(df_speedup: pd.DataFrame):
    """
    Figure 1: Speedup S(K) = T(1)/T(K) vs K for each N.
    Includes ideal linear speedup reference and target line.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    sizes  = sorted(df_speedup["N"].unique())
    k_vals = sorted(df_speedup["K"].unique())
    cmap   = plt.cm.Blues(np.linspace(0.5, 0.9, len(sizes)))

    for i, n in enumerate(sizes):
        subset = df_speedup[df_speedup["N"] == n].sort_values("K")
        ks     = subset["K"].values
        su     = subset["speedup"].values
        ax.plot(ks, su, "o-", color=cmap[i], linewidth=2,
                markersize=7, label=f"N = {n:,}")
        for k, s in zip(ks, su):
            ax.annotate(f"{s:.2f}×", (k, s),
                        textcoords="offset points", xytext=(5, 3),
                        fontsize=9, color=cmap[i])

    # ideal linear speedup
    ax.plot(k_vals, k_vals, "k--", linewidth=1.2, alpha=0.5, label="Ideal linear")

    # target line S(4) >= 1.67
    ax.axhline(y=1.67, color="red", linestyle=":", linewidth=1.5,
               label="Target: S(4) ≥ 1.67×")

    ax.set_xlabel("Number of Parallel Workers (K)")
    ax.set_ylabel("Speedup S(K) = T(1) / T(K)")
    ax.set_title("Figure 1: Speedup Curve — Parallel Sentence Embedding (PDC)")
    ax.set_xticks(k_vals)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0.8)

    path = os.path.join(PLOTS_DIR, "fig1_speedup_curve.pdf")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
def plot_processing_time(df_speedup: pd.DataFrame):
    """
    Figure 2: Processing time (seconds) grouped bar chart.
    Shows T(K) for K=1, 2, 4 side by side for each N.
    """
    sizes   = sorted(df_speedup["N"].unique())
    k_vals  = sorted(df_speedup["K"].unique())
    x       = np.arange(len(sizes))
    width   = 0.25
    colors  = [COLORS["seq"], "#5580CC", COLORS["pdc"]]

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, k in enumerate(k_vals):
        times = []
        for n in sizes:
            row = df_speedup[(df_speedup["N"] == n) & (df_speedup["K"] == k)]
            times.append(row["mean_time_s"].values[0] if len(row) else 0)
        bars = ax.bar(x + i * width, times, width, label=f"K={k}",
                      color=colors[i], alpha=0.85, edgecolor="white")
        for bar, t in zip(bars, times):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{t:.1f}s", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Dataset Size (N)")
    ax.set_ylabel("Mean Processing Time (seconds)")
    ax.set_title("Figure 2: Embedding Time — Sequential vs Parallel (PDC)")
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"{n:,}" for n in sizes])
    ax.legend(title="Workers (K)")
    ax.grid(True, axis="y", alpha=0.3)

    path = os.path.join(PLOTS_DIR, "fig2_processing_time.pdf")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
def plot_nmi_ablation(df_cluster: pd.DataFrame):
    """
    Figure 3: NMI Ablation — Temporal-Semantic vs Semantic-Only.
    Grouped bar chart across dataset sizes.
    """
    sizes  = sorted(df_cluster["N"].unique())
    x      = np.arange(len(sizes))
    width  = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))

    nmi_ts  = [df_cluster[df_cluster["N"] == n]["NMI_temporal"].values[0] for n in sizes]
    nmi_sem = [df_cluster[df_cluster["N"] == n]["NMI_semantic"].values[0] for n in sizes]

    b1 = ax.bar(x - width/2, nmi_ts,  width, label="Temporal-Semantic (proposed)",
                color=COLORS["ts"], alpha=0.85, edgecolor="white")
    b2 = ax.bar(x + width/2, nmi_sem, width, label="Semantic-Only (ablation)",
                color=COLORS["sem"], alpha=0.85, edgecolor="white")

    for bar, val in zip(list(b1) + list(b2), nmi_ts + nmi_sem):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    # target line
    ax.axhline(y=0.70, color="red", linestyle=":", linewidth=1.5,
               label="Target: NMI ≥ 0.70")
    # Nakshatri baseline (entity purity 82.69 → normalised ~0.72 equiv)
    ax.axhline(y=0.72, color="green", linestyle="--", linewidth=1.2, alpha=0.7,
               label="Nakshatri et al. (2023): 0.72")

    ax.set_xlabel("Dataset Size (N)")
    ax.set_ylabel("NMI Score")
    ax.set_title("Figure 3: Clustering Ablation — Temporal vs Semantic-Only (NLP)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n:,}" for n in sizes])
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)

    path = os.path.join(PLOTS_DIR, "fig3_nmi_ablation.pdf")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
def plot_nmi_stability(df_cluster: pd.DataFrame):
    """
    Figure 4: NMI stability across dataset sizes.
    Shows NMI does not degrade by more than 0.02 as N increases.
    """
    sizes   = sorted(df_cluster["N"].unique())
    nmi_ts  = [df_cluster[df_cluster["N"] == n]["NMI_temporal"].values[0] for n in sizes]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sizes, nmi_ts, "o-", color=COLORS["ts"], linewidth=2,
            markersize=8, label="NMI (temporal-semantic)")

    for n, v in zip(sizes, nmi_ts):
        ax.annotate(f"{v:.3f}", (n, v),
                    textcoords="offset points", xytext=(5, 5), fontsize=10)

    # tolerance band: ±0.02 from first value
    base = nmi_ts[0]
    ax.fill_between(sizes,
                    [base - 0.02] * len(sizes),
                    [base + 0.02] * len(sizes),
                    alpha=0.15, color=COLORS["ts"],
                    label="±0.02 tolerance band")

    ax.axhline(y=0.70, color="red", linestyle=":", linewidth=1.5, label="Target: NMI ≥ 0.70")

    ax.set_xlabel("Dataset Size (N)")
    ax.set_ylabel("NMI Score")
    ax.set_title("Figure 4: NMI Stability across Dataset Scales (NLP)")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    path = os.path.join(PLOTS_DIR, "fig4_nmi_stability.pdf")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
def plot_rouge_comparison(df_rouge: pd.DataFrame):
    """
    Figure 5: ROUGE scores comparison — BART-large-CNN vs baseline target.
    """
    sizes = sorted(df_rouge["N"].unique())
    x     = np.arange(len(sizes))
    width = 0.3

    r1 = [df_rouge[df_rouge["N"] == n]["ROUGE-1"].values[0] for n in sizes]
    r2 = [df_rouge[df_rouge["N"] == n]["ROUGE-2"].values[0] for n in sizes]
    rL = [df_rouge[df_rouge["N"] == n]["ROUGE-L"].values[0] for n in sizes]

    fig, ax = plt.subplots(figsize=(8, 5))

    b1 = ax.bar(x - width, r1, width, label="ROUGE-1", color="#4488CC", alpha=0.85)
    b2 = ax.bar(x,         r2, width, label="ROUGE-2", color=COLORS["bart"], alpha=0.85)
    b3 = ax.bar(x + width, rL, width, label="ROUGE-L", color="#88AADD", alpha=0.85)

    for bars, vals in [(b1, r1), (b2, r2), (b3, rL)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    # Zhang et al. 2025 ROUGE-2 = 30.88
    ax.axhline(y=30.88, color="orange", linestyle="--", linewidth=1.5,
               label="Zhang et al. (2025) ROUGE-2 = 30.88")
    ax.axhline(y=28.0, color="red", linestyle=":", linewidth=1.5,
               label="Target: ROUGE-2 ≥ 28")

    ax.set_xlabel("Dataset Size (N)")
    ax.set_ylabel("ROUGE Score (F1 × 100)")
    ax.set_title("Figure 5: Summarization Quality — ROUGE Scores (NLP)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n:,}" for n in sizes])
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    path = os.path.join(PLOTS_DIR, "fig5_rouge_comparison.pdf")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
def plot_elbow(k_values: list, wcss_list: list, best_k: int, n: int):
    """
    Figure 6: Elbow Method — WCSS vs K for one dataset size.
    """
    if not k_values or not wcss_list:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(k_values, wcss_list, "o-", color=COLORS["nlp"], linewidth=2, markersize=6)
    ax.axvline(x=best_k, color="red", linestyle="--", linewidth=1.5,
               label=f"Selected K = {best_k}")
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Within-Cluster Sum of Squares (WCSS)")
    ax.set_title(f"Figure 6: Elbow Method — Optimal K Selection (N={n:,})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = os.path.join(PLOTS_DIR, f"fig6_elbow_N{n}.pdf")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
def generate_all_plots(speedup_csv: str = None,
                       cluster_csv: str = None,
                       rouge_csv: str   = None,
                       clustering_results: dict = None):
    """
    Master function: load CSVs and generate all 6 figures.
    """
    print("\n" + "=" * 60)
    print("GENERATING ALL PAPER FIGURES")
    print("=" * 60)

    # ── Figure 1 & 2: Speedup ────────────────────────────────────────────────
    sp_path = speedup_csv or os.path.join(RESULTS_DIR, "speedup_results.csv")
    if os.path.exists(sp_path):
        df_sp = pd.read_csv(sp_path)
        plot_speedup_curve(df_sp)
        plot_processing_time(df_sp)
    else:
        print(f"[SKIP] {sp_path} not found — skipping speedup plots")

    # ── Figure 3 & 4: Clustering ─────────────────────────────────────────────
    cl_path = cluster_csv or os.path.join(RESULTS_DIR, "clustering_results.csv")
    if os.path.exists(cl_path):
        df_cl = pd.read_csv(cl_path)
        plot_nmi_ablation(df_cl)
        plot_nmi_stability(df_cl)
    else:
        print(f"[SKIP] {cl_path} not found — skipping clustering plots")

    # ── Figure 5: ROUGE ──────────────────────────────────────────────────────
    rg_path = rouge_csv or os.path.join(RESULTS_DIR, "rouge_results.csv")
    if os.path.exists(rg_path):
        df_rg = pd.read_csv(rg_path)
        plot_rouge_comparison(df_rg)
    else:
        print(f"[SKIP] {rg_path} not found — skipping ROUGE plot")

    # ── Figure 6: Elbow plots (from in-memory results) ───────────────────────
    if clustering_results:
        for n, res in clustering_results.items():
            if res.get("k_values") and res.get("wcss_list"):
                plot_elbow(res["k_values"], res["wcss_list"], res["K"], n)

    print(f"\n[DONE] All plots saved to: {PLOTS_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    generate_all_plots()
