"use client";
import { useState, useEffect } from "react";
import { useSearchParams } from "next/navigation";
import { api } from "@/lib/api";

export default function ClustersPage() {
  const searchParams  = useSearchParams();
  const highlightId   = searchParams.get("highlight");

  const [clusters, setClusters] = useState<any[]>([]);
  const [selected, setSelected] = useState<any>(null);
  const [loading,  setLoading]  = useState(true);
  const [running,  setRunning]  = useState(false);

  // Load cluster list on mount
  useEffect(() => {
    api.clusters.list(50)
      .then((c: any) => {
        setClusters(c);
        // Auto-select highlighted cluster
        if (highlightId) {
          const found = c.find((cl: any) => String(cl.id) === highlightId);
          if (found) loadCluster(found.id);
        }
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [highlightId]);

  const loadCluster = async (id: number) => {
    try {
      const detail = await api.clusters.get(id);
      setSelected(detail);
    } catch { /* silent */ }
  };

  const runPipeline = async () => {
    setRunning(true);
    try {
      await api.clusters.run(1000);
      // Poll for results after pipeline completes
      setTimeout(async () => {
        const c = await api.clusters.list(50);
        setClusters(c);
        setRunning(false);
      }, 35000);
    } catch {
      setRunning(false);
      alert("Pipeline failed — make sure the API is running.");
    }
  };

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-gray-800 pb-5">
        <div>
          <h1 className="text-2xl font-extrabold text-white">Storyline Explorer</h1>
          <p className="text-sm text-gray-400 mt-1">
            News articles grouped into event clusters — each cluster is one unfolding story.
          </p>
        </div>
        <button
          onClick={runPipeline} disabled={running}
          className="flex items-center gap-2 px-5 py-2.5 bg-indigo-600 hover:bg-indigo-500 disabled:bg-gray-800 disabled:text-gray-500 text-white rounded-xl text-sm font-semibold transition-all cursor-pointer"
        >
          {running
            ? <><span className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" /> Building clusters...</>
            : "⚡ Run Pipeline"}
        </button>
      </div>

      {running && (
        <div className="p-4 bg-indigo-500/10 border border-indigo-500/20 text-indigo-300 text-sm rounded-xl font-medium">
          🔄 Pipeline is running — embedding articles, clustering into events, and generating AI summaries.
          This takes ~30 seconds. The page will refresh automatically.
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        {/* Cluster List */}
        <div className="lg:col-span-1 glass-panel rounded-2xl overflow-hidden border border-gray-800/60">
          <div className="px-4 py-3 border-b border-gray-800 bg-gray-900/30">
            <p className="text-sm font-bold text-gray-200">
              Event Clusters
              {clusters.length > 0 && (
                <span className="ml-2 text-xs font-normal text-gray-500">({clusters.length} groups)</span>
              )}
            </p>
          </div>

          {loading ? (
            <div className="p-6 text-center text-gray-500 text-sm">Loading...</div>
          ) : clusters.length === 0 ? (
            <div className="p-8 text-center text-gray-500 text-sm space-y-3">
              <div className="text-3xl">🗞️</div>
              <p className="font-medium">No event clusters yet.</p>
              <p className="text-xs text-gray-600">Click "⚡ Run Pipeline" to embed and cluster your articles into storylines.</p>
            </div>
          ) : (
            <div className="divide-y divide-gray-800/60 max-h-[680px] overflow-y-auto">
              {clusters.map(c => {
                const isHighlighted = String(c.id) === highlightId;
                const isSelected    = selected?.id === c.id;
                return (
                  <button
                    key={c.id}
                    onClick={() => loadCluster(c.id)}
                    className={`w-full text-left px-4 py-3.5 hover:bg-gray-800/40 transition-colors relative ${
                      isSelected ? "bg-indigo-500/10 border-r-2 border-indigo-500" : ""
                    }`}
                  >
                    {isHighlighted && !isSelected && (
                      <span className="absolute top-2 right-2 w-2 h-2 bg-indigo-400 rounded-full animate-pulse" />
                    )}
                    <p className="text-sm font-semibold text-gray-200 line-clamp-2 leading-snug">
                      {c.label || `Cluster #${c.id}`}
                    </p>
                    <div className="flex items-center gap-2 mt-1">
                      <span className="text-xs text-indigo-400 font-mono font-bold">{c.size} articles</span>
                      {isHighlighted && (
                        <span className="text-[10px] bg-indigo-500/20 text-indigo-400 px-1.5 py-0.5 rounded font-bold uppercase tracking-wide">
                          Your Article
                        </span>
                      )}
                    </div>
                  </button>
                );
              })}
            </div>
          )}
        </div>

        {/* Cluster Detail */}
        <div className="lg:col-span-2 space-y-4">
          {selected ? (
            <>
              {/* Cluster Summary Card */}
              <div className="glass-panel rounded-2xl p-6 space-y-4 border border-gray-800/60">
                <div>
                  <div className="flex items-start justify-between gap-3">
                    <h2 className="text-xl font-extrabold text-white leading-snug">
                      {selected.label || `Cluster #${selected.id}`}
                    </h2>
                    <div className="flex-shrink-0 bg-indigo-500/10 border border-indigo-500/25 rounded-xl px-3 py-1.5 text-center">
                      <p className="text-2xl font-black text-indigo-400">{selected.size}</p>
                      <p className="text-[10px] text-gray-500 font-bold uppercase tracking-wide">Articles</p>
                    </div>
                  </div>
                  <p className="text-xs text-gray-500 mt-1">
                    Last updated: {new Date(selected.updated_at).toLocaleString()}
                  </p>
                </div>

                <div className="p-4 bg-gray-900/50 border border-gray-800 rounded-xl">
                  <p className="text-xs font-bold text-indigo-400 mb-2 uppercase tracking-wide">
                    🤖 AI-Generated Storyline Summary
                  </p>
                  <p className="text-sm text-gray-200 leading-relaxed">
                    {selected.summary || "Summary not yet generated — run the pipeline to create summaries."}
                  </p>
                </div>

                <div className="p-4 bg-gray-900/30 border border-gray-800 rounded-xl">
                  <p className="text-xs font-bold text-gray-400 uppercase tracking-wide mb-2">
                    What this cluster means
                  </p>
                  <p className="text-sm text-gray-400 leading-relaxed">
                    These {selected.size} articles were grouped together because they cover the same
                    unfolding news event, based on semantic (meaning) and temporal (timing) similarity.
                    The AI summary above combines their key points into one storyline.
                    Any stock signals generated from articles in this cluster share the same underlying news context.
                  </p>
                </div>
              </div>

              {/* Top Articles */}
              {selected.articles?.length > 0 && (
                <div className="glass-panel rounded-2xl overflow-hidden border border-gray-800/60">
                  <div className="px-5 py-3.5 border-b border-gray-800 bg-gray-900/30">
                    <p className="text-sm font-bold text-gray-200">
                      Articles in This Cluster
                      <span className="ml-2 text-xs font-normal text-gray-500">({selected.articles.length} total)</span>
                    </p>
                  </div>
                  <div className="divide-y divide-gray-800/60 max-h-[400px] overflow-y-auto">
                    {selected.articles.map((a: any) => (
                      <div key={a.id} className="px-5 py-4 hover:bg-gray-900/30 transition-colors">
                        <p className="text-sm font-medium text-gray-200 leading-snug">{a.title}</p>
                        <div className="flex items-center gap-3 text-xs text-gray-500 mt-1.5">
                          {a.source && <span className="text-gray-600 font-semibold">{a.source}</span>}
                          {a.published && (
                            <span>• {new Date(a.published).toLocaleDateString()}</span>
                          )}
                          {a.url && (
                            <a href={a.url} target="_blank" rel="noreferrer"
                               className="text-indigo-400 hover:text-indigo-300 hover:underline font-semibold transition-colors">
                              → Read article
                            </a>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="glass-panel rounded-2xl p-16 text-center border border-gray-800/60 space-y-3">
              <div className="text-4xl">🗞️</div>
              <p className="text-gray-400 font-semibold">Select a cluster to view its storyline</p>
              <p className="text-sm text-gray-600 max-w-sm mx-auto">
                Each cluster is a group of news articles about the same event. Click one on the left to see the AI summary and all the articles inside it.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
