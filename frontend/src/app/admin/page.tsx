"use client";
import { useState, useEffect, useCallback } from "react";
import { api } from "@/lib/api";

// ── Small helpers ──────────────────────────────────────────────────────────────
function StatCard({ label, value, sub, color = "indigo" }: {
  label: string; value: string | number | null; sub?: string; color?: string
}) {
  const clr: Record<string, string> = {
    indigo: "text-indigo-400",
    emerald: "text-emerald-400",
    rose:    "text-rose-400",
    amber:   "text-amber-400",
    gray:    "text-gray-400",
  };
  return (
    <div className="glass-panel p-5 rounded-2xl border border-gray-800/60 space-y-1">
      <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider">{label}</p>
      <p className={`text-3xl font-black ${clr[color] ?? clr.indigo}`}>
        {value === null || value === undefined ? "—" : value}
      </p>
      {sub && <p className="text-xs text-gray-500">{sub}</p>}
    </div>
  );
}

function PhaseTag({ phase }: { phase: string }) {
  const isReal = phase.toLowerCase().includes("phase 1");
  return (
    <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-black tracking-widest border uppercase ${
      isReal
        ? "bg-emerald-500/15 text-emerald-400 border-emerald-500/30"
        : "bg-amber-500/15 text-amber-400 border-amber-500/30"
    }`}>
      {isReal ? "✅" : "⚠️"} {phase}
    </span>
  );
}

// ── Main Admin Page ────────────────────────────────────────────────────────────
export default function AdminPage() {
  const [status,       setStatus]       = useState<any>(null);
  const [accuracy,     setAccuracy]     = useState<any>(null);
  const [loading,      setLoading]      = useState(true);
  const [labelLoading, setLabelLoading] = useState(false);
  const [trainLoading, setTrainLoading] = useState(false);
  const [message,      setMessage]      = useState("");

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const [s, a] = await Promise.all([
        fetch("/api/admin/status",   { headers: { Authorization: `Bearer ${localStorage.getItem("token")}` } }).then(r => r.json()),
        fetch("/api/admin/accuracy", { headers: { Authorization: `Bearer ${localStorage.getItem("token")}` } }).then(r => r.json()),
      ]);
      setStatus(s);
      setAccuracy(a);
    } catch { /* silent */ }
    finally { setLoading(false); }
  }, []);

  useEffect(() => { load(); }, [load]);

  const handleLabel = async () => {
    setLabelLoading(true); setMessage("");
    try {
      const r = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/admin/label`, {
        method: "POST",
        headers: { Authorization: `Bearer ${localStorage.getItem("token")}` },
      });
      const d = await r.json();
      setMessage(`✅ Labeling complete — labeled: ${d.labeled}, skipped: ${d.skipped}`);
      load();
    } catch { setMessage("❌ Labeling failed."); }
    finally { setLabelLoading(false); }
  };

  const handleRetrain = async () => {
    setTrainLoading(true); setMessage("");
    try {
      const r = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/admin/retrain`, {
        method: "POST",
        headers: { Authorization: `Bearer ${localStorage.getItem("token")}` },
      });
      const d = await r.json();
      if (d.status === "success") {
        setMessage(`✅ Model retrained! Val accuracy: ${(d.metrics?.val_accuracy * 100).toFixed(1)}%  F1: ${(d.metrics?.val_f1_macro).toFixed(3)}`);
      } else {
        setMessage(`⚠️ ${d.reason}`);
      }
      load();
    } catch { setMessage("❌ Retraining failed."); }
    finally { setTrainLoading(false); }
  };

  if (loading) return (
    <div className="flex items-center justify-center min-h-[60vh]">
      <span className="w-8 h-8 border-2 border-indigo-400 border-t-transparent rounded-full animate-spin" />
    </div>
  );

  const pct = (v: number | null) => v != null ? `${(v * 100).toFixed(1)}%` : "—";

  return (
    <div className="space-y-8 max-w-6xl mx-auto animate-slide-up">
      {/* Header */}
      <div className="flex items-start justify-between border-b border-gray-800 pb-5">
        <div>
          <h1 className="text-3xl font-extrabold tracking-tight text-white">Model Control Centre</h1>
          <p className="text-sm text-gray-400 mt-1">
            Phase 0 → Phase 1 upgrade pipeline — real-data labeling, retraining, and live accuracy tracking
          </p>
        </div>
        <button onClick={load}
          className="px-4 py-2 bg-gray-900 hover:bg-gray-800 border border-gray-800 rounded-xl text-sm font-semibold text-gray-300 transition-all cursor-pointer">
          🔄 Refresh
        </button>
      </div>

      {/* Status Banner */}
      {status && (
        <div className="glass-panel p-5 rounded-2xl border border-gray-800/60 flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="space-y-2">
            <p className="text-xs text-gray-400 uppercase tracking-wider font-semibold">Current Model Phase</p>
            <PhaseTag phase={status.model_phase} />
            {status.trained_at && (
              <p className="text-xs text-gray-500">Last trained: {new Date(status.trained_at).toLocaleString()}</p>
            )}
          </div>
          <div className="flex items-center gap-3">
            {!status.ready_for_retrain ? (
              <div className="p-3 bg-amber-500/10 border border-amber-500/20 rounded-xl text-xs text-amber-400 font-medium max-w-xs">
                ⚠️ Need {status.min_samples_needed - status.labeled_signals} more labeled signals before retraining is possible.
                Use &ldquo;Label Outcomes&rdquo; below to start.
              </div>
            ) : (
              <div className="p-3 bg-emerald-500/10 border border-emerald-500/20 rounded-xl text-xs text-emerald-400 font-medium max-w-xs">
                ✅ {status.labeled_signals} labeled signals ready. You can retrain now!
              </div>
            )}
          </div>
        </div>
      )}

      {/* Stats Grid */}
      {status && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <StatCard label="Total Signals"    value={status.total_signals}    sub="all users, all time"       color="indigo" />
          <StatCard label="Labeled"          value={status.labeled_signals}  sub="real T+3 outcomes fetched" color="emerald" />
          <StatCard label="Unlabeled"        value={status.unlabeled_signals} sub="need labeling"            color="amber" />
          <StatCard label="Live Accuracy"    value={status.live_prediction_accuracy != null ? pct(status.live_prediction_accuracy) : "—"}
                    sub="predicted vs actual"  color={status.live_prediction_accuracy > 0.55 ? "emerald" : "amber"} />
        </div>
      )}

      {/* Training Metrics */}
      {status && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <StatCard label="Val Accuracy"  value={status.val_accuracy  != null ? pct(status.val_accuracy)  : "N/A (Phase 0)"} color="indigo" />
          <StatCard label="Val F1 Macro"  value={status.val_f1_macro  != null ? status.val_f1_macro.toFixed(3) : "N/A"}       color="indigo" />
          <StatCard label="Training Data" value={status.training_samples} sub="samples used in last fit"   color="gray" />
          <StatCard label="Model Size"    value={`${status.model_file_kb} KB`} sub="XGBoost pickle"        color="gray" />
        </div>
      )}

      {/* Actions */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        {/* Label Outcomes */}
        <div className="glass-panel p-6 rounded-2xl border border-gray-800/60 space-y-4">
          <div>
            <h2 className="text-lg font-bold text-white">Step 1 — Label Outcomes</h2>
            <p className="text-xs text-gray-400 mt-1">
              For each signal older than 3 days, fetch the real T+3 stock price from Yahoo Finance and
              write the actual direction (up / flat / down) and return % to the database.
              This creates the real training dataset for Phase 1.
            </p>
          </div>
          <div className="p-3 bg-gray-900/50 border border-gray-800 rounded-xl text-xs text-gray-400 space-y-1 font-mono">
            <p>⏱ Runtime: ~1–2 min for 100 signals</p>
            <p>📊 Schedule: auto-runs every 6 hours</p>
            <p>🔄 Safe to run multiple times (idempotent)</p>
          </div>
          <button onClick={handleLabel} disabled={labelLoading}
            className="w-full py-3 bg-indigo-600 hover:bg-indigo-500 disabled:bg-gray-800 disabled:text-gray-500 text-white rounded-xl text-sm font-semibold transition-all cursor-pointer flex items-center justify-center gap-2">
            {labelLoading
              ? <><span className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" /> Labeling...</>
              : "🏷️ Label Outcomes Now"}
          </button>
        </div>

        {/* Retrain */}
        <div className="glass-panel p-6 rounded-2xl border border-gray-800/60 space-y-4">
          <div>
            <h2 className="text-lg font-bold text-white">Step 2 — Retrain Model</h2>
            <p className="text-xs text-gray-400 mt-1">
              Uses all labeled signals to retrain XGBoost with TimeSeriesSplit cross-validation.
              Requires at least {status?.min_samples_needed ?? 30} labeled samples.
              The new model is hot-swapped live — no server restart needed.
            </p>
          </div>
          <div className="p-3 bg-gray-900/50 border border-gray-800 rounded-xl text-xs text-gray-400 space-y-1 font-mono">
            <p>⏱ Runtime: ~30–60 seconds</p>
            <p>📊 Schedule: auto-runs every 24 hours</p>
            <p>🔄 Model hot-swaps without server restart</p>
          </div>
          <button onClick={handleRetrain} disabled={trainLoading || !status?.ready_for_retrain}
            className="w-full py-3 bg-emerald-600 hover:bg-emerald-500 disabled:bg-gray-800 disabled:text-gray-500 text-white rounded-xl text-sm font-semibold transition-all cursor-pointer flex items-center justify-center gap-2">
            {trainLoading
              ? <><span className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" /> Training...</>
              : "🧠 Retrain on Real Data"}
          </button>
        </div>
      </div>

      {/* Message */}
      {message && (
        <div className={`p-4 rounded-xl text-sm font-medium border ${
          message.startsWith("✅")
            ? "bg-emerald-500/10 border-emerald-500/20 text-emerald-400"
            : message.startsWith("⚠️")
            ? "bg-amber-500/10 border-amber-500/20 text-amber-400"
            : "bg-rose-500/10 border-rose-500/20 text-rose-400"
        }`}>
          {message}
        </div>
      )}

      {/* Accuracy Breakdown */}
      {accuracy && accuracy.total_labeled > 0 && (
        <div className="space-y-5">
          <div className="glass-panel rounded-2xl overflow-hidden border border-gray-800/60">
            <div className="px-6 py-4 border-b border-gray-800 bg-gray-900/30 flex items-center justify-between">
              <h2 className="text-md font-bold text-gray-200">Prediction Accuracy by Ticker</h2>
              <span className="text-xs text-gray-500">{accuracy.total_labeled} labeled signals total — overall: {pct(accuracy.overall_accuracy)}</span>
            </div>
            <div className="divide-y divide-gray-800/60">
              {Object.entries(accuracy.accuracy_by_ticker ?? {}).map(([ticker, data]: [string, any]) => (
                <div key={ticker} className="px-6 py-4 flex items-center justify-between gap-4">
                  <div className="flex items-center gap-3">
                    <span className="text-xs font-mono font-bold text-indigo-400 bg-indigo-500/10 px-2 py-0.5 rounded border border-indigo-500/20">
                      ${ticker}
                    </span>
                    <div className="w-32 h-2 bg-gray-900 rounded-full overflow-hidden border border-gray-800">
                      <div className="h-full rounded-full transition-all duration-700"
                        style={{ width: `${data.accuracy * 100}%`, backgroundColor: data.accuracy > 0.55 ? "#10b981" : data.accuracy > 0.45 ? "#f59e0b" : "#ef4444" }} />
                    </div>
                    <span className="text-sm font-bold text-gray-200">{pct(data.accuracy)}</span>
                  </div>
                  <div className="text-right text-xs text-gray-500 space-y-0.5">
                    <p>{data.correct}/{data.total} correct</p>
                    <p className={data.avg_return > 0 ? "text-emerald-400" : data.avg_return < 0 ? "text-rose-400" : "text-gray-400"}>
                      avg return: {data.avg_return > 0 ? "+" : ""}{(data.avg_return * 100).toFixed(2)}%
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Confusion Matrix */}
          <div className="glass-panel p-6 rounded-2xl border border-gray-800/60">
            <h2 className="text-md font-bold text-gray-200 mb-4">Confusion Matrix (Predicted → Actual)</h2>
            <div className="overflow-x-auto">
              <table className="w-full text-xs text-center">
                <thead>
                  <tr className="text-gray-500">
                    <th className="py-2 px-3 text-left">Predicted ↓ / Actual →</th>
                    <th className="py-2 px-3 text-emerald-400">↑ UP</th>
                    <th className="py-2 px-3 text-gray-400">— FLAT</th>
                    <th className="py-2 px-3 text-rose-400">↓ DOWN</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-800/40">
                  {Object.entries(accuracy.confusion_matrix ?? {}).map(([pred, row]: [string, any]) => (
                    <tr key={pred}>
                      <td className="py-2 px-3 text-left font-semibold text-gray-400">{pred.toUpperCase()}</td>
                      {["up", "flat", "down"].map(actual => (
                        <td key={actual} className={`py-2 px-3 font-mono font-bold ${
                          pred === actual ? "text-indigo-400" : "text-gray-600"
                        }`}>
                          {row[actual] ?? 0}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="text-xs text-gray-600 mt-3">Diagonal = correct predictions. Off-diagonal = errors.</p>
          </div>
        </div>
      )}
    </div>
  );
}
