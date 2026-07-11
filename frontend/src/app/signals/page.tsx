"use client";
import { useState, useEffect } from "react";
import { api } from "@/lib/api";

function DirectionBadge({ direction }: { direction: string }) {
  const cls = direction === "up" ? "badge-up" : direction === "down" ? "badge-down" : "badge-flat";
  return (
    <span className={`px-2 py-0.5 rounded-full text-xs font-bold uppercase ${cls}`}>
      {direction === "up" ? "▲" : direction === "down" ? "▼" : "—"} {direction}
    </span>
  );
}

export default function SignalsPage() {
  const [signals, setSignals]   = useState<any[]>([]);
  const [loading, setLoading]   = useState(true);
  const [filter,  setFilter]    = useState<string>("all");
  const [ticker,  setTicker]    = useState("");
  const [trading, setTrading]   = useState<Record<number, boolean>>({});
  const [tradeMsg, setTradeMsg] = useState<Record<number, string>>({});

  useEffect(() => {
    api.signals.list({ limit: 100 })
      .then((s: any) => setSignals(s))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  const filtered = signals.filter(s => {
    if (filter !== "all" && s.direction !== filter) return false;
    if (ticker && !s.ticker.includes(ticker.toUpperCase())) return false;
    return true;
  });

  const handleTrade = async (sig: any) => {
    setTrading(prev => ({...prev, [sig.id]: true}));
    try {
      await api.portfolio.trade(sig.id);
      setTradeMsg(prev => ({...prev, [sig.id]: "✅ Paper trade placed!"}));
    } catch (err: any) {
      setTradeMsg(prev => ({...prev, [sig.id]: `❌ ${err.message}`}));
    } finally {
      setTrading(prev => ({...prev, [sig.id]: false}));
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Signal Feed</h1>
        <p className="text-sm text-gray-400 mt-1">All generated financial signals — FinBERT + RAG + XGBoost</p>
      </div>

      {/* Filters */}
      <div className="flex gap-3 items-center">
        {["all", "up", "down", "flat"].map(d => (
          <button
            key={d}
            onClick={() => setFilter(d)}
            className={`px-4 py-1.5 rounded-xl text-sm font-medium transition-all
              ${filter === d
                ? "bg-indigo-600 text-white"
                : "bg-gray-800 text-gray-400 hover:bg-gray-700"}`}
          >
            {d === "all" ? "All" : d === "up" ? "▲ Bullish" : d === "down" ? "▼ Bearish" : "— Flat"}
          </button>
        ))}
        <input
          value={ticker}
          onChange={e => setTicker(e.target.value)}
          placeholder="Filter ticker..."
          className="ml-auto bg-gray-800 border border-gray-700 rounded-xl px-3 py-1.5 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-indigo-500 w-36"
        />
      </div>

      {/* Table */}
      <div className="glass rounded-2xl overflow-hidden">
        {loading ? (
          <div className="p-10 text-center text-gray-500">Loading signals...</div>
        ) : filtered.length === 0 ? (
          <div className="p-10 text-center text-gray-500">No signals match your filters.</div>
        ) : (
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-800 text-gray-400 text-xs uppercase">
                <th className="px-5 py-3 text-left">Ticker</th>
                <th className="px-5 py-3 text-left">Headline</th>
                <th className="px-5 py-3 text-left">Direction</th>
                <th className="px-5 py-3 text-right">Confidence</th>
                <th className="px-5 py-3 text-right">Impact</th>
                <th className="px-5 py-3 text-right">Quant</th>
                <th className="px-5 py-3 text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800">
              {filtered.map(sig => (
                <tr key={sig.id} className="hover:bg-gray-800/30 transition-colors">
                  <td className="px-5 py-3">
                    <span className="font-mono font-bold text-indigo-400">{sig.ticker}</span>
                  </td>
                  <td className="px-5 py-3 max-w-xs">
                    <p className="text-gray-300 line-clamp-1">{sig.headline}</p>
                  </td>
                  <td className="px-5 py-3">
                    <DirectionBadge direction={sig.direction} />
                  </td>
                  <td className="px-5 py-3 text-right">
                    <span className={`font-mono text-xs ${sig.confidence > 0.7 ? "text-green-400" : sig.confidence > 0.5 ? "text-yellow-400" : "text-gray-400"}`}>
                      {Math.round(sig.confidence * 100)}%
                    </span>
                  </td>
                  <td className="px-5 py-3 text-right font-mono text-xs">
                    <span className={sig.impact_score > 0 ? "text-green-400" : sig.impact_score < 0 ? "text-red-400" : "text-gray-400"}>
                      {sig.impact_score > 0 ? "+" : ""}{(sig.impact_score * 100).toFixed(0)}
                    </span>
                  </td>
                  <td className="px-5 py-3 text-right font-mono text-xs">
                    {sig.quant_score != null ? (
                      <span className={sig.quant_score > 0 ? "text-green-400" : "text-red-400"}>
                        {sig.quant_score > 0 ? "+" : ""}{(sig.quant_score * 100).toFixed(1)}
                      </span>
                    ) : "—"}
                  </td>
                  <td className="px-5 py-3 text-right">
                    {tradeMsg[sig.id] ? (
                      <span className="text-xs">{tradeMsg[sig.id]}</span>
                    ) : (
                      <button
                        onClick={() => handleTrade(sig)}
                        disabled={trading[sig.id] || sig.acted_on}
                        className="px-3 py-1 rounded-lg text-xs font-semibold bg-indigo-600/20 text-indigo-400 hover:bg-indigo-600/40 disabled:opacity-40 transition-all border border-indigo-500/30"
                      >
                        {sig.acted_on ? "Traded" : trading[sig.id] ? "..." : "Trade"}
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
