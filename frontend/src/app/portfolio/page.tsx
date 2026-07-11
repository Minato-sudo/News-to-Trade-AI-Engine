"use client";
import { useState, useEffect } from "react";
import { api } from "@/lib/api";

export default function PortfolioPage() {
  const [account,  setAccount]  = useState<any>(null);
  const [summary,  setSummary]  = useState<any>(null);
  const [trades,   setTrades]   = useState<any[]>([]);
  const [loading,  setLoading]  = useState(true);

  useEffect(() => {
    Promise.all([
      api.portfolio.account(),
      api.portfolio.summary(),
      api.portfolio.trades(),
    ]).then(([acc, sum, trd]: any[]) => {
      setAccount(acc);
      setSummary(sum);
      setTrades(trd);
    }).catch(() => {}).finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-500">Loading portfolio...</div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Portfolio</h1>
        <p className="text-sm text-gray-400 mt-1">Alpaca paper trading — no real capital involved</p>
      </div>

      {/* Account Cards */}
      {account && (
        <div className="grid grid-cols-4 gap-4">
          {[
            { label: "Portfolio Value", value: `$${Number(account.account?.portfolio_value || 10000).toLocaleString()}`,  color: "indigo" },
            { label: "Cash",           value: `$${Number(account.account?.cash            || 10000).toLocaleString()}`,  color: "violet" },
            { label: "Total P&L",      value: `$${summary?.total_pnl?.toFixed(2) || "0.00"}`,                           color: summary?.total_pnl >= 0 ? "emerald" : "red" },
            { label: "Win Rate",       value: `${summary?.win_rate_pct?.toFixed(1) || "0"}%`,                           color: "amber" },
          ].map(card => (
            <div key={card.label} className="glass p-4 rounded-2xl">
              <p className="text-xs text-gray-400">{card.label}</p>
              <p className="text-2xl font-bold text-white mt-1">{card.value}</p>
              <p className="text-xs text-gray-500 mt-1">
                {card.label === "Win Rate" ? `${summary?.wins || 0}W / ${summary?.losses || 0}L` :
                 card.label === "Cash" ? "Available" : "Paper trading"}
              </p>
            </div>
          ))}
        </div>
      )}

      {/* Open Positions */}
      {account?.open_positions?.length > 0 && (
        <div className="glass rounded-2xl overflow-hidden">
          <div className="px-5 py-3 border-b border-gray-800">
            <p className="text-sm font-semibold text-gray-300">Open Positions</p>
          </div>
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-800 text-gray-400 text-xs uppercase">
                <th className="px-5 py-3 text-left">Ticker</th>
                <th className="px-5 py-3 text-right">Qty</th>
                <th className="px-5 py-3 text-right">Avg Entry</th>
                <th className="px-5 py-3 text-right">Current</th>
                <th className="px-5 py-3 text-right">Unrealized P&L</th>
              </tr>
            </thead>
            <tbody>
              {account.open_positions.map((pos: any) => (
                <tr key={pos.ticker} className="border-b border-gray-800 hover:bg-gray-800/30">
                  <td className="px-5 py-3 font-mono font-bold text-indigo-400">{pos.ticker}</td>
                  <td className="px-5 py-3 text-right text-gray-300">{pos.qty}</td>
                  <td className="px-5 py-3 text-right text-gray-300">${pos.avg_entry?.toFixed(2)}</td>
                  <td className="px-5 py-3 text-right text-gray-300">${pos.current?.toFixed(2)}</td>
                  <td className={`px-5 py-3 text-right font-mono ${(pos.unrealized_pnl || 0) >= 0 ? "text-green-400" : "text-red-400"}`}>
                    {(pos.unrealized_pnl || 0) >= 0 ? "+" : ""}${pos.unrealized_pnl?.toFixed(2)}
                    <span className="text-xs ml-1 opacity-70">({pos.unrealized_pnl_pct?.toFixed(1)}%)</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Trade History */}
      <div className="glass rounded-2xl overflow-hidden">
        <div className="px-5 py-3 border-b border-gray-800">
          <p className="text-sm font-semibold text-gray-300">Trade History</p>
        </div>
        {trades.length === 0 ? (
          <div className="p-10 text-center text-gray-500 text-sm">
            No trades yet. Generate a signal and click "Trade" to paper trade.
          </div>
        ) : (
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-800 text-gray-400 text-xs uppercase">
                <th className="px-5 py-3 text-left">Ticker</th>
                <th className="px-5 py-3 text-left">Side</th>
                <th className="px-5 py-3 text-right">Qty</th>
                <th className="px-5 py-3 text-right">Entry</th>
                <th className="px-5 py-3 text-right">P&L</th>
                <th className="px-5 py-3 text-right">Status</th>
                <th className="px-5 py-3 text-right">Date</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800">
              {trades.map(t => (
                <tr key={t.id} className="hover:bg-gray-800/30 transition-colors">
                  <td className="px-5 py-3 font-mono font-bold text-indigo-400">{t.ticker}</td>
                  <td className="px-5 py-3">
                    <span className={`text-xs font-bold ${t.direction === "buy" ? "text-green-400" : "text-red-400"}`}>
                      {t.direction?.toUpperCase()}
                    </span>
                  </td>
                  <td className="px-5 py-3 text-right text-gray-300">{t.quantity}</td>
                  <td className="px-5 py-3 text-right text-gray-300">${t.entry_price?.toFixed(2)}</td>
                  <td className={`px-5 py-3 text-right font-mono ${(t.pnl || 0) >= 0 ? "text-green-400" : "text-red-400"}`}>
                    {t.pnl != null ? `${t.pnl >= 0 ? "+" : ""}$${t.pnl.toFixed(2)}` : "—"}
                  </td>
                  <td className="px-5 py-3 text-right">
                    <span className={`text-xs px-2 py-0.5 rounded-full ${t.status === "open" ? "badge-up" : "badge-flat"}`}>
                      {t.status}
                    </span>
                  </td>
                  <td className="px-5 py-3 text-right text-xs text-gray-500">
                    {new Date(t.opened_at).toLocaleDateString()}
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
