"use client";

import { useState, useEffect, useCallback } from "react";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine
} from "recharts";

// ── Helpers ────────────────────────────────────────────────────────────────────
function SignalBadge({ direction }: { direction: string }) {
  const map: Record<string, { cls: string; icon: string; label: string }> = {
    up:   { cls: "bg-emerald-500/15 text-emerald-400 border-emerald-500/30", icon: "▲", label: "BULLISH" },
    down: { cls: "bg-rose-500/15 text-rose-400 border-rose-500/30",         icon: "▼", label: "BEARISH" },
    flat: { cls: "bg-gray-500/15 text-gray-400 border-gray-500/30",         icon: "—", label: "NEUTRAL" },
  };
  const m = map[direction] ?? map.flat;
  return (
    <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-black tracking-widest border uppercase ${m.cls}`}>
      <span>{m.icon}</span> {m.label}
    </span>
  );
}

function ActionBadge({ hint }: { hint: string }) {
  const isGreen  = hint.toLowerCase().includes("bullish") || hint.toLowerCase().includes("holding");
  const isRed    = hint.toLowerCase().includes("caution") || hint.toLowerCase().includes("bearish");
  const isYellow = hint.toLowerCase().includes("wait") || hint.toLowerCase().includes("neutral") || hint.toLowerCase().includes("low");
  const cls = isGreen
    ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/25"
    : isRed
    ? "bg-rose-500/10 text-rose-400 border-rose-500/25"
    : "bg-amber-500/10 text-amber-400 border-amber-500/25";
  const icon = isGreen ? "✅" : isRed ? "⚠️" : "⏳";
  return (
    <span className={`inline-flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-bold border ${cls}`}>
      {icon} {hint}
    </span>
  );
}

function ConfidenceBar({ value }: { value: number }) {
  const pct   = Math.round(value * 100);
  const color = pct > 70 ? "#10b981" : pct > 55 ? "#f59e0b" : "#ef4444";
  return (
    <div className="flex items-center gap-3 mt-1">
      <div className="flex-1 h-2 bg-gray-900 rounded-full overflow-hidden border border-gray-800">
        <div className="h-full rounded-full transition-all duration-700" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
      <span className="text-xs font-mono font-bold text-gray-300 w-10 text-right">{pct}%</span>
    </div>
  );
}

// Custom recharts tooltip
const PriceTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  const d = payload[0];
  return (
    <div className="bg-gray-900 border border-gray-700 rounded-xl px-3 py-2 text-xs shadow-xl">
      <p className="text-gray-400 mb-1">{label}</p>
      <p style={{ color: d.stroke }} className="font-bold text-sm">${d.value}</p>
      {payload[0]?.payload?.projected && (
        <p className="text-gray-500 mt-0.5">Projected</p>
      )}
    </div>
  );
};

// ── Main Dashboard ─────────────────────────────────────────────────────────────
export default function DashboardPage() {
  const router = useRouter();

  const [signals,      setSignals]      = useState<any[]>([]);
  const [authed,       setAuthed]       = useState(false);
  const [isRegistering,setIsRegistering]= useState(false);
  const [authForm,     setAuthForm]     = useState({ username: "", email: "", password: "" });
  const [authError,    setAuthError]    = useState("");
  const [authLoading,  setAuthLoading]  = useState(false);

  // Ingestion
  const [ingesting,    setIngesting]    = useState(false);
  const [ingestCount,  setIngestCount]  = useState<number | null>(null);

  // Signal generator
  const [ticker,       setTicker]       = useState("AAPL");
  const [headline,     setHeadline]     = useState("");
  const [genLoading,   setGenLoading]   = useState(false);
  const [genError,     setGenError]     = useState("");

  // Latest result + holistic analysis
  const [latestSig,    setLatestSig]    = useState<any | null>(null);
  const [analysis,     setAnalysis]     = useState<any | null>(null);
  const [analysisLoading, setAnalysisLoading] = useState(false);

  // ── Init ──────────────────────────────────────────────────────────────────
  useEffect(() => {
    const token = typeof window !== "undefined" ? localStorage.getItem("token") : null;
    if (token) { setAuthed(true); loadSignals(); }
  }, []);

  const loadSignals = async () => {
    try {
      const sigs = await api.signals.list({ limit: 10 });
      setSignals(sigs);
    } catch { /* silent */ }
  };

  // Load holistic ticker analysis whenever ticker changes (and user is authed)
  const loadAnalysis = useCallback(async (t: string) => {
    if (!authed) return;
    setAnalysisLoading(true);
    try {
      const a = await api.signals.tickerAnalysis(t);
      setAnalysis(a);
    } catch { setAnalysis(null); }
    finally { setAnalysisLoading(false); }
  }, [authed]);

  useEffect(() => {
    if (authed && ticker) loadAnalysis(ticker);
  }, [authed, ticker, loadAnalysis]);

  // ── Auth ──────────────────────────────────────────────────────────────────
  const handleAuth = async (e: React.FormEvent) => {
    e.preventDefault();
    setAuthError("");
    setAuthLoading(true);
    try {
      if (isRegistering) {
        await api.auth.register(authForm.username, authForm.email, authForm.password);
        await api.auth.login(authForm.username, authForm.password);
      } else {
        await api.auth.login(authForm.username, authForm.password);
      }
      setAuthed(true);
      loadSignals();
    } catch (err: any) {
      setAuthError(err.message || "Authentication failed.");
    } finally {
      setAuthLoading(false);
    }
  };

  // ── Ingestion ─────────────────────────────────────────────────────────────
  const triggerIngestion = async () => {
    setIngesting(true);
    setIngestCount(null);
    try {
      const res = await api.ingest.fetch() as any;
      setIngestCount(res.count);
      loadSignals();
    } catch { /* silent */ }
    finally { setIngesting(false); }
  };

  // ── Signal Generation ─────────────────────────────────────────────────────
  const handleGenerateSignal = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!headline.trim()) return;
    setGenLoading(true);
    setGenError("");
    setLatestSig(null);
    try {
      const sig = await api.signals.generate(ticker.toUpperCase(), headline);
      setSignals(prev => [sig, ...prev]);
      setLatestSig(sig);
      // Refresh holistic analysis for this ticker
      // Refresh holistic analysis for this ticker
      loadAnalysis(ticker);
      // Auto-redirect removed per user request so they can test multiple tickers.
    } catch (err: any) {
      setGenError(err.message || "Failed to analyse signal.");
    } finally {
      setGenLoading(false);
    }
  };

  // ── Auth Gate ──────────────────────────────────────────────────────────────
  // view: "login" | "register" | "forgot" | "otp" | "reset"
  const [authView,     setAuthView]     = useState<"login"|"register"|"forgot"|"otp"|"reset">("login");
  const [fpEmail,      setFpEmail]      = useState("");
  const [fpOtp,        setFpOtp]        = useState("");
  const [fpNewPw,      setFpNewPw]      = useState("");
  const [fpMsg,        setFpMsg]        = useState("");
  const [fpLoading,    setFpLoading]    = useState(false);

  const handleForgotSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setFpMsg(""); setFpLoading(true);
    try {
      const res = await fetch("http://localhost:8000/api/users/forgot-password", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: fpEmail }),
      });
      const data = await res.json();
      setFpMsg(data.message || "Code sent!");
      setAuthView("otp");
    } catch { setFpMsg("Request failed. Try again."); }
    finally { setFpLoading(false); }
  };

  const handleOtpSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setFpMsg(""); setFpLoading(true);
    try {
      const res = await fetch("http://localhost:8000/api/users/verify-otp", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: fpEmail, otp: fpOtp }),
      });
      if (!res.ok) { const d = await res.json(); throw new Error(d.detail); }
      setAuthView("reset");
      setFpMsg("");
    } catch (err: any) { setFpMsg(err.message || "Invalid code."); }
    finally { setFpLoading(false); }
  };

  const handleResetSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setFpMsg(""); setFpLoading(true);
    try {
      const res = await fetch("http://localhost:8000/api/users/reset-password", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: fpEmail, otp: fpOtp, new_password: fpNewPw }),
      });
      if (!res.ok) { const d = await res.json(); throw new Error(d.detail); }
      setFpMsg("✅ Password reset! You can now sign in.");
      setTimeout(() => { setAuthView("login"); setFpMsg(""); setFpEmail(""); setFpOtp(""); setFpNewPw(""); }, 2500);
    } catch (err: any) { setFpMsg(err.message || "Reset failed."); }
    finally { setFpLoading(false); }
  };

  if (!authed) {
    const isFp = authView === "forgot" || authView === "otp" || authView === "reset";
    return (
      <div className="flex items-center justify-center min-h-[85vh] gradient-bg">
        <div className="glass-panel p-8 rounded-3xl w-full max-w-md space-y-6 animate-slide-up">
          <div className="text-center space-y-2">
            <div className="w-16 h-16 mx-auto rounded-2xl bg-gradient-to-br from-indigo-500 to-indigo-700 flex items-center justify-center text-3xl shadow-xl shadow-indigo-500/10">
              ⚡
            </div>
            <h1 className="text-2xl font-extrabold tracking-tight text-white mt-4">
              {authView === "login"    && "Welcome to Storyline"}
              {authView === "register" && "Create your account"}
              {authView === "forgot"  && "Forgot Password"}
              {authView === "otp"     && "Enter Verification Code"}
              {authView === "reset"   && "Set New Password"}
            </h1>
            <p className="text-sm text-gray-400">
              {authView === "login"    && "Sign in to access your financial signal dashboard"}
              {authView === "register" && "Sign up to start receiving AI financial intelligence signals"}
              {authView === "forgot"  && "Enter your registered email to receive a reset code"}
              {authView === "otp"     && `We sent a 6-digit code to ${fpEmail}`}
              {authView === "reset"   && "Choose a new password for your account"}
            </p>
          </div>

          {/* ── Login ── */}
          {authView === "login" && (
            <form onSubmit={handleAuth} className="space-y-4">
              <div>
                <label className="block text-xs font-semibold text-gray-400 uppercase tracking-wider mb-1.5">Username or Email</label>
                <input type="text" required value={authForm.username}
                  onChange={e => setAuthForm(f => ({ ...f, username: e.target.value }))}
                  placeholder="trading_pro or name@email.com"
                  className="w-full input-premium rounded-xl px-4 py-3 text-sm text-white placeholder-gray-500 focus:outline-none" />
              </div>
              <div>
                <label className="block text-xs font-semibold text-gray-400 uppercase tracking-wider mb-1.5">Password</label>
                <input type="password" required value={authForm.password}
                  onChange={e => setAuthForm(f => ({ ...f, password: e.target.value }))}
                  placeholder="••••••••"
                  className="w-full input-premium rounded-xl px-4 py-3 text-sm text-white placeholder-gray-500 focus:outline-none" />
                <button type="button" onClick={() => { setAuthView("forgot"); setAuthError(""); }}
                  className="text-xs text-indigo-400 hover:text-indigo-300 mt-1.5 float-right transition-colors">
                  Forgot password?
                </button>
              </div>
              {authError && (
                <div className="p-3.5 bg-rose-500/10 border border-rose-500/20 text-rose-400 text-xs rounded-xl font-medium">⚠️ {authError}</div>
              )}
              <button type="submit" disabled={authLoading}
                className="w-full py-3 btn-primary text-white rounded-xl text-sm font-semibold transition-all cursor-pointer disabled:opacity-50">
                {authLoading ? "Signing in..." : "Sign In"}
              </button>
              <div className="border-t border-gray-800 pt-4 text-center">
                <button type="button" onClick={() => { setAuthView("register"); setAuthError(""); }}
                  className="text-xs font-semibold text-indigo-400 hover:text-indigo-300 transition-colors">
                  Don&apos;t have an account? Sign Up
                </button>
              </div>
            </form>
          )}

          {/* ── Register ── */}
          {authView === "register" && (
            <form onSubmit={handleAuth} className="space-y-4">
              <div>
                <label className="block text-xs font-semibold text-gray-400 uppercase tracking-wider mb-1.5">Email Address</label>
                <input type="email" required value={authForm.email}
                  onChange={e => setAuthForm(f => ({ ...f, email: e.target.value }))}
                  placeholder="name@company.com"
                  className="w-full input-premium rounded-xl px-4 py-3 text-sm text-white placeholder-gray-500 focus:outline-none" />
              </div>
              <div>
                <label className="block text-xs font-semibold text-gray-400 uppercase tracking-wider mb-1.5">Username</label>
                <input type="text" required value={authForm.username}
                  onChange={e => setAuthForm(f => ({ ...f, username: e.target.value }))}
                  placeholder="trading_pro"
                  className="w-full input-premium rounded-xl px-4 py-3 text-sm text-white placeholder-gray-500 focus:outline-none" />
              </div>
              <div>
                <label className="block text-xs font-semibold text-gray-400 uppercase tracking-wider mb-1.5">Password</label>
                <input type="password" required value={authForm.password}
                  onChange={e => setAuthForm(f => ({ ...f, password: e.target.value }))}
                  placeholder="••••••••"
                  className="w-full input-premium rounded-xl px-4 py-3 text-sm text-white placeholder-gray-500 focus:outline-none" />
              </div>
              {authError && (
                <div className="p-3.5 bg-rose-500/10 border border-rose-500/20 text-rose-400 text-xs rounded-xl font-medium">⚠️ {authError}</div>
              )}
              <button type="submit" disabled={authLoading}
                className="w-full py-3 btn-primary text-white rounded-xl text-sm font-semibold transition-all cursor-pointer disabled:opacity-50">
                {authLoading ? "Creating account..." : "Sign Up"}
              </button>
              <div className="border-t border-gray-800 pt-4 text-center">
                <button type="button" onClick={() => { setAuthView("login"); setAuthError(""); }}
                  className="text-xs font-semibold text-indigo-400 hover:text-indigo-300 transition-colors">
                  Already have an account? Sign In
                </button>
              </div>
            </form>
          )}

          {/* ── Step 1: Enter Email ── */}
          {authView === "forgot" && (
            <form onSubmit={handleForgotSubmit} className="space-y-4">
              <div>
                <label className="block text-xs font-semibold text-gray-400 uppercase tracking-wider mb-1.5">Registered Email</label>
                <input type="email" required value={fpEmail}
                  onChange={e => setFpEmail(e.target.value)}
                  placeholder="name@company.com"
                  className="w-full input-premium rounded-xl px-4 py-3 text-sm text-white placeholder-gray-500 focus:outline-none" />
              </div>
              {fpMsg && <div className="p-3 bg-indigo-500/10 border border-indigo-500/20 text-indigo-300 text-xs rounded-xl">{fpMsg}</div>}
              <button type="submit" disabled={fpLoading}
                className="w-full py-3 btn-primary text-white rounded-xl text-sm font-semibold cursor-pointer disabled:opacity-50">
                {fpLoading ? "Sending..." : "Send Reset Code"}
              </button>
              <div className="text-center pt-2">
                <button type="button" onClick={() => setAuthView("login")}
                  className="text-xs text-indigo-400 hover:text-indigo-300 transition-colors">← Back to Sign In</button>
              </div>
            </form>
          )}

          {/* ── Step 2: Enter OTP ── */}
          {authView === "otp" && (
            <form onSubmit={handleOtpSubmit} className="space-y-4">
              <div>
                <label className="block text-xs font-semibold text-gray-400 uppercase tracking-wider mb-1.5">6-Digit Code</label>
                <input type="text" required maxLength={6} value={fpOtp}
                  onChange={e => setFpOtp(e.target.value.replace(/\D/g, ""))}
                  placeholder="123456"
                  className="w-full input-premium rounded-xl px-4 py-3 text-2xl font-mono tracking-[0.5em] text-center text-white placeholder-gray-600 focus:outline-none" />
                <p className="text-xs text-gray-500 mt-1.5 text-center">Check your email (also check spam)</p>
              </div>
              {fpMsg && <div className="p-3 bg-rose-500/10 border border-rose-500/20 text-rose-400 text-xs rounded-xl">{fpMsg}</div>}
              <button type="submit" disabled={fpLoading || fpOtp.length !== 6}
                className="w-full py-3 btn-primary text-white rounded-xl text-sm font-semibold cursor-pointer disabled:opacity-50">
                {fpLoading ? "Verifying..." : "Verify Code"}
              </button>
              <div className="text-center space-y-1 pt-1">
                <button type="button" onClick={handleForgotSubmit as any}
                  className="text-xs text-gray-500 hover:text-indigo-400 transition-colors block mx-auto">Resend code</button>
                <button type="button" onClick={() => setAuthView("login")}
                  className="text-xs text-indigo-400 hover:text-indigo-300 transition-colors">← Back to Sign In</button>
              </div>
            </form>
          )}

          {/* ── Step 3: New Password ── */}
          {authView === "reset" && (
            <form onSubmit={handleResetSubmit} className="space-y-4">
              <div>
                <label className="block text-xs font-semibold text-gray-400 uppercase tracking-wider mb-1.5">New Password</label>
                <input type="password" required minLength={6} value={fpNewPw}
                  onChange={e => setFpNewPw(e.target.value)}
                  placeholder="Min 6 characters"
                  className="w-full input-premium rounded-xl px-4 py-3 text-sm text-white placeholder-gray-500 focus:outline-none" />
              </div>
              {fpMsg && (
                <div className={`p-3 border text-xs rounded-xl ${fpMsg.startsWith("✅") ? "bg-emerald-500/10 border-emerald-500/20 text-emerald-400" : "bg-rose-500/10 border-rose-500/20 text-rose-400"}`}>
                  {fpMsg}
                </div>
              )}
              <button type="submit" disabled={fpLoading}
                className="w-full py-3 btn-primary text-white rounded-xl text-sm font-semibold cursor-pointer disabled:opacity-50">
                {fpLoading ? "Resetting..." : "Reset Password"}
              </button>
            </form>
          )}
        </div>
      </div>
    );
  }

  // ── Stats ──────────────────────────────────────────────────────────────────
  const upCount   = signals.filter(s => s.direction === "up").length;
  const downCount = signals.filter(s => s.direction === "down").length;
  const avgConf   = signals.length
    ? signals.reduce((a, s) => a + s.confidence, 0) / signals.length
    : 0;

  // Build chart data: merge historical + projected
  const chartData = [
    ...(analysis?.price_history ?? []).map((p: any) => ({ ...p, type: "history" })),
    ...(analysis?.projected_prices ?? []).map((p: any) => ({ ...p, type: "projected" })),
  ];
  const joinDate = analysis?.price_history?.slice(-1)[0]?.date;

  return (
    <div className="space-y-8 animate-slide-up max-w-7xl mx-auto">

      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-gray-800 pb-5">
        <div>
          <h1 className="text-3xl font-extrabold tracking-tight text-white">Markets Hub</h1>
          <p className="text-sm text-gray-400 mt-1">Real-time news-to-signal intelligence — each signal is your own private analysis</p>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={triggerIngestion} disabled={ingesting}
            className="flex items-center gap-2 px-5 py-2.5 bg-gray-900 hover:bg-gray-800 border border-gray-800 rounded-xl text-sm font-semibold text-gray-200 transition-all cursor-pointer disabled:opacity-50"
          >
            {ingesting ? <span className="w-4 h-4 border-2 border-indigo-400 border-t-transparent rounded-full animate-spin" /> : "📥"}
            {ingesting ? "Ingesting..." : "Fetch New Articles"}
          </button>
        </div>
      </div>

      {ingestCount !== null && (
        <div className="p-3 bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-sm rounded-xl font-medium flex items-center justify-between">
          <span>✅ Fetched and deduplicated {ingestCount} new articles.</span>
          <button onClick={() => setIngestCount(null)} className="text-xs hover:underline">Dismiss</button>
        </div>
      )}

      {/* Stats Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
        {[
          { label: "Your Signals", value: signals.length, desc: "signals generated by you", icon: "📊" },
          { label: "Bullish Calls",  value: upCount,         desc: "positive news signals",   icon: "📈" },
          { label: "Bearish Calls",  value: downCount,       desc: "negative news signals",   icon: "📉" },
          { label: "Avg Confidence", value: `${Math.round(avgConf * 100)}%`, desc: "model calibrated score", icon: "🎯" },
        ].map((stat, idx) => (
          <div key={idx} className="glass-panel p-5 rounded-2xl flex items-center justify-between border border-gray-800/60">
            <div className="space-y-1">
              <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider">{stat.label}</span>
              <p className="text-3xl font-black text-white">{stat.value}</p>
              <p className="text-xs text-gray-500">{stat.desc}</p>
            </div>
            <div className="text-3xl p-3 bg-gray-900/60 border border-gray-800 rounded-xl">{stat.icon}</div>
          </div>
        ))}
      </div>

      {/* Analysis Form + Holistic Ticker View */}
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">

        {/* ── Left: Input Form ── */}
        <div className="lg:col-span-2 space-y-5">
          <div className="glass-panel p-6 rounded-2xl border border-gray-800/60 space-y-4">
            <div>
              <h2 className="text-lg font-bold text-white">Analyse a News Article</h2>
              <p className="text-xs text-gray-400 mt-0.5">
                Paste any news headline. The AI will classify it, match it to past events, and give you a plain-English verdict.
              </p>
            </div>

            <form onSubmit={handleGenerateSignal} className="space-y-4">
              <div>
                <label className="block text-xs font-semibold text-gray-400 uppercase tracking-wider mb-1.5">
                  Stock Ticker
                </label>
                <div className="flex gap-2 flex-wrap mb-2">
                  {["AAPL", "TSLA", "NVDA", "MSFT", "RELIANCE.NS", "RY.TO", "BHP.AX", "VOW3.DE"].map(t => (
                    <button
                      key={t} type="button" onClick={() => setTicker(t)}
                      className={`px-3 py-1 rounded-lg text-xs font-mono transition-all border ${
                        ticker === t
                          ? "bg-indigo-500/10 border-indigo-500/40 text-indigo-400"
                          : "bg-gray-900 border-gray-800 text-gray-400 hover:text-gray-200"
                      }`}
                    >
                      ${t}
                    </button>
                  ))}
                </div>
                <input
                  value={ticker}
                  onChange={e => setTicker(e.target.value.toUpperCase())}
                  placeholder="e.g. AAPL, RELIANCE.NS, VOW3.DE"
                  maxLength={15}
                  className="w-full input-premium rounded-xl px-3 py-2.5 text-sm font-mono text-white placeholder-gray-500 focus:outline-none uppercase"
                />
                <p className="text-[10px] text-gray-500 mt-1.5 px-1 font-medium">
                  * You can type <b>any</b> global stock ticker recognized by Yahoo Finance (add extensions like .NS for India, .DE for Germany, .TO for Canada).
                </p>
              </div>

              <div>
                <label className="block text-xs font-semibold text-gray-400 uppercase tracking-wider mb-1.5">
                  News Headline or Article Text
                </label>
                <textarea
                  value={headline}
                  onChange={e => setHeadline(e.target.value)}
                  placeholder="E.g., Apple reports record quarterly earnings, beating Wall Street estimates by 12%..."
                  rows={4}
                  className="w-full input-premium rounded-xl px-3 py-2.5 text-sm text-white placeholder-gray-500 focus:outline-none resize-none"
                />
              </div>

              {genError && (
                <div className="p-3 bg-rose-500/10 border border-rose-500/20 text-rose-400 text-xs rounded-xl font-medium">
                  {genError}
                </div>
              )}

              <button
                type="submit" disabled={genLoading || !headline.trim()}
                className="w-full py-3 btn-primary text-white rounded-xl text-sm font-semibold transition-all cursor-pointer disabled:opacity-50"
              >
                {genLoading
                  ? <span className="flex items-center justify-center gap-2">
                      <span className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                      Running Analysis...
                    </span>
                  : "⚡ Analyse This Article"}
              </button>
            </form>
          </div>

          {/* ── This Article's Result Card ── */}
          {latestSig && (
            <div className="glass-panel p-6 rounded-2xl border border-indigo-500/30 space-y-4 animate-slide-up">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-xs font-bold text-indigo-400 uppercase tracking-wider mb-1">This Article's Signal</p>
                  <p className="text-xs font-mono text-gray-500">${latestSig.ticker}</p>
                </div>
                <SignalBadge direction={latestSig.direction} />
              </div>

              <div>
                <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">Confidence</p>
                <ConfidenceBar value={latestSig.confidence} />
              </div>

              {/* Probability breakdown */}
              {latestSig.prob_up != null && (
                <div className="grid grid-cols-3 gap-2 text-center">
                  {[
                    { label: "Bullish", value: latestSig.prob_up, color: "#10b981" },
                    { label: "Neutral", value: latestSig.prob_flat, color: "#9ca3af" },
                    { label: "Bearish", value: latestSig.prob_down, color: "#ef4444" },
                  ].map(p => (
                    <div key={p.label} className="bg-gray-900/60 border border-gray-800 rounded-xl px-2 py-2">
                      <p className="text-[10px] text-gray-500 mb-1 font-semibold uppercase tracking-wide">{p.label}</p>
                      <p className="text-sm font-black" style={{ color: p.color }}>
                        {Math.round((p.value ?? 0) * 100)}%
                      </p>
                    </div>
                  ))}
                </div>
              )}

              {/* Plain-English */}
              {latestSig.plain_english && (
                <div className="p-4 bg-gray-900/40 border border-gray-800 rounded-xl">
                  <p className="text-xs font-bold text-gray-400 uppercase tracking-wide mb-2">What This Means</p>
                  <p className="text-sm text-gray-200 leading-relaxed">{latestSig.plain_english}</p>
                </div>
              )}

              {/* Action Hint */}
              {latestSig.action_hint && (
                <ActionBadge hint={latestSig.action_hint} />
              )}

              {latestSig.cluster_id && (
                <p className="text-[10px] text-indigo-400 font-semibold animate-pulse">
                  📖 Taking you to the Storylines page for the related event cluster...
                </p>
              )}
            </div>
          )}
        </div>

        {/* ── Right: Holistic Ticker Analysis ── */}
        <div className="lg:col-span-3 space-y-5">

          {/* Price Chart */}
          <div className="glass-panel p-6 rounded-2xl border border-gray-800/60">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h2 className="text-lg font-bold text-white">${ticker} — 30-Day Price + Projection</h2>
                <p className="text-xs text-gray-400 mt-0.5">Historical close prices + AI-projected next 3 trading days</p>
              </div>
              {analysisLoading && (
                <span className="w-4 h-4 border-2 border-indigo-400 border-t-transparent rounded-full animate-spin" />
              )}
            </div>

            {chartData.length > 0 ? (
              <ResponsiveContainer width="100%" height={220}>
                <LineChart data={chartData} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                  <XAxis
                    dataKey="date"
                    tick={{ fill: "#6b7280", fontSize: 10 }}
                    tickFormatter={d => d.slice(5)}
                    interval={5}
                  />
                  <YAxis
                    tick={{ fill: "#6b7280", fontSize: 10 }}
                    domain={["auto", "auto"]}
                  />
                  <Tooltip content={<PriceTooltip />} />
                  {joinDate && (
                    <ReferenceLine
                      x={joinDate}
                      stroke="#6366f1"
                      strokeDasharray="4 3"
                      label={{ value: "Today", fill: "#818cf8", fontSize: 10, position: "top" }}
                    />
                  )}
                  {/* Historical line */}
                  <Line
                    type="monotone"
                    data={chartData.filter(d => d.type === "history")}
                    dataKey="price"
                    stroke="#6366f1"
                    strokeWidth={2}
                    dot={false}
                    activeDot={{ r: 4, fill: "#818cf8" }}
                  />
                  {/* Projected line */}
                  <Line
                    type="monotone"
                    data={[
                      // bridge from last historical point
                      ...(analysis?.price_history?.slice(-1) ?? []).map((p: any) => ({ ...p, type: "projected" })),
                      ...(analysis?.projected_prices ?? []),
                    ]}
                    dataKey="price"
                    stroke={
                      analysis?.overall_direction === "up"
                        ? "#10b981"
                        : analysis?.overall_direction === "down"
                        ? "#ef4444"
                        : "#f59e0b"
                    }
                    strokeWidth={2}
                    strokeDasharray="5 4"
                    dot={{ r: 3, fill: "#111827", stroke: "#10b981", strokeWidth: 2 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-[220px] flex items-center justify-center text-gray-600 text-sm">
                {analysisLoading ? "Loading price data..." : "Price data unavailable (no internet access to Yahoo Finance)"}
              </div>
            )}
          </div>

          {/* Overall Ticker Picture */}
          <div className="glass-panel p-6 rounded-2xl border border-gray-800/60 space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-lg font-bold text-white">Overall Picture for ${ticker}</h2>
                <p className="text-xs text-gray-400 mt-0.5">Based on ALL recent news signals — not just this one article</p>
              </div>
              {analysis && <SignalBadge direction={analysis.overall_direction} />}
            </div>

            {analysis ? (
              <>
                {/* Signal breakdown bars */}
                <div className="space-y-3">
                  {[
                    { label: "Bullish signals", count: analysis.up_count,   total: analysis.signal_count, color: "#10b981" },
                    { label: "Bearish signals", count: analysis.down_count, total: analysis.signal_count, color: "#ef4444" },
                    { label: "Neutral signals", count: analysis.flat_count, total: analysis.signal_count, color: "#6b7280" },
                  ].map(b => (
                    <div key={b.label}>
                      <div className="flex justify-between text-xs text-gray-400 mb-1">
                        <span className="font-semibold">{b.label}</span>
                        <span className="font-mono" style={{ color: b.color }}>
                          {b.count} / {b.total}
                        </span>
                      </div>
                      <div className="h-2 bg-gray-900 rounded-full overflow-hidden border border-gray-800">
                        <div
                          className="h-full rounded-full transition-all duration-700"
                          style={{
                            width: b.total ? `${(b.count / b.total) * 100}%` : "0%",
                            backgroundColor: b.color,
                          }}
                        />
                      </div>
                    </div>
                  ))}
                </div>

                {/* Overall plain-English */}
                <div className="p-4 bg-gray-900/40 border border-gray-800 rounded-xl">
                  <p className="text-xs font-bold text-gray-400 uppercase tracking-wide mb-2">Big Picture Summary</p>
                  <p className="text-sm text-gray-200 leading-relaxed">{analysis.plain_english}</p>
                </div>

                <div className="text-xs text-gray-500 font-medium">
                  {analysis.signal_count} total signals analysed in the last 30 days for ${ticker}
                </div>
              </>
            ) : (
              <div className="text-sm text-gray-500 py-4">
                {analysisLoading ? "Loading analysis..." : `No signals yet for ${ticker}. Generate one above!`}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Signal History Feed */}
      <div className="glass-panel rounded-2xl border border-gray-800/60 overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-800 flex items-center justify-between bg-gray-900/30">
          <h2 className="text-md font-bold text-gray-200">Your Signal History</h2>
          <span className="flex items-center gap-1.5 text-xs font-bold text-emerald-400 bg-emerald-500/10 px-2.5 py-1 rounded-full border border-emerald-500/20 uppercase tracking-wider animate-pulse">
            <span className="w-1.5 h-1.5 bg-emerald-400 rounded-full" />
            Live
          </span>
        </div>

        {signals.length === 0 ? (
          <div className="px-6 py-16 text-center text-gray-500 text-sm">
            No signals yet — paste a headline above and click Analyse.
          </div>
        ) : (
          <div className="divide-y divide-gray-800/60">
            {signals.map((sig) => (
              <div key={sig.id} className="px-6 py-4 hover:bg-gray-800/10 transition-colors">
                <div className="flex items-start justify-between gap-6">
                  <div className="flex-1 min-w-0 space-y-1.5">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="text-xs font-mono font-bold text-indigo-400 bg-indigo-500/10 px-2 py-0.5 rounded border border-indigo-500/20">
                        ${sig.ticker}
                      </span>
                      <SignalBadge direction={sig.direction} />
                      {sig.impact_score !== undefined && (
                        <span className={`text-xs font-mono font-semibold ${
                          sig.impact_score > 0 ? "text-emerald-400" : sig.impact_score < 0 ? "text-rose-400" : "text-gray-400"
                        }`}>
                          {sig.impact_score > 0 ? "+" : ""}{(sig.impact_score * 100).toFixed(0)}pts
                        </span>
                      )}
                    </div>
                    <p className="text-sm font-medium text-gray-200 line-clamp-2">{sig.headline}</p>
                    <ConfidenceBar value={sig.confidence} />
                    {sig.action_hint && (
                      <p className="text-xs text-gray-500 font-semibold">{sig.action_hint}</p>
                    )}
                  </div>
                  <div className="text-right shrink-0 space-y-1.5">
                    <p className="text-xs text-gray-500">
                      {new Date(sig.generated_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                    </p>
                    {sig.quant_score != null && (
                      <span className={`inline-block text-xs font-mono px-2 py-0.5 rounded border ${
                        sig.quant_score > 0
                          ? "bg-emerald-500/5 border-emerald-500/20 text-emerald-400"
                          : "bg-rose-500/5 border-rose-500/20 text-rose-400"
                      }`}>
                        QS: {sig.quant_score > 0 ? "+" : ""}{(sig.quant_score * 100).toFixed(1)}
                      </span>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
