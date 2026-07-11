// lib/api.ts — API client for backend communication
const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

let _token: string | null = null;
if (typeof window !== "undefined") {
  _token = localStorage.getItem("token");
}

export function setToken(t: string | null) { 
  _token = t; 
  if (typeof window !== "undefined") {
    if (t) localStorage.setItem("token", t);
    else localStorage.removeItem("token");
  }
}
export function getToken() { return _token; }

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(options.headers as Record<string, string> || {}),
  };
  const wasLoggedIn = !!_token;
  if (_token) headers["Authorization"] = `Bearer ${_token}`;

  const res = await fetch(`${API_URL}${path}`, { ...options, headers });
  
  // Auto-logout on token expiration (only if we were actually logged in)
  if (res.status === 401 && wasLoggedIn) {
    setToken(null);
    if (typeof window !== "undefined") {
      window.location.reload();
    }
  }

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "API error");
  }
  return res.json();
}

// ── Auth ─────────────────────────────────────────────────────────────────────
export const api = {
  auth: {
    register: (u: string, e: string, p: string) =>
      request("/api/users/register", {
        method: "POST",
        body: JSON.stringify({ username: u, email: e, password: p }),
      }),
    login: async (username: string, password: string) => {
      const form = new URLSearchParams({ username, password });
      const res = await fetch(`${API_URL}/api/users/login`, {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: form.toString(),
      });
      if (!res.ok) throw new Error("Login failed");
      const data = await res.json();
      setToken(data.access_token);
      return data;
    },
    me: () => request<{ id: number; username: string; email: string }>("/api/users/me"),
  },

  signals: {
    list: (params?: { ticker?: string; direction?: string; limit?: number }) =>
      request<any[]>(`/api/signals/?${new URLSearchParams(params as any)}`),

    generate: (ticker: string, headline: string, text?: string) =>
      request<any>("/api/signals/generate", {
        method: "POST",
        body: JSON.stringify({ ticker, headline, text }),
      }),

    get: (id: number) => request<any>(`/api/signals/${id}`),

    tickerAnalysis: (ticker: string, days = 30) =>
      request<any>(`/api/signals/ticker-analysis/${ticker}?days=${days}`),
  },

  clusters: {
    list:   (limit = 20, offset = 0) =>
      request<any[]>(`/api/clusters/?limit=${limit}&offset=${offset}`),
    get:    (id: number) => request<any>(`/api/clusters/${id}`),
    run:    (n = 1000)   => request<any>(`/api/clusters/run-pipeline?n_articles=${n}`, { method: "POST" }),
  },

  portfolio: {
    account:  () => request<any>("/api/portfolio/account"),
    trade:    (signal_id: number) =>
      request<any>("/api/portfolio/trade", {
        method: "POST",
        body: JSON.stringify({ signal_id }),
      }),
    trades:   (status?: string) =>
      request<any[]>(`/api/portfolio/trades${status ? `?status=${status}` : ""}`),
    summary:  () => request<any>("/api/portfolio/summary"),
  },

  ingest: {
    fetch: () => request<any>("/api/ingest/fetch", { method: "POST" }),
  },
};
