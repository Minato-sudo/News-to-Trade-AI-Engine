// frontend/src/app/api/price/route.ts
// This Next.js API Route runs ON VERCEL (full internet access) — NOT on Hugging Face.
// It fetches stock price history directly from Yahoo Finance, bypassing HF's network block.

import { NextRequest, NextResponse } from "next/server";

export async function GET(req: NextRequest) {
  const { searchParams } = new URL(req.url);
  const ticker = (searchParams.get("ticker") || "AAPL").toUpperCase();
  const days = parseInt(searchParams.get("days") || "35", 10);

  try {
    // Calculate Unix timestamps for Yahoo Finance query
    const now = Math.floor(Date.now() / 1000);
    const from = now - days * 24 * 60 * 60;

    const yahooUrl =
      `https://query1.finance.yahoo.com/v8/finance/chart/${ticker}` +
      `?period1=${from}&period2=${now}&interval=1d&events=history`;

    const resp = await fetch(yahooUrl, {
      headers: {
        "User-Agent":
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
      },
      // Revalidate every 60 minutes on Vercel's edge cache
      next: { revalidate: 3600 },
    });

    if (!resp.ok) {
      return NextResponse.json({ error: "Yahoo Finance fetch failed", prices: [], meta: {} }, { status: 200 });
    }

    const json = await resp.json();
    const result = json?.chart?.result?.[0];

    if (!result) {
      return NextResponse.json({ prices: [], meta: {} });
    }

    const timestamps: number[] = result.timestamp ?? [];
    const closes: number[] = result.indicators?.quote?.[0]?.close ?? [];
    const meta = result.meta ?? {};

    const prices = timestamps
      .map((ts: number, i: number) => ({
        date: new Date(ts * 1000).toISOString().slice(0, 10),
        price: closes[i] != null ? Math.round(closes[i] * 100) / 100 : null,
      }))
      .filter((p: { date: string; price: number | null }) => p.price !== null);

    return NextResponse.json({
      prices,
      meta: {
        currency: meta.currency,
        symbol: meta.symbol,
        regularMarketPrice: meta.regularMarketPrice,
      },
    });
  } catch (err: unknown) {
    console.error("[price/route] Error:", err);
    return NextResponse.json({ prices: [], meta: {} });
  }
}
