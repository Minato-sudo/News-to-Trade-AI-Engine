import type { Metadata } from "next";
import { Inter } from "next/font/google";
import { Suspense } from "react";
import "./globals.css";
import { Sidebar } from "@/components/Sidebar";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Storyline to Signal | Financial Intelligence Platform",
  description:
    "Real-time financial signal intelligence powered by AI news clustering and FinBERT sentiment analysis",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.className} bg-gray-950 text-gray-100 min-h-screen`}>
        <div className="flex min-h-screen">
          <Sidebar />
          <main className="flex-1 ml-64 p-6">
            <Suspense fallback={<div className="text-gray-500 text-sm">Loading...</div>}>
              {children}
            </Suspense>
          </main>
        </div>
      </body>
    </html>
  );
}
