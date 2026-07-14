"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { setToken, api } from "@/lib/api";

const navItems = [
  { href: "/",          label: "Dashboard",  icon: "⚡" },
  { href: "/signals",   label: "Signals",    icon: "📡" },
  { href: "/clusters",  label: "Storylines", icon: "🗞️"  },
  { href: "/portfolio", label: "Portfolio",  icon: "💼" },
  { href: "/admin",     label: "Model Lab",  icon: "🧠" },
];

export function Sidebar() {
  const path = usePathname();
  const [username, setUsername] = useState<string | null>(null);

  useEffect(() => {
    // Fetch logged-in user info from backend
    api.auth.me()
      .then((me) => setUsername(me.username))
      .catch(() => setUsername(null));
  }, []);

  const handleLogout = () => {
    setToken(null);
    window.location.reload();
  };

  return (
    <aside className="fixed left-0 top-0 h-full w-64 bg-gray-950/80 backdrop-blur-xl border-r border-gray-800/60 flex flex-col z-50">
      {/* Logo Section */}
      <div className="px-6 py-6 border-b border-gray-800/60">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-500 to-indigo-700 flex items-center justify-center text-xl shadow-lg shadow-indigo-500/10">
            📊
          </div>
          <div>
            <p className="text-sm font-black tracking-tight text-white">Storyline</p>
            <p className="text-xs font-bold text-indigo-400">to Signal</p>
          </div>
        </div>
        <p className="text-[10px] font-semibold text-gray-500 uppercase tracking-widest mt-3.5">
          AI Financial Intelligence
        </p>
      </div>

      {/* Nav Link Section */}
      <nav className="flex-1 px-4 py-6 space-y-2">
        {navItems.map((item) => {
          const active = path === item.href;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-200 text-sm font-semibold
                ${active
                  ? "bg-indigo-500/10 text-indigo-400 border border-indigo-500/25 shadow-md shadow-indigo-500/5"
                  : "text-gray-400 hover:bg-gray-900/60 hover:text-gray-200"
                }`}
            >
              <span className="text-base">{item.icon}</span>
              {item.label}
              {active && (
                <span className="ml-auto w-1.5 h-1.5 rounded-full bg-indigo-400 animate-pulse" />
              )}
            </Link>
          );
        })}
      </nav>

      {/* User Profile + Log Out */}
      <div className="p-4 border-t border-gray-800/60 space-y-3">
        {/* User Badge */}
        {username && (
          <div className="flex items-center gap-3 px-3 py-2.5 bg-gray-900/60 border border-gray-800/60 rounded-xl">
            <div className="w-8 h-8 rounded-full bg-gradient-to-br from-indigo-500 to-violet-600 flex items-center justify-center text-sm font-black text-white flex-shrink-0">
              {username.charAt(0).toUpperCase()}
            </div>
            <div className="min-w-0">
              <p className="text-xs font-bold text-white truncate">{username}</p>
              <p className="text-[10px] text-gray-500 font-medium">Signed in</p>
            </div>
          </div>
        )}

        <button
          onClick={handleLogout}
          className="w-full py-2.5 bg-gray-900/80 hover:bg-gray-900 border border-gray-800/60 text-gray-400 hover:text-gray-200 rounded-xl text-xs font-bold transition-all cursor-pointer"
        >
          Sign Out
        </button>

        <div className="bg-gray-900/40 border border-gray-800/40 p-3 rounded-xl">
          <p className="text-[10px] font-bold text-gray-500 uppercase tracking-wide">Pipeline Mode</p>
          <p className="text-xs font-semibold text-gray-300 mt-0.5">CLUST-MCMS-P + FinBERT</p>
        </div>
      </div>
    </aside>
  );
}
