#!/bin/bash
# run.sh — Start the full Storyline-to-Signal stack (dev mode)
# Usage: bash run.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export HF_ENDPOINT=https://hf-mirror.com

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║       From Storyline to Signal — Dev Launcher           ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ── Check venv ────────────────────────────────────────────────────────────────
if [ ! -f "venv/bin/python" ]; then
    echo "❌ Virtual environment not found. Run: python3 -m venv venv && venv/bin/pip install -r requirements.txt"
    exit 1
fi

# ── Start FastAPI backend ─────────────────────────────────────────────────────
echo "🚀 Starting FastAPI backend on http://localhost:8000 ..."
venv/bin/python -m uvicorn api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    --log-level info &

API_PID=$!
echo "   API PID: $API_PID"

# ── Start Next.js frontend ────────────────────────────────────────────────────
if [ -d "frontend/node_modules" ]; then
    echo "🌐 Starting Next.js frontend on http://localhost:3000 ..."
    cd frontend && npm run dev &
    FRONTEND_PID=$!
    cd ..
    echo "   Frontend PID: $FRONTEND_PID"
else
    echo "⚠️  Frontend dependencies not installed. Run: cd frontend && npm install"
fi

echo ""
echo "✅ Stack running!"
echo "   API:      http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"
echo "   Frontend: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop all services."

# ── Wait and cleanup ──────────────────────────────────────────────────────────
trap "echo ''; echo 'Stopping...'; kill $API_PID $FRONTEND_PID 2>/dev/null; exit 0" SIGINT SIGTERM
wait
