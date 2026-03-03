#!/bin/bash

# AI Notes Knowledge Hub — Start Script
# Starts both the FastAPI backend and Next.js frontend

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$PROJECT_DIR/backend"
FRONTEND_DIR="$PROJECT_DIR/frontend"

echo ""
echo "==================================="
echo "  AI Notes Knowledge Hub"
echo "==================================="
echo ""

# Start backend
echo "▶ Starting FastAPI backend on http://localhost:8000"
cd "$BACKEND_DIR"
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

sleep 3

# Start frontend
echo "▶ Starting Next.js frontend on http://localhost:3000"
cd "$FRONTEND_DIR"
npm run dev &
FRONTEND_PID=$!

echo ""
echo "✅ Both services started!"
echo ""
echo "  Frontend:   http://localhost:3000"
echo "  Backend API: http://localhost:8000"
echo "  API Docs:   http://localhost:8000/docs"
echo "  Admin:      http://localhost:3000/admin"
echo ""
echo "  Admin credentials: admin / changeme123 (from backend/.env)"
echo ""
echo "  Press Ctrl+C to stop all services"
echo ""

# Wait for Ctrl+C
trap "echo ''; echo 'Stopping...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" INT
wait
