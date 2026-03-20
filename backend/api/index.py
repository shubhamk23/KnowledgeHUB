"""Vercel Python serverless entry point.

Vercel routes all requests to this ASGI handler.
The `rootDirectory` for this Vercel project is `backend/`, so imports
resolve relative to that directory.
"""
from app.main import app  # noqa: F401 — Vercel picks up the `app` ASGI callable
