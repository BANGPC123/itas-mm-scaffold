"""FastAPI application entrypoint.

Run with: uvicorn backend.main:app --reload --port 8000
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import context, pipeline

app = FastAPI(
    title="ITAS-MM API",
    description=(
        "Intelligent Multimodal Traffic Assistance System — "
        "perception, context, reasoning, and interaction endpoints."
    ),
    version="0.1.0",
)

# ASSUMPTION: permissive CORS for local dev with a separate Vite frontend
# origin. Restrict allow_origins before any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(context.router)
app.include_router(pipeline.router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
