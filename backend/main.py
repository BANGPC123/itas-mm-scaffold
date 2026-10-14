"""FastAPI application entrypoint.

Run with: uvicorn backend.main:app --reload --port 8000
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from contextlib import asynccontextmanager
import asyncio
import os

from backend.routers import context, pipeline

@asynccontextmanager
async def lifespan(app):
    try:
        if os.environ.get('ITAS_WARMUP')=='1':
            from backend.startup import warmup_product
            await asyncio.wrap_future(pipeline.get_inference_worker().submit(
                warmup_product,pipeline.get_pipeline(),pipeline.get_speech_engine()))
        yield
    finally:
        if pipeline.get_pipeline.cache_info().currsize:
            active=pipeline.get_pipeline()
            reader=getattr(active,'supplementary_reader',None)
            if reader is not None:reader.close()
            engine=getattr(active,'tts_engine',None)
            if engine is not None:engine.close()
        if pipeline.get_speech_engine.cache_info().currsize:pipeline.get_speech_engine().close()
        if pipeline.get_inference_worker.cache_info().currsize:
            pipeline.get_inference_worker().shutdown(wait=True,cancel_futures=True)
            pipeline.get_inference_worker.cache_clear()


app = FastAPI(
    title="ITAS-MM API",
    description=(
        "Intelligent Multimodal Traffic Assistance System — "
        "perception, context, reasoning, and interaction endpoints."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# ASSUMPTION: permissive CORS for local dev with a separate Vite frontend
# origin. Restrict allow_origins before any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(context.router)
app.include_router(pipeline.router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

frontend_dist = Path(__file__).resolve().parents[1] / "frontend/dist"
review_dir = Path(__file__).resolve().parents[1] / 'output/review'
if review_dir.is_dir():
    app.mount('/review', StaticFiles(directory=review_dir,html=True), name='review')
if frontend_dist.is_dir():
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
