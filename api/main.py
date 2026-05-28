"""Model OCR API Gateway - OpenAI-compatible API for PDF OCR.

Routes:
    GET  /v1/models           - List available models
    POST /v1/chat/completions - Chat completions (images as base64)
    POST /v1/ocr              - Upload PDF, get OCR result
    GET  /health              - Health check
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from openai_compat import router as openai_router
from pdf import router as pdf_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://glm-ocr:8000")
API_KEY = os.getenv("API_KEY", "")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hooks."""
    import httpx
    logger.info("Starting API gateway")
    logger.info("Model server endpoint: %s", VLLM_BASE_URL)

    # Verify vLLM is reachable
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{VLLM_BASE_URL}/v1/models")
            if resp.status_code == 200:
                models = resp.json().get("data", [])
                logger.info("vLLM connected - models: %s", [m["id"] for m in models])
            else:
                logger.warning("vLLM returned status %d on startup", resp.status_code)
    except Exception as exc:
        logger.warning("Could not reach vLLM on startup: %s", exc)

    yield

    logger.info("Shutting down API gateway")


app = FastAPI(
    title="Model OCR API",
    description="OpenAI-compatible OCR API powered by a multimodal HF model",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Auth middleware
@app.middleware("http")
async def auth_middleware(request, call_next):
    if not API_KEY:
        return await call_next(request)
    if request.url.path in ("/health", "/docs", "/openapi.json", "/redoc"):
        return await call_next(request)
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer ") and auth[7:] == API_KEY:
        return await call_next(request)
    if auth == API_KEY:
        return await call_next(request)
    from fastapi import HTTPException
    raise HTTPException(status_code=401, detail="Invalid API key")

# Mount routers
app.include_router(openai_router)
app.include_router(pdf_router)


@app.get("/health")
async def health():
    import httpx
    import time

    vllm_status = "unknown"
    start = time.time()
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{VLLM_BASE_URL}/v1/health")
            vllm_status = "healthy" if resp.status_code == 200 else f"error_{resp.status_code}"
    except Exception:
        vllm_status = "unreachable"
    latency = time.time() - start

    return {
        "status": "ok",
        "service": "model-ocr-api",
        "vllm": {"status": vllm_status, "latency_ms": round(latency * 1000, 1)},
        "endpoints": {
            "chat": "/v1/chat/completions",
            "models": "/v1/models",
            "ocr": "/v1/ocr",
        },
    }


@app.get("/")
async def root():
    return {
        "service": "Model OCR API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "GET  /v1/models": "List models",
            "POST /v1/chat/completions": "Chat with images (OpenAI-compatible)",
            "POST /v1/ocr": "OCR a PDF file",
            "GET  /health": "Health check",
        },
    }
