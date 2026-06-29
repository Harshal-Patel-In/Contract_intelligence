import os
from pathlib import Path

# On Render, skip offline mode so the model can be downloaded
on_render = os.environ.get("RENDER", "") == "true"
if not on_render:
    hf_cache = Path(os.path.expanduser("~/.cache/huggingface/hub"))
    model_cached = hf_cache.exists() and any(hf_cache.glob("*all-MiniLM-L6-v2*"))
    if model_cached:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    else:
        os.environ["TRANSFORMERS_OFFLINE"] = "0"
        os.environ["HF_DATASETS_OFFLINE"] = "0"

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from api.router import router
from core.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


def ensure_ollama_running():
    """Only needed when using local Ollama provider."""
    if settings.llm_provider != "ollama":
        logger.info(f"LLM provider is '{settings.llm_provider}' — skipping Ollama startup check.")
        return True

    import socket
    import subprocess
    import time
    from urllib.parse import urlparse

    url = urlparse(settings.ollama_url)
    host = url.hostname or "127.0.0.1"
    port = url.port or 11434

    # Check if Ollama port is open
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1)
    try:
        s.connect((host, port))
        s.close()
        logger.info(f"Ollama is already running on {host}:{port}.")
        return True
    except socket.error:
        pass

    # Only attempt to start ollama serve locally if host is localhost/127.0.0.1
    if host not in ("127.0.0.1", "localhost"):
        logger.warning(f"Ollama is not reachable on {host}:{port}. Cannot start serve automatically for remote hosts.")
        return False

    logger.info("Ollama is not running. Attempting to start 'ollama serve' in background...")
    try:
        startupinfo = None
        creationflags = 0
        if os.name == "nt":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
            
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            startupinfo=startupinfo,
            creationflags=creationflags
        )
        
        # Poll the port for up to 30 seconds
        for i in range(30):
            time.sleep(1)
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(1)
            try:
                s.connect((host, port))
                s.close()
                logger.info(f"Ollama started successfully after {i+1} seconds.")
                return True
            except socket.error:
                pass
        logger.error("Ollama serve took too long to start.")
        return False
    except Exception as e:
        logger.error(f"Failed to start Ollama automatically: {e}")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    import asyncio
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")

    # Only ensure Ollama is running if using local provider
    if settings.llm_provider == "ollama":
        await asyncio.to_thread(ensure_ollama_running)
        logger.info(f"Ollama model: {settings.ollama_model}")
    else:
        logger.info(f"LLM provider: {settings.llm_provider}/{settings.llm_model}")

    logger.info(f"Qdrant: {settings.qdrant_url}")

    # Pre-load embeddings lazily — download/fail in background, don't block startup
    asyncio.create_task(_preload_embeddings())

    # Warm up Ollama only when using local provider
    if settings.llm_provider == "ollama":
        try:
            import requests as _req
            logger.info(f"Warming up Ollama ({settings.ollama_model})...")
            warmup = await asyncio.to_thread(
                lambda: _req.post(
                    settings.ollama_url,
                    json={"model": settings.ollama_model, "prompt": "OK", "stream": False,
                          "options": {"num_predict": 1, "num_ctx": settings.ollama_num_ctx}},
                    timeout=120,
                )
            )
            if warmup.status_code == 200:
                logger.info("Ollama warmed up — model loaded in VRAM.")
            else:
                logger.warning(f"Ollama warmup returned {warmup.status_code}")
        except Exception as e:
            logger.warning(f"Ollama warmup failed (non-fatal, will retry on first request): {e}")

    yield
    logger.info("Shutting down...")


async def _preload_embeddings():
    """Pre-load embedding model in the background so it doesn't block startup."""
    if on_render:
        logger.info("Running on Render — skipping SentenceTransformer pre-load to conserve memory.")
        return
    import asyncio
    try:
        import torch
        device = settings.embedding_device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Pre-loading SentenceTransformer embedding model on device={device}...")
        from core.vector_db import _get_search_model
        await asyncio.to_thread(_get_search_model)
        logger.info("SentenceTransformer ready.")
    except Exception as e:
        logger.warning(f"Embedding model pre-load failed (non-fatal, will load on first request): {e}")


app = FastAPI(
    title=settings.app_name,
    description="AI-powered legal contract analysis platform using RAG + LLM for clause extraction, risk scoring, and compliance checking.",
    version=settings.app_version,
    lifespan=lifespan,
)


# Serve static dashboard assets (ensure build directory exists so it can be mounted)
dist_assets_path = Path("frontend/dist/assets")
dist_assets_path.mkdir(parents=True, exist_ok=True)
app.mount("/assets", StaticFiles(directory="frontend/dist/assets"), name="assets")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root():
    """Serve the dashboard index file directly at the root URL."""
    react_index = Path("frontend/dist/index.html")
    if react_index.exists():
        return FileResponse(react_index)
    return FileResponse("static/index.html")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch all unhandled exceptions and return structured JSON."""
    logger.error(f"Unhandled error on {request.method} {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


@app.get("/health")
def health_check():
    """Health check endpoint for load balancers and monitoring."""
    services = {}
    
    # Check Redis
    try:
        from core.redis_client import redis_client
        redis_client.ping()
        services["redis"] = "healthy"
    except Exception:
        services["redis"] = "unhealthy"
    
    # Check Qdrant
    try:
        from core.vector_db import client
        client.get_collections()
        services["qdrant"] = "healthy"
    except Exception:
        services["qdrant"] = "unhealthy"
    
    # Check LLM provider
    if settings.llm_provider == "ollama":
        try:
            import requests
            resp = requests.get("http://localhost:11434/api/tags", timeout=3)
            if resp.status_code == 200:
                services["ollama"] = "healthy"
            else:
                services["ollama"] = "unhealthy"
        except Exception:
            services["ollama"] = "unhealthy"
    else:
        # Groq API health check — key configured = healthy (no rate-limit calls needed)
        if settings.llm_api_key:
            services["groq"] = "healthy"
        else:
            services["groq"] = "unhealthy"
    
    overall = "healthy" if all(v == "healthy" for v in services.values()) else "degraded"
    
    return {
        "status": overall,
        "version": settings.app_version,
        "services": services,
    }


app.include_router(router)