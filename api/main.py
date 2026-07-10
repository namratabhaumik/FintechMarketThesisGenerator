"""FastAPI application entry point."""

import os

# Must be set before any imports that touch FAISS / ONNX Runtime.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

import logging  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from slowapi.errors import RateLimitExceeded  # noqa: E402
from slowapi.middleware import SlowAPIMiddleware  # noqa: E402

from supabase import acreate_client  # noqa: E402

from api.auth import init_client_pool, shutdown_client_pool  # noqa: E402
from api.deps import init_dependencies  # noqa: E402
from api.routes import router  # noqa: E402
from api.security import (  # noqa: E402
    GLOBAL_RATE_LIMIT_ENABLED,
    limiter,
    rate_limit_handler,
)
from api.supabase_job_manager import SupabaseJobManager  # noqa: E402
from config.settings import AppConfig  # noqa: E402
from core.utils.logging import setup_logging  # noqa: E402
from dependency_injection.container import ServiceContainer  # noqa: E402

setup_logging()

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Wire dependencies on server startup, not at import.

    Building the container and Supabase client at import time meant a bare
    `import api.main` needed live credentials, which blocked dumping the OpenAPI
    schema on a fresh clone (the TS type-gen step). Deferring to lifespan keeps
    import side-effect-free: the schema can be introspected without a DB, and
    the app only touches Supabase once a server actually starts.
    """
    config = AppConfig.from_env()
    if not config.supabase.enabled:
        raise RuntimeError(
            "Supabase is required to run the API (the jobs table is the single "
            "state carrier). Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY."
        )

    # Initialize the client pool for per-request user-scoped clients
    init_client_pool(pool_size=20)

    container = ServiceContainer(config)
    # acreate_client is a coroutine, so the async Supabase client is built here
    # (in the running loop) rather than in the manager's __init__.
    client = await acreate_client(
        config.supabase.url, config.supabase.service_role_key
    )
    job_manager = SupabaseJobManager(client)
    init_dependencies(container, job_manager)
    logger.info("FinThesis FastAPI app initialized (Supabase backend)")
    yield
    # Shutdown: close all pooled clients
    await shutdown_client_pool()


app = FastAPI(
    title="FinThesis API",
    description="Fintech market thesis generation and refinement",
    version="0.2.0",
    lifespan=lifespan,
)

# for the frontend to be deployed as a separate origin; the default covers local dev.
_default_cors = "http://localhost:3000,http://127.0.0.1:3000"
_cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOW_ORIGINS", _default_cors).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
    # lets cross-origin browser JS read the Location set on POST /theses (201).
    expose_headers=["Location"],
)

# Rate limiting: per-route limits (see routes.py) always apply; the global
# ceiling is opt-in via RATE_LIMIT_DEFAULT and only then needs the middleware.
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_handler)
if GLOBAL_RATE_LIMIT_ENABLED:
    app.add_middleware(SlowAPIMiddleware)

app.include_router(router)
