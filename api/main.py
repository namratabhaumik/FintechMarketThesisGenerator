"""FastAPI application entry point."""

import os

# Must be set before any imports that touch FAISS / ONNX Runtime.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import logging  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402

from dotenv import load_dotenv  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from api.deps import init_dependencies  # noqa: E402
from api.routes import router  # noqa: E402
from api.supabase_job_manager import SupabaseJobManager  # noqa: E402
from config.settings import AppConfig  # noqa: E402
from core.utils.logging import setup_logging  # noqa: E402
from dependency_injection.container import ServiceContainer  # noqa: E402

load_dotenv()
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

    container = ServiceContainer(config)
    job_manager = SupabaseJobManager(
        url=config.supabase.url,
        service_role_key=config.supabase.service_role_key,
    )
    init_dependencies(container, job_manager)
    logger.info("FinThesis FastAPI app initialized (Supabase backend)")
    yield


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
)

app.include_router(router)
