"""FastAPI application entry point."""

import os

# Must be set before any imports that touch FAISS / ONNX Runtime.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import logging  # noqa: E402

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

app = FastAPI(
    title="FinThesis API",
    description="Fintech market thesis generation and refinement",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

config = AppConfig.from_env()
container = ServiceContainer(config)

job_manager = SupabaseJobManager(
    url=config.supabase.url,
    anon_key=config.supabase.anon_key,
)

init_dependencies(container, job_manager)
app.include_router(router)

logger.info("FinThesis FastAPI app initialized (Supabase backend)")
