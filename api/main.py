"""FastAPI application entry point."""

import os

# Must be set before FAISS/ONNX Runtime are imported to avoid OpenMP conflict on macOS
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import logging

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.job_manager import JobManager
from api.deps import init_dependencies
from api.routes import router
from config.settings import AppConfig
from core.utils.logging import setup_logging
from dependency_injection.container import ServiceContainer

load_dotenv()
setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="FinThesis API",
    description="Fintech market research thesis generation and refinement",
    version="0.1.0",
)

# CORS — allow all origins in dev, restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize shared singletons
config = AppConfig.from_env()
container = ServiceContainer(config)
job_manager = JobManager()

# Wire dependencies into routes
init_dependencies(container, job_manager)

# Register routes
app.include_router(router)

logger.info("FinThesis FastAPI app initialized")
