"""FastAPI dependency injection via Depends().

Routes depend on the IJobManager abstraction, not on any concrete
implementation. The actual backend (Supabase, Postgres, etc.) is
chosen at app startup in main.py.
"""

from core.interfaces.job_manager import IJobManager
from dependency_injection.container import ServiceContainer

# Singleton instances, set during app startup
_container: ServiceContainer | None = None
_job_manager: IJobManager | None = None


def init_dependencies(container: ServiceContainer, job_manager: IJobManager):
    """Called once at app startup to set the singleton instances."""
    global _container, _job_manager
    _container = container
    _job_manager = job_manager


def get_container() -> ServiceContainer:
    """FastAPI dependency — inject ServiceContainer into route handlers."""
    if _container is None:
        raise RuntimeError("ServiceContainer not initialized")
    return _container


def get_job_manager() -> IJobManager:
    """FastAPI dependency — inject IJobManager into route handlers."""
    if _job_manager is None:
        raise RuntimeError("IJobManager not initialized")
    return _job_manager
