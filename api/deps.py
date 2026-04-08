"""FastAPI dependency injection via Depends()."""

from api.supabase_job_manager import SupabaseJobManager
from dependency_injection.container import ServiceContainer

# Singleton instances, set during app startup
_container: ServiceContainer | None = None
_job_manager: SupabaseJobManager | None = None


def init_dependencies(container: ServiceContainer, job_manager: SupabaseJobManager):
    """Called once at app startup to set the singleton instances."""
    global _container, _job_manager
    _container = container
    _job_manager = job_manager


def get_container() -> ServiceContainer:
    """FastAPI dependency — inject ServiceContainer into route handlers."""
    if _container is None:
        raise RuntimeError("ServiceContainer not initialized")
    return _container


def get_job_manager() -> SupabaseJobManager:
    """FastAPI dependency — inject SupabaseJobManager into route handlers."""
    if _job_manager is None:
        raise RuntimeError("SupabaseJobManager not initialized")
    return _job_manager
