"""FastAPI dependency injection via Depends().

Job endpoints get their per-request, RLS-scoped job manager from
api.auth.get_user_job_manager; this module only carries the shared
ServiceContainer singleton, set once at app startup in main.py.
"""

from dependency_injection.container import ServiceContainer

# Singleton instance, set during app startup
_container: ServiceContainer | None = None


def init_dependencies(container: ServiceContainer):
    """Called once at app startup to set the singleton instances."""
    global _container
    _container = container


def get_container() -> ServiceContainer:
    """FastAPI dependency — inject ServiceContainer into route handlers."""
    if _container is None:
        raise RuntimeError("ServiceContainer not initialized")
    return _container
