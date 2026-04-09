"""Abstract interface for job persistence backends."""

from abc import ABC, abstractmethod
from typing import Any, Optional

from api.schemas import JobStatus


class IJobManager(ABC):
    """Protocol for thesis generation job storage.

    Implementations decide how/where jobs are persisted (in-memory,
    Supabase, Postgres, etc.). Routes depend on this abstraction so the
    backend can be swapped without touching API code.
    """

    @abstractmethod
    def create_job(self, query: str) -> Any:
        """Create a new job and return a job-like object."""
        pass

    @abstractmethod
    def get_job(self, job_id: str) -> Optional[Any]:
        """Fetch a job by ID. Returns None if not found."""
        pass

    @abstractmethod
    def update_status(
        self, job_id: str, status: JobStatus, progress: Optional[str] = None
    ) -> None:
        """Update job status and optional progress message."""
        pass

    @abstractmethod
    def update_job(self, job_id: str, **fields) -> None:
        """Persist arbitrary field updates on the job."""
        pass

    @abstractmethod
    def list_jobs(self) -> list:
        """List all jobs, most recent first."""
        pass
