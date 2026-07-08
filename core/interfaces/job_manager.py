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
    async def create_job(self, query: str) -> Any:
        """Create a new job and return a job-like object."""
        pass

    @abstractmethod
    async def get_job(self, job_id: str) -> Optional[Any]:
        """Fetch a job by ID. Returns None if not found."""
        pass

    @abstractmethod
    async def update_status(
        self, job_id: str, status: JobStatus, progress: Optional[str] = None
    ) -> None:
        """Update job status and optional progress message."""
        pass

    @abstractmethod
    async def update_job(self, job_id: str, **fields) -> None:
        """Persist arbitrary field updates on the job."""
        pass

    @abstractmethod
    async def list_jobs(
        self,
        limit: Optional[int] = None,
        offset: int = 0,
        status: Optional[str] = None,
    ) -> list:
        """List jobs, most recent first.

        limit/offset paginate at the storage layer (limit=None returns all);
        status filters by refinement_status.
        """
        pass

    @abstractmethod
    async def match_jobs(
        self,
        query_embedding: Any,
        exclude_id: str,
        top_n: int = 3,
        min_similarity: float = 0.86,
    ) -> list:
        """Past runs most similar to query_embedding, ranked in the storage layer.
        """
        pass
