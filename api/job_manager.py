"""In-memory job manager for background thesis generation.
This module defines a Job class to represent individual thesis generation jobs and a JobManager class to manage these jobs in a thread-safe manner. 
The JobManager allows creating new jobs, retrieving existing jobs by ID, updating job status, and listing all jobs. 
"""

import logging
import uuid
from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, List, Optional

from core.models.article import Article
from core.models.thesis import StructuredThesis
from api.schemas import JobStatus

logger = logging.getLogger(__name__)


@dataclass
class Job:
    """Represents a thesis generation job."""
    id: str
    query: str
    status: JobStatus = JobStatus.PENDING
    progress: Optional[str] = None
    thesis: Optional[StructuredThesis] = None
    articles: List[Article] = field(default_factory=list)
    error: Optional[str] = None
    # Refinement state
    refinement_count: int = 0
    refinement_status: str = "refining"
    feedback_history: List[List[str]] = field(default_factory=list)
    execution_log: list = field(default_factory=list)
    retrieved_docs: list = field(default_factory=list)


class JobManager:
    """Thread-safe in-memory job store."""

    def __init__(self):
        self._jobs: Dict[str, Job] = {}
        self._lock = Lock() #lock for? How does it work?

    def create_job(self, query: str) -> Job:
        """Create a new job and return it."""
        job_id = uuid.uuid4().hex[:12]
        job = Job(id=job_id, query=query)
        with self._lock:
            self._jobs[job_id] = job
        logger.info(f"Job created: {job_id} for query: {query!r}")
        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        """Get a job by ID."""
        with self._lock:
            return self._jobs.get(job_id)

    def update_status(self, job_id: str, status: JobStatus, progress: Optional[str] = None):
        """Update job status and optional progress message."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = status
                if progress is not None:
                    job.progress = progress

    def list_jobs(self) -> List[Job]:
        """List all jobs, most recent first."""
        with self._lock:
            return list(reversed(self._jobs.values()))