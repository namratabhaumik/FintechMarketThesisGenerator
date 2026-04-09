"""Supabase-backed job manager for persistent thesis generation jobs.

All job state is stored in a Supabase `jobs` table so it survives
server restarts. Uses _RowProxy to expose Supabase dict rows as
attribute-accessible objects (job.id, job.thesis, etc.) matching the
interface that route handlers expect.
"""

import logging
import uuid
from typing import Any, Dict, List, Optional

from supabase import Client, create_client

from api.schemas import JobStatus
from api.serializers import (
    rehydrate_articles,
    rehydrate_docs,
    rehydrate_thesis,
    serialise_job_fields,
)
from core.interfaces.job_manager import IJobManager

logger = logging.getLogger(__name__)

TABLE = "jobs"


class SupabaseJobManager(IJobManager):
    """Persistent job store backed by Supabase."""

    def __init__(self, url: str, anon_key: str):
        self._client: Client = create_client(url, anon_key)
        logger.info("SupabaseJobManager connected")

    def create_job(self, query: str):
        """Create a new job row and return it."""
        job_id = uuid.uuid4().hex[:12]
        row: Dict[str, Any] = {
            "id": job_id,
            "query": query,
            "status": JobStatus.PENDING.value,
            "progress": None,
            "thesis": None,
            "articles": [],
            "error": None,
            "refinement_count": 0,
            "refinement_status": "refining",
            "feedback_history": [],
            "execution_log": [],
            "retrieved_docs": [],
        }
        self._client.table(TABLE).insert(row).execute()
        logger.info(f"Job created in Supabase: {job_id} for query: {query!r}")
        return _RowProxy(row)

    def get_job(self, job_id: str):
        """Fetch a job by ID. Returns None if not found."""
        resp = (
            self._client.table(TABLE)
            .select("*")
            .eq("id", job_id)
            .maybe_single()
            .execute()
        )
        if resp is None or not resp.data:
            return None
        return _RowProxy(resp.data)

    def update_status(
        self, job_id: str, status: JobStatus, progress: Optional[str] = None
    ):
        """Update job status and optional progress message."""
        update: Dict[str, Any] = {"status": status.value}
        if progress is not None:
            update["progress"] = progress
        self._client.table(TABLE).update(update).eq("id", job_id).execute()

    def update_job(self, job_id: str, **fields):
        """Persist arbitrary field updates to the job row."""
        payload = serialise_job_fields(**fields)
        if payload:
            self._client.table(TABLE).update(payload).eq("id", job_id).execute()

    def list_jobs(self) -> list:
        """List all jobs, most recent first."""
        resp = (
            self._client.table(TABLE)
            .select("*")
            .order("created_at", desc=True)
            .execute()
        )
        return [_RowProxy(row) for row in resp.data]


class _RowProxy:
    """Exposes a Supabase row dict as attributes so route handlers can use
    job.id, job.thesis, etc. instead of job["id"], job["thesis"].

    Deserialization (JSON dicts → domain objects) is delegated to
    api.serializers so this class only handles attribute mapping.
    """

    def __init__(self, data: dict):
        self._data = data
        self.id: str = data["id"]
        self.query: str = data["query"]
        self.status: JobStatus = JobStatus(data["status"])
        self.progress: Optional[str] = data.get("progress")
        self.error: Optional[str] = data.get("error")
        self.refinement_count: int = data.get("refinement_count", 0)
        self.refinement_status: str = data.get("refinement_status", "refining")
        self.feedback_history: list = data.get("feedback_history", [])
        self.execution_log: list = data.get("execution_log", [])
        self.thesis = rehydrate_thesis(data.get("thesis"))
        self.articles = rehydrate_articles(data.get("articles", []))
        self.retrieved_docs = rehydrate_docs(data.get("retrieved_docs", []))
