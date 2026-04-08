"""Supabase-backed job manager for persistent thesis generation jobs.

All job state is stored in a Supabase `jobs` table so it survives
server restarts. Uses _RowProxy to expose Supabase dict rows as
attribute-accessible objects (job.id, job.thesis, etc.) matching the
interface that route handlers expect.
"""

import logging
import uuid
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from supabase import Client, create_client

from api.schemas import JobStatus
from core.models.article import Article
from core.models.thesis import StructuredThesis

logger = logging.getLogger(__name__)

TABLE = "jobs"


class SupabaseJobManager:
    """Persistent job store backed by Supabase."""

    def __init__(self, url: str, anon_key: str):
        self._client: Client = create_client(url, anon_key)
        logger.info("SupabaseJobManager connected")

    def create_job(self, query: str):
        """Create a new job row and return it."""
        job_id = uuid.uuid4().hex[:12]
        row = {
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
        if not resp.data:
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
        """Persist arbitrary field updates to the job row.

        Handles serialisation of dataclass fields (thesis, articles)
        and LangChain Documents (retrieved_docs) to JSON-safe dicts.
        """
        payload: Dict[str, Any] = {}
        for key, value in fields.items():
            if key == "thesis" and value is not None:
                payload[key] = asdict(value)
            elif key == "articles":
                payload[key] = [asdict(a) for a in value]
            elif key == "status" and isinstance(value, JobStatus):
                payload[key] = value.value
            elif key == "retrieved_docs":
                payload[key] = _serialise_docs(value)
            else:
                payload[key] = value
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


def _serialise_docs(docs: list) -> list:
    """Convert LangChain Documents to JSON-safe dicts."""
    result = []
    for d in docs:
        if hasattr(d, "page_content"):
            result.append({"page_content": d.page_content, "metadata": d.metadata})
        else:
            result.append(d)
    return result


class _RowProxy:
    """Exposes a Supabase row dict as attributes so route handlers can use
    job.id, job.thesis, etc. instead of job["id"], job["thesis"].
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

        # Rehydrate thesis from stored JSON dict → StructuredThesis
        raw_thesis = data.get("thesis")
        if raw_thesis and isinstance(raw_thesis, dict):
            self.thesis: Optional[StructuredThesis] = StructuredThesis(**raw_thesis)
        else:
            self.thesis = None

        # Rehydrate articles from stored JSON dicts → Article objects
        raw_articles = data.get("articles", [])
        self.articles: List[Article] = []
        for a in raw_articles:
            if isinstance(a, dict):
                self.articles.append(Article(**a))

        # Retrieved docs stay as plain dicts (used by refinement agent)
        self.retrieved_docs: list = data.get("retrieved_docs", [])
