"""Supabase-backed job manager for persistent thesis generation jobs.

All job state is stored in a Supabase `jobs` table so it survives
server restarts. Uses _RowProxy to expose Supabase dict rows as
attribute-accessible objects (job.id, job.thesis, etc.) matching the
interface that route handlers expect.
"""

import logging
import uuid
from typing import Any, Dict, Optional

from postgrest.types import CountMethod
from supabase import AsyncClient

from api.schemas import JobStatus
from api.serializers import (
    rehydrate_docs,
    rehydrate_query_embedding,
    rehydrate_thesis,
    serialise_job_fields,
)
from core.interfaces.job_manager import IJobManager

logger = logging.getLogger(__name__)

TABLE = "jobs"


class SupabaseJobManager(IJobManager):
    """Persistent job store backed by Supabase (async client).

    One client on the single event loop, requests never share 
    an HTTP/2 connection across threads.
    """

    def __init__(self, client: AsyncClient):
        self._client = client
        logger.info("SupabaseJobManager connected")

    async def create_job(self, query: str, **fields):
        """Create a new job row and return it.

        Extra fields (thesis, retrieved_docs, status, ...) are serialised and
        merged over the defaults (a caller holding a finished result persists 
        the whole job in ONE atomic insert); a crash never strand a half-written 
        (pending, thesis-less) row.
        """
        job_id = uuid.uuid4().hex[:12]
        row: Dict[str, Any] = {
            "id": job_id,
            "query": query,
            "status": JobStatus.PENDING.value,
            "progress": None,
            "thesis": None,
            "error": None,
            "refinement_count": 0,
            "refinement_status": "N/A",  # becomes "refining" only once the user refines
            "feedback_history": [],
            "execution_log": [],
            "retrieved_docs": [],
        }
        row.update(serialise_job_fields(**fields))
        await self._client.table(TABLE).insert(row).execute()
        logger.info(f"Job created in Supabase: {job_id} for query: {query!r}")
        return _RowProxy(row)

    async def get_job(self, job_id: str):
        """Fetch a job by ID. Returns None if not found."""
        resp = (
            await self._client.table(TABLE)
            .select("*")
            .eq("id", job_id)
            .maybe_single()
            .execute()
        )
        if resp is None or not resp.data:
            return None
        return _RowProxy(resp.data)

    async def update_status(
        self, job_id: str, status: JobStatus, progress: Optional[str] = None
    ):
        """Update job status and optional progress message."""
        update: Dict[str, Any] = {"status": status.value}
        if progress is not None:
            update["progress"] = progress
        await self._client.table(TABLE).update(update).eq("id", job_id).execute()

    async def update_job(self, job_id: str, **fields):
        """Persist arbitrary field updates to the job row."""
        payload = serialise_job_fields(**fields)
        if payload:
            await self._client.table(TABLE).update(payload).eq("id", job_id).execute()

    async def update_job_guarded(
        self, job_id: str, guards: Dict[str, Any], **fields
    ) -> bool:
        """Persist field updates only if the row still matches `guards`.

        Optimistic concurrency for racing writers (approve during an in-flight
        refinement, or two tabs refining the same job): each guard column must
        still hold its expected value (None means IS NULL), checked atomically
        in the UPDATE's WHERE clause. Returns False when another writer got
        there first - the caller decides how to surface the conflict.

        DB errors intentionally propagate: the caller must tell a failed persist 
        (exception -> 500) apart from a lost race (False -> 409); swallowing 
        errors into False would report an outage as a user conflict.
        """
        payload = serialise_job_fields(**fields)
        if not payload:
            return True
        query = self._client.table(TABLE).update(
            payload, count=CountMethod.exact
        ).eq("id", job_id)
        for column, expected in guards.items():
            if expected is None:
                query = query.is_(column, "null")
            else:
                query = query.eq(column, expected)
        resp = await query.execute()
        # count is authoritative when present; representation rows otherwise.
        updated = bool(getattr(resp, "count", None) or getattr(resp, "data", None))
        if not updated:
            logger.warning(
                f"Guarded update lost the race for job {job_id} (guards={guards})"
            )
        return updated

    async def list_jobs(
        self,
        limit: Optional[int] = None,
        offset: int = 0,
        status: Optional[str] = None,
    ) -> list:
        """List jobs, most recent first, filtering/paginating at the DB.

        limit=None returns all rows (episodic recall needs the full set);
        status filters on refinement_status.
        """
        query = self._client.table(TABLE).select("*").order("created_at", desc=True)
        if status is not None:
            query = query.eq("refinement_status", status)
        if limit is not None:
            # Supabase range() is inclusive on both ends, 0-indexed.
            query = query.range(offset, offset + limit - 1)
        resp = await query.execute()
        return [_RowProxy(row) for row in resp.data]

    async def match_jobs(
        self,
        query_embedding: Any,
        exclude_id: str,
        top_n: int = 3,
        min_similarity: float = 0.86,
    ) -> list:
        """Past runs most similar to query_embedding, ranked in the DB.

        Delegates cosine ranking, the similarity floor, and exclusion of the
        current run + run-less rows to the match_jobs pgvector RPC (episodic
        recall never pulls the whole jobs table into Python).
        """
        # arg name = query_vec; param = query_embedding
        params = {
            "query_vec": query_embedding,
            "exclude_id": exclude_id,
            "match_count": top_n,
            "min_similarity": min_similarity,
        }
        resp = await self._client.rpc("match_jobs", params).execute()
        return resp.data or []


class _RowProxy:
    """Exposes a Supabase row dict as attributes so route handlers can use
    job.id, job.thesis, etc. instead of job["id"], job["thesis"].

    Deserialization (JSON dicts → domain objects) is delegated to
    api.serializers so this class only handles attribute mapping.
    """

    def __init__(self, data: dict):
        self.id: str = data["id"]
        self.query: str = data["query"]
        self.status: JobStatus = JobStatus(data["status"])
        self.progress: Optional[str] = data.get("progress")
        self.error: Optional[str] = data.get("error")
        self.refinement_count: int = data.get("refinement_count", 0)
        self.refinement_status: str = data.get("refinement_status", "N/A")
        self.created_at: Optional[str] = data.get("created_at")
        self.feedback_history: list = data.get("feedback_history", [])
        self.execution_log: list = data.get("execution_log", [])
        self.approved_at: Optional[str] = data.get("approved_at")
        self.query_embedding: Optional[list] = rehydrate_query_embedding(
            data.get("query_embedding")
        )
        self.thesis = rehydrate_thesis(data.get("thesis"))
        self.thesis_history = [
            t for t in (rehydrate_thesis(x) for x in data.get("thesis_history") or [])
            if t is not None
        ]
        self.retrieved_docs = rehydrate_docs(data.get("retrieved_docs", []))
