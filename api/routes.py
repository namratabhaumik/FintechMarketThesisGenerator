"""FastAPI routes for thesis generation and refinement.

Synchronous flow mirroring the Streamlit app: a thesis request embeds the
query once, retrieves from the persistent corpus, generates, and persists the
completed job in one request. Failure means no row is created - in-flight
state never reaches the DB, so a crash cannot strand a job.

/theses is the collection; refinement rounds and approval are sub-resources 
of one thesis. Errors carry {code, message} in the detail.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response

from api.deps import get_container, get_job_manager
from api.security import (
    GENERATE_LIMIT,
    REFINE_LIMIT,
    limiter,
    require_api_key,
)
from api.schemas import (
    JobResponse,
    JobStatus,
    RefinementRequest,
    RefinementStatus,
    RelatedThesisResponse,
    SourceResponse,
    ThesisRequest,
    ThesisResponse,
    ThesisSummaryResponse,
)
from config.settings import FEEDBACK_OPTIONS
from core.agents.hallucination_detector import HallucinationDetector
from core.interfaces.job_manager import IJobManager
from core.services.episodic_recall import RECALL_MIN_SIMILARITY
from dependency_injection.container import ServiceContainer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


# --- Helpers ---

def _error(status_code: int, code: str, message: str) -> HTTPException:
    """Build an HTTPException whose detail is a machine-readable {code, message}."""
    return HTTPException(status_code=status_code, detail={"code": code, "message": message})


def _thesis_to_response(thesis) -> ThesisResponse:
    """Convert StructuredThesis dataclass to Pydantic response."""
    as_of = thesis.confidence_as_of
    return ThesisResponse(
        key_themes=thesis.key_themes,
        risks=thesis.risks,
        investment_signals=thesis.investment_signals,
        sources=thesis.sources,
        raw_output=thesis.raw_output,
        opportunity_score=thesis.opportunity_score,
        confidence_level=thesis.confidence_level,
        confidence_as_of=as_of.isoformat() if as_of else None,
        recommendation=thesis.recommendation,
        key_risk_factors=thesis.key_risk_factors,
    )


def _sources_from_docs(docs) -> List[SourceResponse]:
    """Each source article once, in relevance order (mirrors the Streamlit expander).

    rehydrate_docs passes malformed stored values through as-is, so a doc with
    unusable metadata is skipped rather than failing the whole response.
    """
    seen_urls = set()
    sources = []
    for doc in docs:
        try:
            meta = getattr(doc, "metadata", None) or {}
            url = meta.get("url", "")
            if url and url in seen_urls:
                continue
            seen_urls.add(url)
            sources.append(
                SourceResponse(
                    title=meta.get("title") or "Untitled",
                    url=url or None,
                    published_at=meta.get("published_at"),
                )
            )
        except Exception:
            logger.warning(f"Skipping doc with unusable metadata: {doc!r:.80}")
    return sources


async def _related_for(job, jm: IJobManager) -> List[RelatedThesisResponse]:
    """Episodic recall for a job, from its stored query embedding. Never raises.

    Ranking happens in the DB (match_jobs / pgvector).
    """
    embedding = getattr(job, "query_embedding", None)
    if not embedding:
        return []
    try:
        rows = await jm.match_jobs(embedding, job.id, min_similarity=RECALL_MIN_SIMILARITY)
        return [RelatedThesisResponse(**r) for r in rows]
    except Exception:
        logger.exception("Failed to compute related past theses")
        return []


async def _job_to_response(
    job, jm: IJobManager, hallucination: Optional[dict] = None
) -> JobResponse:
    """Convert internal job to the full API representation."""
    return JobResponse(
        job_id=job.id,
        query=job.query,
        status=job.status,
        created_at=job.created_at,
        error=job.error,
        thesis=_thesis_to_response(job.thesis) if job.thesis else None,
        thesis_history=[_thesis_to_response(t) for t in job.thesis_history],
        refinement_count=job.refinement_count,
        refinement_status=job.refinement_status,
        feedback_history=job.feedback_history,
        execution_log=job.execution_log,
        approved_at=job.approved_at,
        sources=_sources_from_docs(job.retrieved_docs),
        related_theses=await _related_for(job, jm),
        hallucination=hallucination,
    )


async def _get_job_or_404(jm: IJobManager, job_id: str):
    job = await jm.get_job(job_id)
    if not job:
        raise _error(404, "job_not_found", f"No thesis job with id '{job_id}'")
    return job


# --- Endpoints ---

@router.post(
    "/theses",
    status_code=201,
    response_model=JobResponse,
    dependencies=[Depends(require_api_key)],
    tags=["theses"],
)
@limiter.limit(GENERATE_LIMIT)
async def create_thesis(
    request: Request,
    payload: ThesisRequest,
    response: Response,
    container: ServiceContainer = Depends(get_container),
    jm: IJobManager = Depends(get_job_manager),
):
    """Generate a thesis synchronously and persist the completed job.

    The job row is created only after generation succeeds, so failures
    surface as HTTP errors rather than stranded rows. The blocking pipeline
    (embed / retrieve / generate) runs in a worker thread via asyncio.to_thread
    so it never blocks the event loop.
    """
    query = payload.query

    # Embed the query once and reuse it: MMR retrieval searches the corpus
    # with it, and episodic recall stores it to rank past runs. On failure,
    # retrieval re-embeds internally and recall is skipped.
    query_embedding = None
    try:
        query_embedding = await asyncio.to_thread(
            lambda: container.get_embedding_model().get_embeddings().embed_query(query)
        )
    except Exception:
        logger.exception("Failed to embed query; retrieval will re-embed")

    try:
        docs = await asyncio.to_thread(
            lambda: container.get_retrieval_service().retrieve(
                query, k=5, query_embedding=query_embedding
            )
        )
    except Exception:
        logger.exception("Retrieval failed")
        raise _error(500, "retrieval_failed", "Could not retrieve documents from the corpus")
    if not docs:
        raise _error(
            422,
            "no_relevant_documents",
            "No relevant documents found in the corpus for this query",
        )

    try:
        thesis = await asyncio.to_thread(
            lambda: container.get_thesis_service().generate_thesis(query, docs)
        )
    except Exception:
        logger.exception("Thesis generation failed")
        raise _error(502, "generation_failed", "The language model failed to generate a thesis")

    try:
        job = await jm.create_job(query)
        await jm.update_job(
            job.id,
            status=JobStatus.COMPLETED,
            thesis=thesis,
            retrieved_docs=docs,
            refinement_count=0,
            refinement_status=RefinementStatus.NOT_APPLICABLE,  # until the user refines
            feedback_history=[],
            execution_log=[],
            thesis_history=[],
            query_embedding=query_embedding,
        )
        # Re-fetch so the response carries DB-populated fields (created_at).
        created = await jm.get_job(job.id)
    except Exception:
        logger.exception("Failed to persist thesis job")
        raise _error(500, "persistence_failed", "Thesis was generated but could not be saved")

    response.headers["Location"] = f"/api/theses/{job.id}"
    return await _job_to_response(created, jm)


@router.get("/theses", response_model=List[ThesisSummaryResponse], tags=["theses"])
async def list_theses(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    status: Optional[RefinementStatus] = Query(
        None, description="Filter by refinement_status (e.g. 'refining' for the resume picker)"
    ),
    jm: IJobManager = Depends(get_job_manager),
):
    """List thesis jobs, most recent first (slim representations).

    Pagination + optional status filter applied in db
    """
    status_value = status.value if status else None
    jobs = await jm.list_jobs(limit=limit, offset=offset, status=status_value)
    return [
        ThesisSummaryResponse(
            job_id=j.id,
            query=j.query,
            status=j.status,
            created_at=j.created_at,
            refinement_count=j.refinement_count,
            refinement_status=j.refinement_status,
            approved_at=j.approved_at,
            opportunity_score=j.thesis.opportunity_score if j.thesis else None,
            recommendation=j.thesis.recommendation if j.thesis else None,
        )
        for j in jobs
    ]


@router.get("/theses/{job_id}", response_model=JobResponse, tags=["theses"])
async def get_thesis(job_id: str, jm: IJobManager = Depends(get_job_manager)):
    """Full state of one thesis job (rehydrates everything the UI shows)."""
    job = await _get_job_or_404(jm, job_id)
    return await _job_to_response(job, jm)


@router.post(
    "/theses/{job_id}/refinements",
    response_model=JobResponse,
    dependencies=[Depends(require_api_key)],
    tags=["theses"],
)
@limiter.limit(REFINE_LIMIT)
async def create_refinement(
    request: Request,
    job_id: str,
    payload: RefinementRequest,
    container: ServiceContainer = Depends(get_container),
    jm: IJobManager = Depends(get_job_manager),
):
    """Run one refinement round and return the updated thesis state.

    The LangGraph agent invocation is offloaded to a worker thread so its
    blocking LLM calls don't block the event loop.
    """
    job = await _get_job_or_404(jm, job_id)
    if not job.thesis:
        raise _error(409, "thesis_not_generated", "Thesis not yet generated")
    if job.approved_at:
        raise _error(409, "already_approved", "An approved thesis cannot be refined")
    if job.refinement_status == RefinementStatus.ESCALATED:
        raise _error(409, "max_refinements_reached", "Max refinements reached")

    try:
        graph, langfuse_handler = await asyncio.to_thread(container.get_refinement_graph)
    except NotImplementedError as e:
        raise _error(501, "refinement_not_supported", str(e))

    # Snapshot the current thesis into history before refining, so the
    # Previous Versions panel keeps every round (mirrors the Streamlit flow).
    thesis_history = list(job.thesis_history) + [job.thesis]

    langgraph_state = {
        "topic": job.query,
        "documents": job.retrieved_docs,
        "current_thesis": job.thesis,
        "feedback_history": job.feedback_history + [payload.feedback],
        "refinement_count": job.refinement_count,
        "status": "refining",
        "execution_log": job.execution_log,
        "messages": [],
    }

    invoke_config = {"callbacks": [langfuse_handler]} if langfuse_handler else {}
    try:
        result_state = await asyncio.to_thread(
            graph.invoke, langgraph_state, config=invoke_config
        )
    except NotImplementedError as e:
        raise _error(501, "refinement_not_supported", str(e))
    except Exception:
        logger.exception("Error during thesis refinement")
        raise _error(502, "refinement_failed", "The refinement agent failed to run")

    detector = HallucinationDetector()
    hallucination = detector.analyze(result_state.get("messages", []))

    try:
        await jm.update_job(
            job_id,
            thesis=result_state["current_thesis"],
            thesis_history=thesis_history,
            refinement_count=result_state["refinement_count"],
            refinement_status=result_state["status"],
            feedback_history=result_state["feedback_history"],
            execution_log=result_state.get("execution_log", []),
        )
        updated = await jm.get_job(job_id)
    except Exception:
        logger.exception("Failed to persist refinement")
        raise _error(500, "persistence_failed", "Refinement ran but could not be saved")

    return await _job_to_response(updated, jm, hallucination=hallucination)


@router.put(
    "/theses/{job_id}/approval",
    response_model=JobResponse,
    dependencies=[Depends(require_api_key)],
    tags=["theses"],
)
async def approve_thesis(job_id: str, jm: IJobManager = Depends(get_job_manager)):
    """Approve a thesis (idempotent: re-approving returns the existing state).

    Approval is terminal: the approval time is stamped and the run is marked
    "refined" so it no longer appears as resumable.
    """
    job = await _get_job_or_404(jm, job_id)
    if not job.thesis:
        raise _error(409, "thesis_not_generated", "Thesis not yet generated")
    if not job.approved_at:
        ts = datetime.now(timezone.utc).isoformat()
        try:
            await jm.update_job(
                job_id, approved_at=ts, refinement_status=RefinementStatus.REFINED
            )
            job = await jm.get_job(job_id)
        except Exception:
            logger.exception("Failed to record approval")
            raise _error(500, "persistence_failed", "Approval could not be saved")
        logger.info(f"Thesis approved at {ts} (job {job_id})")
    return await _job_to_response(job, jm)


@router.get("/feedback-options", response_model=List[str], tags=["meta"])
def get_feedback_options():
    """The fixed set of refinement feedback reasons the UI offers."""
    return FEEDBACK_OPTIONS


@router.get("/health", tags=["meta"])
@limiter.exempt
def health_check():
    """Health check endpoint. Exempt from limits so a global ceiling
    (RATE_LIMIT_DEFAULT) never trips the platform's health probes."""
    return {"status": "ok"}