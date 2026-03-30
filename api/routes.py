"""FastAPI routes for thesis generation and refinement."""

import asyncio
import json
import logging
from typing import List

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sse_starlette.sse import EventSourceResponse

from api.deps import get_container, get_job_manager
from api.job_manager import Job, JobManager
from api.schemas import (
    ArticleResponse,
    JobResponse,
    JobStatus,
    RefinementRequest,
    RefinementResponse,
    ThesisRequest,
    ThesisResponse,
)
from core.agents.hallucination_detector import HallucinationDetector
from dependency_injection.container import ServiceContainer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


# --- Helpers ---

def _thesis_to_response(thesis) -> ThesisResponse:
    """Convert StructuredThesis dataclass to Pydantic response."""
    return ThesisResponse(
        key_themes=thesis.key_themes,
        risks=thesis.risks,
        investment_signals=thesis.investment_signals,
        sources=thesis.sources,
        raw_output=thesis.raw_output,
        opportunity_score=thesis.opportunity_score,
        confidence_level=thesis.confidence_level,
        recommendation=thesis.recommendation,
        key_risk_factors=thesis.key_risk_factors,
    )


def _articles_to_response(articles) -> List[ArticleResponse]:
    """Convert Article dataclasses to Pydantic responses."""
    return [
        ArticleResponse(title=a.title, source=a.source, url=a.url)
        for a in articles
    ]


def _job_to_response(job: Job) -> JobResponse:
    """Convert internal Job to API response."""
    return JobResponse(
        job_id=job.id,
        status=job.status,
        progress=job.progress,
        query=job.query,
        thesis=_thesis_to_response(job.thesis) if job.thesis else None,
        articles=_articles_to_response(job.articles),
        error=job.error,
    )


# --- Background pipeline ---

def _run_thesis_pipeline(job_id: str, container: ServiceContainer, jm: JobManager):
    """Run the full thesis generation pipeline in a background thread.

    Container and JobManager are passed explicitly since Depends() only works
    in route handler signatures, not in background tasks.
    """
    job = jm.get_job(job_id)
    if not job:
        return

    try:
        # Step 1: Fetch articles
        jm.update_status(job_id, JobStatus.FETCHING_ARTICLES, "Fetching fintech news...")
        ingestion = container.get_ingestion_service()
        articles = ingestion.fetch_articles(query="fintech", limit=5)
        if not articles:
            jm.update_status(job_id, JobStatus.FAILED, "No articles found")
            job.error = "No articles found from RSS feeds"
            return
        job.articles = articles

        # Step 2: Build vectorstore
        jm.update_status(job_id, JobStatus.BUILDING_VECTORSTORE, "Building FAISS vectorstore...")
        documents = ingestion.convert_to_documents(articles)
        retrieval = container.get_retrieval_service()
        retrieval.build_vectorstore(documents)

        # Step 3: Retrieve relevant docs
        jm.update_status(job_id, JobStatus.RETRIEVING, "Retrieving relevant context...")
        docs = retrieval.retrieve(job.query, k=5)
        if not docs:
            jm.update_status(job_id, JobStatus.FAILED, "No relevant documents found")
            job.error = "No relevant documents found for query"
            return
        job.retrieved_docs = docs

        # Step 4: Generate thesis
        jm.update_status(job_id, JobStatus.GENERATING, "Generating market thesis...")
        thesis_service = container.get_thesis_service()
        thesis = thesis_service.generate_thesis(job.query, docs)
        job.thesis = thesis

        jm.update_status(job_id, JobStatus.COMPLETED, "Thesis generated successfully")
        logger.info(f"Job {job_id} completed successfully")

    except Exception as e:
        logger.exception(f"Job {job_id} failed")
        job.error = str(e)
        jm.update_status(job_id, JobStatus.FAILED, f"Error: {e}")


# --- Endpoints ---

@router.post("/thesis", status_code=202, response_model=JobResponse)
def create_thesis(
    request: ThesisRequest,
    background_tasks: BackgroundTasks,
    container: ServiceContainer = Depends(get_container),
    jm: JobManager = Depends(get_job_manager),
):
    """Start a thesis generation job. Returns immediately with job_id."""
    job = jm.create_job(request.query)
    background_tasks.add_task(_run_thesis_pipeline, job.id, container, jm)
    return _job_to_response(job)


@router.get("/thesis/{job_id}", response_model=JobResponse)
def get_thesis(job_id: str, jm: JobManager = Depends(get_job_manager)):
    """Get current status and result of a thesis job."""
    job = jm.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return _job_to_response(job)


@router.get("/thesis/{job_id}/stream")
async def stream_thesis_progress(
    job_id: str,
    jm: JobManager = Depends(get_job_manager),
):
    """SSE endpoint for real-time progress updates on a thesis job."""
    job = jm.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_generator():
        # stream updates as they come in, until job is completed or failed
        last_status = None
        while True:
            # get latest job status from manager; this is where the "streaming" happens
            # as client keeps connection open and we keep
            # yielding updates whenever there's a status change until job is done
            job = jm.get_job(job_id)
            if not job:
                break

            # Only send when status changes
            if job.status != last_status:
                last_status = job.status
                yield {
                    "event": "progress",
                    "data": json.dumps({
                        "status": job.status.value,
                        "progress": job.progress,
                    }),
                }

            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
                # Send final result
                yield {
                    "event": "done",
                    "data": _job_to_response(job).model_dump_json(),
                }
                break

            # sleep briefly after continuous yielding, to avoid tight loop
            # of spamming database/manager with n requests/second.
            await asyncio.sleep(0.5)

    # make an HTTP connection and keep it open from the first invoke of event_generator()
    return EventSourceResponse(event_generator())


@router.post("/thesis/{job_id}/refine", response_model=RefinementResponse)
def refine_thesis(
    job_id: str,
    request: RefinementRequest,
    container: ServiceContainer = Depends(get_container),
    jm: JobManager = Depends(get_job_manager),
):
    """Refine an existing thesis with user feedback."""
    job = jm.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.thesis:
        raise HTTPException(status_code=400, detail="Thesis not yet generated")
    if job.refinement_status == "escalated":
        raise HTTPException(status_code=400, detail="Max refinements reached")

    try:
        graph, langfuse_handler = container.get_refinement_graph()
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))

    # Build LangGraph state
    langgraph_state = {
        "topic": job.query,
        "documents": job.retrieved_docs,
        "current_thesis": job.thesis,
        "feedback_history": job.feedback_history + [request.feedback],
        "refinement_count": job.refinement_count,
        "status": "refining",
        "execution_log": job.execution_log,
        "messages": [],
    }

    # Run the refinement agent
    invoke_config = {"callbacks": [langfuse_handler]} if langfuse_handler else {}
    result_state = graph.invoke(langgraph_state, config=invoke_config)

    # Analyze for hallucinations
    detector = HallucinationDetector()
    detector.analyze(result_state.get("messages", []))

    # Update job state
    job.thesis = result_state["current_thesis"]
    job.refinement_count = result_state["refinement_count"]
    job.refinement_status = result_state["status"]
    job.feedback_history = result_state["feedback_history"]
    job.execution_log = result_state.get("execution_log", [])

    return RefinementResponse(
        job_id=job.id,
        refinement_count=job.refinement_count,
        status=job.refinement_status,
        thesis=_thesis_to_response(job.thesis),
        execution_log=job.execution_log,
    )


@router.get("/jobs", response_model=List[JobResponse])
def list_jobs(jm: JobManager = Depends(get_job_manager)):
    """List all thesis generation jobs."""
    return [_job_to_response(j) for j in jm.list_jobs()]


@router.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok"}
