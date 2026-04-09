"""FastAPI routes for thesis generation and refinement."""

import asyncio
import json
import logging
from typing import List

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sse_starlette.sse import EventSourceResponse

from api.deps import get_container, get_job_manager
from api.thesis_pipeline import ThesisPipelineService
from core.interfaces.job_manager import IJobManager
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


def _job_to_response(job) -> JobResponse:
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


# --- Endpoints ---

@router.post("/thesis", status_code=202, response_model=JobResponse)
def create_thesis(
    request: ThesisRequest,
    background_tasks: BackgroundTasks,
    container: ServiceContainer = Depends(get_container),
    jm: IJobManager = Depends(get_job_manager),
):
    """Start a thesis generation job. Returns immediately with job_id."""
    job = jm.create_job(request.query)
    pipeline = ThesisPipelineService(container, jm)
    background_tasks.add_task(pipeline.run, job.id)
    return _job_to_response(job)


@router.get("/thesis/{job_id}", response_model=JobResponse)
def get_thesis(job_id: str, jm: IJobManager = Depends(get_job_manager)):
    """Get current status and result of a thesis job."""
    job = jm.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return _job_to_response(job)


@router.get("/thesis/{job_id}/stream")
async def stream_thesis_progress(
    job_id: str,
    jm: IJobManager = Depends(get_job_manager),
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
                # Send final result and close the sse session
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
    jm: IJobManager = Depends(get_job_manager),
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

    # Persist updated state to Supabase
    jm.update_job(
        job_id,
        thesis=result_state["current_thesis"],
        refinement_count=result_state["refinement_count"],
        refinement_status=result_state["status"],
        feedback_history=result_state["feedback_history"],
        execution_log=result_state.get("execution_log", []),
    )

    # Re-fetch for consistent response
    updated_job = jm.get_job(job_id)

    return RefinementResponse(
        job_id=updated_job.id,
        refinement_count=updated_job.refinement_count,
        status=updated_job.refinement_status,
        thesis=_thesis_to_response(updated_job.thesis),
        execution_log=updated_job.execution_log,
    )


@router.get("/jobs", response_model=List[JobResponse])
def list_jobs(jm: IJobManager = Depends(get_job_manager)):
    """List all thesis generation jobs."""
    return [_job_to_response(j) for j in jm.list_jobs()]


@router.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok"}
