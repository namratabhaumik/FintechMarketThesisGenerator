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

from api.auth import (
    AuthUser,
    get_current_user,
    get_user_annotation_manager,
    get_user_job_manager,
    require_admin,
)
from api.deps import get_container
from api.supabase_annotation_manager import SupabaseAnnotationManager
from api.security import (
    GENERATE_LIMIT,
    REFINE_LIMIT,
    limiter,
)
from api.schemas import (
    AnnotationAuthor,
    AnnotationCreateRequest,
    AnnotationResolveRequest,
    AnnotationResponse,
    AnnotationUpdateRequest,
    JobResponse,
    RefinementRequest,
    RefinementStatus,
    RelatedThesisResponse,
    ShareCreateRequest,
    ShareResponse,
    SourceResponse,
    ThesisRequest,
    ThesisResponse,
    ThesisSummaryResponse,
)
from config.settings import FEEDBACK_OPTIONS
from core.agents.hallucination_detector import HallucinationDetector
from core.interfaces.job_manager import IJobManager
from core.services.thesis_generator_service import _tag_strengths_from_documents
from core.utils.observability import trace_request
from dependency_injection.container import ServiceContainer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

# Cosine-similarity floor for episodic recall (passed to the match_jobs RPC).
# Query-to-query comparison, so it depends on the embedding model only:
# 0.86 = same-topic + related sub-topics, drops same-domain-different-topic.
RECALL_MIN_SIMILARITY = 0.86


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
        summary_source=thesis.summary_source,
        summary_status=thesis.summary_status,
        refusal_reason=thesis.refusal_reason,
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
                    # Dedupe keeps the first chunk per URL in MMR order, i.e.
                    # the article's most relevant selected chunk - its
                    # similarity stands for the source.
                    similarity=meta.get("similarity"),
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
        created_at=job.created_at,
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
        user_id=getattr(job, "user_id", None),
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
    tags=["theses"],
)
@limiter.limit(GENERATE_LIMIT)
async def create_thesis(
    request: Request,
    payload: ThesisRequest,
    response: Response,
    container: ServiceContainer = Depends(get_container),
    jm: IJobManager = Depends(get_user_job_manager),
):
    """Generate a thesis synchronously and persist the completed job.

    The job row is created only after generation succeeds, so failures
    surface as HTTP errors rather than stranded rows. Embed/retrieve are
    CPU-bound (ONNX embeddings, in-process vector search) and run in a worker
    thread via asyncio.to_thread; generation calls Gemini natively via
    async LLM clients, so it awaits directly without occupying a thread.
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

    def _retrieve_and_select():
        # `docs` is the wide analytics pool (distinct articles); `summary_docs`
        # is the MMR-narrowed diverse subset the LLM actually reads. Both are
        # persisted so a later refinement reuses the exact same summary docs.
        # Run together in the worker thread (retrieval is I/O, MMR is CPU) so the
        # event loop never blocks and one guard covers both. select_diverse([])
        # returns [], so the empty-corpus case falls through to the 422 below.
        svc = container.get_retrieval_service()
        wide = svc.retrieve(query, query_embedding=query_embedding)
        return wide, svc.select_diverse(wide, query_embedding)

    try:
        docs, summary_docs = await asyncio.to_thread(_retrieve_and_select)
    except Exception as err:
        logger.exception("Retrieval failed")
        raise _error(500, "retrieval_failed", "Could not retrieve documents from the corpus") from err
    if not docs:
        raise _error(
            422,
            "no_relevant_documents",
            "No relevant documents found in the corpus for this query",
        )

    # A thesis is only meaningful if the retrieved evidence grounds all three
    # Silver dimensions. If any of themes/risks/signals has no tag across the
    # retrieved chunks, the score would fall back to a hardcoded neutral and the
    # empty dimension would render blank. Refuse before the LLM call & log which 
    # dimensions were missing. Uses generation's own tag-strength computation, 
    # so passing this guard means generate_thesis will produce a non-empty tag 
    # list for every dimension.
    signal_strength, theme_strength, risk_strength = _tag_strengths_from_documents(docs)
    missing = [
        dim
        for dim, strength in (
            ("themes", theme_strength),
            ("risks", risk_strength),
            ("signals", signal_strength),
        )
        if strength == 0
    ]
    if missing:
        logger.warning(
            "Taxonomy gap: query %r retrieved evidence with no %s tags; refusing "
            "to build a partial thesis. Sources: %s",
            query,
            "/".join(missing),
            [d.metadata.get("url") for d in docs if getattr(d, "metadata", None)],
        )
        raise _error(
            422,
            "insufficient_evidence",
            "The sources retrieved for this specific query don't span themes, "
            "risks, and investment signals together. Try broadening the query.",
        )

    # One Langfuse trace per request: the summarize LLM call (and any gateway
    # retries) nest under it. Langfuse computes per-model dollar cost natively.
    with trace_request("create_thesis", input=query) as trace:
        try:
            thesis = await container.get_thesis_service().generate_thesis(
                query, docs, summary_documents=summary_docs
            )
        except Exception as err:
            logger.exception("Thesis generation failed")
            raise _error(502, "generation_failed", "The language model failed to generate a thesis") from err

        try:
            # One atomic insert with the full completed state: a failure (or crash)
            # here creates NO row, so a half-written job can never appear in the
            # library.
            job = await jm.create_job(
                query,
                thesis=thesis,
                retrieved_docs=docs,
                summary_docs=summary_docs,
                refinement_count=0,
                refinement_status=RefinementStatus.NOT_APPLICABLE,  # until the user refines
                feedback_history=[],
                execution_log=[],
                thesis_history=[],
                query_embedding=query_embedding,
            )
            # Re-fetch so the response carries DB-populated fields (created_at).
            created = await jm.get_job(job.id)
        except Exception as err:
            logger.exception("Failed to persist thesis job")
            raise _error(500, "persistence_failed", "Thesis was generated but could not be saved") from err

        response.headers["Location"] = f"/api/theses/{job.id}"
        trace.annotate(job_id=str(job.id), num_sources=len(docs))
        return await _job_to_response(created, jm)


@router.get("/theses", response_model=List[ThesisSummaryResponse], tags=["theses"])
async def list_theses(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    status: Optional[RefinementStatus] = Query(
        None, description="Filter by refinement_status (e.g. 'refining' for the resume picker)"
    ),
    all_users: bool = Query(
        False,
        alias="all",
        description="Admin only: list every user's theses instead of just the caller's",
    ),
    user: AuthUser = Depends(get_current_user),
    jm: IJobManager = Depends(get_user_job_manager),
):
    """List thesis jobs, most recent first (slim representations).

    Scoped to the caller's own jobs by default. An admin may pass all=true to
    list every user's jobs (the RLS admin policy already grants the read; this
    just drops the explicit per-owner filter). A non-admin asking for all=true
    is refused rather than silently narrowed.

    Pagination + optional status filter applied in db.
    """
    if all_users and user.role != "admin":
        raise _error(403, "forbidden", "Admin role required to list all users' theses")
    # Own view: scope to the caller. Admin all view: every OTHER user's rows,
    # excluding the caller's own so it doesn't duplicate their personal library
    # above it.
    owner_filter = None if all_users else user.id
    exclude_filter = user.id if all_users else None
    status_value = status.value if status else None
    jobs = await jm.list_jobs(
        limit=limit,
        offset=offset,
        status=status_value,
        user_id=owner_filter,
        exclude_user_id=exclude_filter,
    )
    return [
        ThesisSummaryResponse(
            job_id=j.id,
            query=j.query,
            created_at=j.created_at,
            refinement_count=j.refinement_count,
            refinement_status=j.refinement_status,
            approved_at=j.approved_at,
            opportunity_score=j.thesis.opportunity_score if j.thesis else None,
            recommendation=j.thesis.recommendation if j.thesis else None,
            user_id=getattr(j, "user_id", None),
        )
        for j in jobs
    ]


@router.get("/theses/{job_id}", response_model=JobResponse, tags=["theses"])
async def get_thesis(job_id: str, jm: IJobManager = Depends(get_user_job_manager)):
    """Full state of one thesis job (rehydrates everything the UI shows)."""
    job = await _get_job_or_404(jm, job_id)
    return await _job_to_response(job, jm)


@router.post(
    "/theses/{job_id}/refinements",
    response_model=JobResponse,
    tags=["theses"],
)
@limiter.limit(REFINE_LIMIT)
async def create_refinement(
    request: Request,
    job_id: str,
    payload: RefinementRequest,
    container: ServiceContainer = Depends(get_container),
    jm: IJobManager = Depends(get_user_job_manager),
):
    """Run one refinement round and return the updated thesis state.

    The LangGraph agent runs via ainvoke: its LLM calls use async Gemini
    clients natively, so the graph awaits directly without blocking a thread.
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
        raise _error(501, "refinement_not_supported", str(e)) from e

    langgraph_state = {
        "topic": job.query,
        "documents": job.retrieved_docs,
        "summary_documents": job.summary_docs,
        "current_thesis": job.thesis,
        "feedback_history": job.feedback_history + [payload.feedback],
        "refinement_count": job.refinement_count,
        "status": "refining",
        "execution_log": job.execution_log,
        "messages": [],
    }

    # One Langfuse trace per refinement round: the planner LLM (via invoke_config)
    # and the tool-driven rewrite (via the handler attached to GeminiLanguageModel)
    # both nest under this span. Langfuse computes per-model dollar cost natively.
    invoke_config = {"callbacks": [langfuse_handler]} if langfuse_handler else {}
    with trace_request("create_refinement", input=job.query) as trace:
        try:
            result_state = await graph.ainvoke(langgraph_state, config=invoke_config)
        except NotImplementedError as e:
            raise _error(501, "refinement_not_supported", str(e)) from e
        except Exception as err:
            logger.exception("Error during thesis refinement")
            raise _error(
                502,
                "refinement_failed",
                "The language model was unavailable. Please try again shortly.",
            ) from err
        trace.annotate(job_id=job_id)

    detector = HallucinationDetector()
    hallucination = detector.analyze(result_state.get("messages", []))

    # Snapshot the pre-refinement thesis into history (the Previous Versions
    # panel) whenever this round actually executed a tool - kept 1:1 with the
    # Execution Trace, even for a round that changed nothing (e.g. a
    # re-refusal). True skips (no tool call, parse error, unknown tool) leave
    # the thesis untouched and are excluded, since nothing ran to snapshot.
    new_thesis = result_state["current_thesis"]
    execution_log = result_state.get("execution_log", [])
    executed_this_round = bool(execution_log) and execution_log[-1].get("status") == "executed"
    if executed_this_round:
        thesis_history = list(job.thesis_history) + [job.thesis]
    else:
        thesis_history = job.thesis_history

    try:
        # Guarded persist (optimistic concurrency): only write if no other
        # request touched the job while the agent ran - an approval landing
        # mid-refinement (Approve stays clickable in the UI) or a second tab
        # refining the same job. A lost race returns False; DB errors raise.
        persisted = await jm.update_job_guarded(
            job_id,
            {"refinement_count": job.refinement_count, "approved_at": None},
            thesis=new_thesis,
            thesis_history=thesis_history,
            refinement_count=result_state["refinement_count"],
            refinement_status=result_state["status"],
            feedback_history=result_state["feedback_history"],
            execution_log=execution_log,
        )
        updated = await jm.get_job(job_id) if persisted else None
    except Exception as err:
        logger.exception("Failed to persist refinement")
        raise _error(500, "persistence_failed", "Refinement ran but could not be saved") from err
    if not persisted:
        raise _error(
            409,
            "conflict",
            "This thesis was approved or refined elsewhere while the refinement "
            "ran; the round was discarded. Reload to see the latest state.",
        )

    return await _job_to_response(updated, jm, hallucination=hallucination)


@router.put(
    "/theses/{job_id}/approval",
    response_model=JobResponse,
    tags=["theses"],
)
async def approve_thesis(job_id: str, jm: IJobManager = Depends(get_user_job_manager)):
    """Approve a thesis (idempotent: re-approving returns the existing state).

    Approval stamps approved_at and is terminal & refinement_status tracks refinement alone.
    """
    job = await _get_job_or_404(jm, job_id)
    if not job.thesis:
        raise _error(409, "thesis_not_generated", "Thesis not yet generated")
    if not job.approved_at:
        ts = datetime.now(timezone.utc).isoformat()
        updates = {"approved_at": ts}
        if job.refinement_status == RefinementStatus.REFINING:
            updates["refinement_status"] = RefinementStatus.REFINED
        try:
            await jm.update_job(job_id, **updates)
            job = await jm.get_job(job_id)
        except Exception as err:
            logger.exception("Failed to record approval")
            raise _error(500, "persistence_failed", "Approval could not be saved") from err
        logger.info(f"Thesis approved at {ts} (job {job_id})")
    return await _job_to_response(job, jm)


@router.delete("/theses/{job_id}", status_code=204, tags=["theses"])
async def delete_thesis(
    job_id: str,
    _admin=Depends(require_admin),
    jm: IJobManager = Depends(get_user_job_manager),
):
    """Delete any user's thesis job (admin only)."""
    await _get_job_or_404(jm, job_id)
    try:
        await jm.delete_job(job_id)
    except Exception as err:
        logger.exception(f"Failed to delete job {job_id}")
        raise _error(500, "deletion_failed", "Thesis could not be deleted") from err


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

# --- Annotations ---------------------------------------------------------
#
# Visibility and write permission are enforced by RLS (sql/annotations.sql),
# not re-implemented here: a query returning nothing means the policies said no.
# These handlers only translate that into HTTP and keep the root/reply shapes
# from mixing, so a bad request fails as 400 rather than a DB constraint error.


def _annotation_to_response(row: dict, profiles: dict) -> AnnotationResponse:
    """Row + profile lookup -> response. A missing profile still renders: a
    brand-new signup can outrun its profile row, and a nameless author must not
    break the thread."""
    author_id = row.get("user_id") or ""
    profile = profiles.get(author_id, {})
    return AnnotationResponse(
        id=row["id"],
        job_id=row["job_id"],
        version=row["version"],
        body=row["body"],
        section=row.get("section"),
        start_offset=row.get("start_offset"),
        end_offset=row.get("end_offset"),
        quote=row.get("quote"),
        parent_id=row.get("parent_id"),
        resolution=row.get("resolution"),
        resolved_at=row.get("resolved_at"),
        resolved_by=row.get("resolved_by"),
        author=AnnotationAuthor(
            user_id=author_id,
            display_name=profile.get("display_name"),
            avatar_url=profile.get("avatar_url"),
        ),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
    )


async def _with_authors(
    am: SupabaseAnnotationManager, rows: list[dict]
) -> List[AnnotationResponse]:
    """Attach display identities in one profiles query for the whole list."""
    profiles = await am.get_profiles(
        [r.get("user_id") for r in rows if r.get("user_id")]
    )
    return [_annotation_to_response(r, profiles) for r in rows]


@router.get(
    "/theses/{job_id}/annotations",
    response_model=List[AnnotationResponse],
    tags=["annotations"],
)
async def list_annotations(
    job_id: str,
    version: Optional[int] = Query(
        None, gt=0, description="Scope to one thesis version"
    ),
    jm: IJobManager = Depends(get_user_job_manager),
    am: SupabaseAnnotationManager = Depends(get_user_annotation_manager),
):
    """Annotations on a thesis, roots and replies together (threaded by the
    client) so a thread never needs a second round trip.

    The job is fetched first so an inaccessible thesis is a clear 404 rather
    than an empty list that reads as "no notes yet".
    """
    await _get_job_or_404(jm, job_id)
    rows = await am.list_annotations(job_id, version)
    return await _with_authors(am, rows)


@router.post(
    "/theses/{job_id}/annotations",
    response_model=AnnotationResponse,
    status_code=201,
    tags=["annotations"],
)
async def create_annotation(
    job_id: str,
    payload: AnnotationCreateRequest,
    jm: IJobManager = Depends(get_user_job_manager),
    am: SupabaseAnnotationManager = Depends(get_user_annotation_manager),
):
    """Add a root annotation (anchored to a passage) or a reply.

    A root pins to the version the client rendered; that version's text is
    frozen once superseded, which is what keeps plain offsets valid. A reply
    inherits its parent's version and carries no anchor of its own.
    """
    job = await _get_job_or_404(jm, job_id)

    if payload.parent_id:
        anchor_fields = (
            payload.section,
            payload.start_offset,
            payload.end_offset,
            payload.quote,
        )
        if any(f is not None for f in anchor_fields):
            raise _error(
                400, "invalid_annotation", "A reply cannot carry its own anchor"
            )
        parent = await am.get_annotation(payload.parent_id)
        if parent is None or parent.get("job_id") != job_id:
            raise _error(404, "not_found", "Parent annotation not found on this thesis")
        if parent.get("parent_id"):
            raise _error(
                400, "invalid_annotation", "Replies cannot be nested under a reply"
            )
        fields = {
            "job_id": job_id,
            "version": parent["version"],
            "parent_id": payload.parent_id,
            "body": payload.body,
        }
    else:
        missing = [
            name
            for name, value in (
                ("version", payload.version),
                ("section", payload.section),
                ("start_offset", payload.start_offset),
                ("end_offset", payload.end_offset),
                ("quote", payload.quote),
            )
            if value is None
        ]
        if missing:
            raise _error(
                400,
                "invalid_annotation",
                f"An anchored annotation requires: {', '.join(missing)}",
            )
        # Reject a version that does not exist yet: history holds the superseded
        # versions and the current one is refinement_count + 1.
        latest = job.refinement_count + 1
        if payload.version > latest:
            raise _error(
                400,
                "invalid_annotation",
                f"Version {payload.version} does not exist",
            )
        fields = {
            "job_id": job_id,
            "version": payload.version,
            "section": payload.section.value,
            "start_offset": payload.start_offset,
            "end_offset": payload.end_offset,
            "quote": payload.quote,
            "body": payload.body,
        }

    try:
        row = await am.create_annotation(**fields)
    except Exception:
        logger.exception("Failed to create annotation")
        raise _error(
            500, "persistence_failed", "Could not save the annotation"
        ) from None
    if row is None:
        # The insert policy refused it: readable thesis, no comment rights.
        raise _error(
            403, "forbidden", "You do not have permission to annotate this thesis"
        )
    return (await _with_authors(am, [row]))[0]


@router.patch(
    "/annotations/{annotation_id}",
    response_model=AnnotationResponse,
    tags=["annotations"],
)
async def update_annotation(
    annotation_id: str,
    payload: AnnotationUpdateRequest,
    am: SupabaseAnnotationManager = Depends(get_user_annotation_manager),
):
    """Edit a comment's text. Author-only - RLS matches no row for anyone else,
    which surfaces here as 403 rather than a silent no-op."""
    row = await am.update_annotation_body(annotation_id, payload.body)
    if row is None:
        raise _error(403, "forbidden", "You can only edit your own comments")
    return (await _with_authors(am, [row]))[0]


@router.post(
    "/annotations/{annotation_id}/resolution",
    status_code=204,
    tags=["annotations"],
)
async def resolve_annotation(
    annotation_id: str,
    payload: AnnotationResolveRequest,
    am: SupabaseAnnotationManager = Depends(get_user_annotation_manager),
):
    """Tick (accepted), cross (rejected), or reopen (null).

    Any participant may resolve someone else's note, which no RLS policy can
    express without also permitting edits to that note's body - so this goes
    through a definer function that writes only the state columns. A rejection's
    reason is posted separately as a reply, authored by whoever rejected it.
    """
    resolution = payload.resolution.value if payload.resolution else None
    try:
        await am.set_resolution(annotation_id, resolution)
    except Exception as exc:
        message = str(exc)
        if "not authorised" in message:
            raise _error(
                403, "forbidden", "You cannot resolve this annotation"
            ) from exc
        if "not found" in message:
            raise _error(
                404, "not_found", "Annotation not found, or it is a reply"
            ) from exc
        logger.exception("Failed to set annotation resolution")
        raise _error(
            500, "persistence_failed", "Could not update the annotation"
        ) from exc
    return Response(status_code=204)


@router.delete("/annotations/{annotation_id}", status_code=204, tags=["annotations"])
async def delete_annotation(
    annotation_id: str,
    am: SupabaseAnnotationManager = Depends(get_user_annotation_manager),
):
    """Delete a comment (replies cascade). RLS scopes this to the author or the
    thesis owner; anything else deletes nothing."""
    await am.delete_annotation(annotation_id)
    return Response(status_code=204)


# --- Sharing -------------------------------------------------------------


@router.get(
    "/theses/{job_id}/shares",
    response_model=List[ShareResponse],
    tags=["sharing"],
)
async def list_shares(
    job_id: str,
    jm: IJobManager = Depends(get_user_job_manager),
    am: SupabaseAnnotationManager = Depends(get_user_annotation_manager),
):
    """Who this thesis is shared with. RLS shows the owner every share and a
    collaborator only their own."""
    await _get_job_or_404(jm, job_id)
    rows = await am.list_shares(job_id)
    profiles = await am.get_profiles(
        [r.get("user_id") for r in rows if r.get("user_id")]
    )
    return [
        ShareResponse(
            job_id=r["job_id"],
            user_id=r["user_id"],
            role=r["role"],
            granted_by=r.get("granted_by"),
            created_at=r.get("created_at"),
            display_name=profiles.get(r["user_id"], {}).get("display_name"),
            avatar_url=profiles.get(r["user_id"], {}).get("avatar_url"),
        )
        for r in rows
    ]


@router.post(
    "/theses/{job_id}/shares",
    response_model=ShareResponse,
    status_code=201,
    tags=["sharing"],
)
async def create_share(
    job_id: str,
    payload: ShareCreateRequest,
    user: AuthUser = Depends(get_current_user),
    jm: IJobManager = Depends(get_user_job_manager),
    am: SupabaseAnnotationManager = Depends(get_user_annotation_manager),
):
    """Grant a user read (viewer) or read+comment (commenter) access.

    Owner-only, enforced by RLS: sharing is not transitive, so a collaborator
    cannot re-share someone else's thesis. Re-sharing to the same user changes
    their role rather than failing.
    """
    await _get_job_or_404(jm, job_id)
    if payload.user_id == user.id:
        raise _error(
            400, "invalid_share", "A thesis is already accessible to its owner"
        )
    try:
        row = await am.create_share(
            job_id, payload.user_id, payload.role.value, user.id
        )
    except Exception:
        logger.exception("Failed to create share")
        raise _error(
            500, "persistence_failed", "Could not share the thesis"
        ) from None
    if row is None:
        raise _error(403, "forbidden", "Only the owner can share this thesis")
    return ShareResponse(
        job_id=row["job_id"],
        user_id=row["user_id"],
        role=row["role"],
        granted_by=row.get("granted_by"),
        created_at=row.get("created_at"),
    )


@router.delete("/theses/{job_id}/shares/{user_id}", status_code=204, tags=["sharing"])
async def delete_share(
    job_id: str,
    user_id: str,
    am: SupabaseAnnotationManager = Depends(get_user_annotation_manager),
):
    """Revoke access. Owner-only via RLS; revoking a share that is not yours to
    revoke deletes nothing."""
    await am.delete_share(job_id, user_id)
    return Response(status_code=204)
