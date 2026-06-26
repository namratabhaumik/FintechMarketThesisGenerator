# app.py
"""
Streamlit frontend for FinThesis (Refactored with SOLID principles)
Uses dependency injection to manage all dependencies.
"""

import logging
import os
from datetime import datetime, timezone

# Must be set before ONNX Runtime (FastEmbed) is imported to avoid OpenMP conflict on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
from dotenv import load_dotenv

from api.schemas import JobStatus
from api.supabase_job_manager import SupabaseJobManager
from config.settings import AppConfig
from core.agents.hallucination_detector import HallucinationDetector
from core.models.thesis import StructuredThesis
from core.services.episodic_recall import recall_similar
from core.utils.logging import setup_logging
from dependency_injection.container import ServiceContainer

# Load environment variables from .env. override=True so edits 
# take effect on a Clear Cache & Reset (which rebuilds the
# cached container) without a full process restart.
load_dotenv(override=True)

# ---- Setup ----
setup_logging()
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="FinThesis - AI Market Research Assistant", layout="wide")

st.title("FinThesis: Fintech Market Research Assistant")
st.markdown(
    "Generate structured **market theses** from live fintech news using semantic retrieval "
    "and keyword-driven analysis — powered by a local summarizer or an LLM of your choice."
)

# === Refinement Configuration ===
FEEDBACK_OPTIONS = [
    "Too many risks, not enough opportunities",
    "Missing recent market trends",
    "Investment signals are too vague",
    "Opportunity score seems too low",
    "Analysis is too broad, be more specific",
    "Need stronger evidence for key themes",
]

# Cache control for testing
if st.button("🔄 Clear Cache & Reset"):
    st.cache_resource.clear()
    st.session_state.clear()
    st.query_params.clear()
    st.success("Cache cleared! Refresh the page.")
    st.stop()

# Warning about API key
if not os.getenv("GOOGLE_API_KEY"):
    st.warning(
        "No GOOGLE_API_KEY detected in environment. "
        "Gemini calls will fail without a valid API key. (Set GOOGLE_API_KEY in your environment.)"
    )


# === Initialize DI Container (cached for performance) ===
@st.cache_resource
def get_container() -> ServiceContainer:
    """Initialize and cache service container."""
    config = AppConfig.from_env()
    container = ServiceContainer(config)
    logger.info("Service container initialized and cached")
    return container


container = get_container()


# === Refinement Checkpointing in Supabase ===
# st.session_state remains the source of truth for the active
# session. Persisting here just means a server restart between refinement
# rounds doesn't silently lose completed rounds. No-op if Supabase isn't
# configured (config.supabase.enabled is False).


@st.cache_resource
def get_job_manager():
    """Initialize and cache the Supabase job manager, or None if unconfigured."""
    config = AppConfig.from_env()
    if not config.supabase.enabled:
        logger.info("Supabase not configured - refinement checkpointing disabled")
        return None
    return SupabaseJobManager(
        url=config.supabase.url,
        service_role_key=config.supabase.service_role_key,
    )


job_manager = get_job_manager()


def _checkpoint(job_id, **fields) -> None:
    """Persist refinement fields to the job row. Logs and swallows failures."""
    if not job_manager or not job_id:
        return
    try:
        job_manager.update_job(job_id, **fields)
    except Exception:
        logger.exception(f"Failed to checkpoint job {job_id}")


def _persist_approval_once() -> None:
    """Record the approval timestamp on the job the first time the toggle is on.

    Approval is toggle-based (approve only), so we capture just when it happened.
    Guarded by session state so Streamlit reruns don't keep rewriting it.
    """
    if st.session_state.get("approved_at"):
        return
    try:
        ts = datetime.now(timezone.utc).isoformat()
        job_id = st.session_state.get("job_id")
        # Approval is terminal: stamp the time and mark the run "refined" so it
        # drops out of the Resume picker (nothing left to refine).
        _checkpoint(job_id, approved_at=ts, refinement_status="refined")
        st.session_state["approved_at"] = ts
        ref_state = st.session_state.get("refinement_state", {})
        ref_state["status"] = "refined"
        st.session_state["refinement_state"] = ref_state
        logger.info(f"Thesis approved at {ts} (job {job_id})")
    except Exception:
        logger.exception("Failed to record approval")


# 0.55 = same-topic + related sub-topics; drops same-domain-but-different-topic (~0.3-0.45)
RECALL_MIN_SIMILARITY = 0.55


def _compute_related(query_embedding, current_job_id, top_n: int = 3) -> list:
    """Rank past runs by query similarity to the current one (episodic recall).

    Returns lightweight display dicts, excluding the current run and any run
    without a stored embedding.
    """
    if not query_embedding or not job_manager:
        return []
    try:
        jobs = [
            j for j in job_manager.list_jobs()
            if j.id != current_job_id and j.thesis
        ]
        return [
            {
                "job_id": job.id,
                "query": job.query,
                "created_at": job.created_at,
                "score": job.thesis.opportunity_score,
                "recommendation": job.thesis.recommendation,
                "approved": bool(job.approved_at),
                "similarity": round(score, 2),
            }
            for job, score in recall_similar(
                query_embedding, jobs, top_n=top_n, min_score=RECALL_MIN_SIMILARITY
            )
        ]
    except Exception:
        logger.exception("Failed to compute related past theses")
        return []


def _load_job_into_session(job, job_id: str) -> None:
    """Populate session_state from a checkpointed job row."""
    st.session_state["job_id"] = job_id
    st.session_state["generated_thesis"] = job.thesis
    st.session_state["retrieved_docs"] = job.retrieved_docs
    st.session_state["refinement_state"] = {
        "topic": job.query,
        "refinement_count": job.refinement_count,
        "status": job.refinement_status,
        "feedback_history": job.feedback_history,
        "refinement_supported": True,
    }
    st.session_state["execution_log"] = job.execution_log
    st.session_state["refinement_history"] = []
    st.session_state["thesis_count"] = 1
    st.session_state["approved_at"] = job.approved_at
    st.session_state["related_theses"] = _compute_related(job.query_embedding, job_id)
    st.query_params["job_id"] = job_id
    logger.info(f"Loaded job {job_id} from Supabase checkpoint")


def _restore_from_job_id() -> None:
    """Rehydrate session_state from a checkpointed job, if a job_id is in the URL.

    Browser session_state is lost on a new tab, a refresh, or a server
    restart - the job_id query param is the only thing that survives all
    three, so it's the resume key. Runs once: a no-op once generated_thesis
    is already in session_state for this run.
    """
    if "generated_thesis" in st.session_state:
        return
    job_id = st.query_params.get("job_id")
    if not job_id or not job_manager:
        return
    try:
        job = job_manager.get_job(job_id)
    except Exception:
        logger.exception(f"Failed to restore job {job_id}")
        return
    if not job or not job.thesis:
        return
    _load_job_into_session(job, job_id)


def show_resume_picker() -> None:
    """Let the user manually resume a previous session without the exact URL.

    Only the job_id query param survives a fresh tab without it (e.g. a
    bookmark to the bare URL), so this is the fallback resume path: pick a
    past job from Supabase directly instead of needing the link. Only runs
    actively mid-refinement ("refining") appear.
    """
    if "generated_thesis" in st.session_state or not job_manager:
        return
    try:
        jobs = [
            j for j in job_manager.list_jobs()
            if j.thesis and j.refinement_status == "refining"
        ]
    except Exception:
        logger.exception("Failed to list jobs for resume picker")
        return
    if not jobs:
        return

    with st.expander(f"📂 Resume a previous session ({len(jobs)} available)"):
        labels = {
            j.id: (
                f"{j.query} — round {j.refinement_count}/3 ({j.refinement_status}) "
                f"— {(j.created_at or '')[:19]}"
            )
            for j in jobs
        }
        selected_id = st.selectbox(
            "Pick a session to resume:",
            options=list(labels.keys()),
            format_func=lambda jid: labels[jid],
            key="resume_job_select",
        )
        if st.button("Resume", key="resume_job_btn"):
            job = next(j for j in jobs if j.id == selected_id)
            _load_job_into_session(job, selected_id)
            st.rerun()


_restore_from_job_id()


# === UI Helper Functions ===


def show_approval_toggle(thesis_count: int, initial: bool = False) -> bool:
    """Display approval toggle and return its state.

    Args:
        thesis_count: Unique counter for widget key.
        initial: Default on-state - True when the loaded run is already approved,
            so reopening an approved run reflects it instead of showing it as new.

    Returns:
        Boolean indicating if thesis is approved.
    """
    col_toggle, col_spacer = st.columns([1, 2])
    with col_toggle:
        return st.toggle(
            "✅ Approved",
            value=initial,
            key=f"approval_toggle_{thesis_count}"
        )


def show_approved_state(refinement_supported: bool):
    """Display approval confirmation message and optional history.

    Args:
        refinement_supported: Whether refinement is supported for this model.
    """
    st.info("✅ This thesis has been approved. No further refinements needed.")

    history = st.session_state.get("refinement_history", [])
    if history and refinement_supported:
        st.divider()
        show_history(history)


def show_escalation_message():
    """Display message when max refinements reached."""
    st.warning(
        "🔒 **Max refinements reached (3/3)**. "
        "Please refine your original query for a fresh analysis."
    )


def show_refinement_controls(refinement_count: int) -> bool:
    """Display refinement controls (feedback select + refine button).

    Args:
        refinement_count: Current refinement iteration.

    Returns:
        Boolean indicating if refine button was clicked.
    """
    st.subheader(f"📝 Refine Thesis (refinement {refinement_count}/3)")

    selected_feedback = st.multiselect(
        "Select areas to improve:",
        options=FEEDBACK_OPTIONS,
        key=f"feedback_select_{refinement_count}",
        help="Choose one or more feedback items to guide the refinement",
    )

    col_btn, col_info = st.columns([2, 1])

    with col_btn:
        refine_clicked = st.button(
            "🔄 Refine Thesis",
            disabled=(not selected_feedback),
            key=f"refine_btn_{refinement_count}",
        )

    with col_info:
        st.caption(f"Refinements left: {3 - refinement_count}")

    if refine_clicked and selected_feedback:
        _run_refinement_step(selected_feedback)

    return False


def show_history(history: list):
    """Display version history in expander.

    Args:
        history: List of previous StructuredThesis objects.
    """
    with st.expander(f"📜 Previous versions ({len(history)})"):
        for i, prev_thesis in enumerate(history):
            st.caption(f"**Version {i + 1}**")
            st.write(
                f"Score: {prev_thesis.opportunity_score}/5 | "
                f"Recommendation: {prev_thesis.recommendation}"
            )


def show_related_theses(related: list) -> None:
    """Show related past theses surfaced by episodic recall (query similarity)."""
    if not related:
        return
    with st.expander(f"🧠 Related past theses ({len(related)})"):
        for r in related:
            date = (r["created_at"] or "")[:10]
            approved = " · approved" if r["approved"] else ""
            meta = " · ".join(
                p for p in (f"score {r['score']}/5", r["recommendation"], date) if p
            )
            st.markdown(f"**{r['query']}**  \n{meta}{approved} · [open](?job_id={r['job_id']})")
            st.caption(f"similarity {r['similarity']}")


def show_execution_trace(execution_log: list):
    """Display tool execution trace showing what actually executed.

    Args:
        execution_log: List of execution events from the graph.
    """
    if not execution_log:
        return

    with st.expander("📋 Execution Trace", expanded=False):
        st.caption("Tools that actually executed during refinement:")
        for i, event in enumerate(execution_log, 1):
            tool_name = event.get("tool_name", "unknown")
            status = event.get("status", "unknown")
            status_icon = "✅" if status == "executed" else "❌"
            st.write(f"{status_icon} **{i}. {tool_name}** - {status}")
            if event.get("refinement_number"):
                st.caption(f"Refinement #{event['refinement_number']}")
            if event.get("reason"):
                st.caption(f"Reason: {event['reason']}")
            for change in event.get("changes", []):
                st.caption(change)


def show_hallucination_analysis(analysis: dict):
    """Display hallucination detection results only when hallucinations found.

    Args:
        analysis: Hallucination detection analysis dictionary.
    """
    # Only show panel if hallucinations detected (invalid tools found)
    if not analysis or not analysis.get("invalid_tools"):
        return

    st.divider()
    st.warning("⚠️ **Hallucinations Detected**")
    with st.expander("🔍 Tool Call Analysis", expanded=True):
        st.write(analysis["summary"])
        st.error(f"❌ Invalid tools (do not exist): {', '.join(analysis['invalid_tools'])}")


def display_refinement_panel():
    """Display thesis refinement panel with approval, feedback options, and history.

    Orchestrates display of: approval toggle, refinement controls, history,
    and escalation messages based on thesis state from session.
    """
    ref_state = st.session_state.get("refinement_state", {})
    refinement_count = ref_state.get("refinement_count", 0)
    status = ref_state.get("status", "refining")
    thesis_count = st.session_state.get("thesis_count", 0)
    refinement_supported = ref_state.get("refinement_supported", True)

    st.divider()

    # Show approval toggle (defaults on when this run was already approved)
    is_approved = show_approval_toggle(
        thesis_count, initial=bool(st.session_state.get("approved_at"))
    )

    # If approved, persist the approval timestamp once, then show confirmation
    # and history (if supported), and exit.
    if is_approved:
        _persist_approval_once()
        show_approved_state(refinement_supported)
        return

    # If max refinements reached, show escalation message
    if status == "escalated":
        show_escalation_message()
    else:
        # Show refinement controls
        show_refinement_controls(refinement_count)

    # Show history (if available and refinement is supported)
    history = st.session_state.get("refinement_history", [])
    if history and refinement_supported:
        st.divider()
        show_history(history)


def _run_refinement_step(selected_feedback: list):
    """Execute one refinement iteration using LangGraph.

    Invokes the refinement graph with user feedback, updates session state,
    and triggers a Streamlit rerun to display results.

    Args:
        selected_feedback: List of feedback options selected by user.
    """
    ref_state = st.session_state.get("refinement_state", {})
    current_thesis = st.session_state.get("generated_thesis")
    docs = st.session_state.get("retrieved_docs", [])

    if not current_thesis or not docs:
        st.error("Thesis or documents not found in session state")
        return

    # Save current thesis to history before refining
    history = st.session_state.get("refinement_history", [])
    history.append(current_thesis)
    st.session_state["refinement_history"] = history

    # Get the LangGraph refinement graph and Langfuse handler
    try:
        graph, langfuse_handler = container.get_refinement_graph()
    except NotImplementedError as e:
        st.error(f"Refinement not supported: {e}")
        return

    # Build LangGraph state for this step
    langgraph_state = {
        "topic": ref_state.get("topic", ""),
        "documents": docs,
        "current_thesis": current_thesis,
        "feedback_history": ref_state.get("feedback_history", []) + [selected_feedback],
        "refinement_count": ref_state.get("refinement_count", 0),
        "status": "refining",
        "execution_log": st.session_state.get("execution_log", []),
        "messages": [],
    }

    with st.spinner("🔧 Refining thesis based on your feedback..."):
        try:
            # Invoke the graph — Langfuse traces the full graph when handler is set
            invoke_config = {"callbacks": [langfuse_handler]} if langfuse_handler else {}
            result_state = graph.invoke(langgraph_state, config=invoke_config)
        except NotImplementedError as e:
            st.error(f"Refinement not available: {e}. Try using Gemini instead of local mode.")
            # Mark refinement as not supported (don't show history)
            ref_state["refinement_supported"] = False
            st.session_state["refinement_state"] = ref_state
            return
        except Exception as e:
            logger.exception("Error during thesis refinement")
            st.error(f"Refinement failed: {str(e)}")
            return

    # Analyze actual tool calls from LLM messages for hallucinations
    refined_thesis = result_state["current_thesis"]
    detector = HallucinationDetector()
    hallucination_analysis = detector.analyze(result_state.get("messages", []))

    # Persist updated state to session
    st.session_state["generated_thesis"] = refined_thesis
    st.session_state["hallucination_analysis"] = hallucination_analysis
    st.session_state["execution_log"] = result_state.get("execution_log", [])
    st.session_state["refinement_state"] = {
        "topic": result_state["topic"],
        "refinement_count": result_state["refinement_count"],
        "status": result_state["status"],
        "feedback_history": result_state["feedback_history"],
    }

    # Checkpoint this round so a restart doesn't lose it
    _checkpoint(
        st.session_state.get("job_id"),
        thesis=refined_thesis,
        refinement_count=result_state["refinement_count"],
        refinement_status=result_state["status"],
        feedback_history=result_state["feedback_history"],
        execution_log=result_state.get("execution_log", []),
    )

    # Rerun to display updated thesis
    st.rerun()


def display_structured_thesis(thesis: StructuredThesis):
    """Display structured thesis in Streamlit UI.

    Args:
        thesis: StructuredThesis object to display.
    """
    # Display opportunity score and recommendation prominently
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Investment Score", f"{thesis.opportunity_score}/5")

    with col2:
        st.metric("Confidence Level", f"{int(thesis.confidence_level * 100)}%")
        if thesis.confidence_as_of:
            st.caption(f"trends as of {thesis.confidence_as_of}")

    with col3:
        # Color code the recommendation
        if thesis.recommendation == "Pursue":
            st.success(f"✅ {thesis.recommendation}")
        elif thesis.recommendation == "Investigate":
            st.info(f"🔍 {thesis.recommendation}")
        else:
            st.warning(f"⏭️ {thesis.recommendation}")

    st.divider()

    st.subheader("Key Themes")
    themes = thesis.key_themes
    st.write("\n".join(f"- {t}" for t in themes) or "No themes found.")

    st.subheader("Risks")
    risks = thesis.risks
    st.write("\n".join(f"- {r}" for r in risks) or "No risks found.")

    if thesis.key_risk_factors:
        st.caption(f"⚠️ Key Risk Factors: {', '.join(thesis.key_risk_factors)}")

    st.subheader("Investment Signals")
    signals = thesis.investment_signals
    st.write("\n".join(f"- {s}" for s in signals) or "No signals found.")


# === Main Application ===

show_resume_picker()

query = st.text_input(
    "Enter a market topic or question:",
    placeholder="e.g., Future of Digital Lending in Asia"
)

if st.button("Generate Thesis"):
    if not query or not query.strip():
        st.warning("Please enter a non-empty query.")
    else:
        try:
            # Get services from DI container
            retrieval_service = container.get_retrieval_service()
            thesis_service = container.get_thesis_service()

            # Step 1: Retrieve from the corpus
            with st.spinner("Retrieving relevant context from the corpus..."):
                docs = retrieval_service.retrieve(query, k=5)

                if not docs:
                    st.warning(
                        "No relevant documents found."
                    )
                    st.stop()

            # Step 2: Generate thesis
            with st.spinner("Generating market thesis with Gemini..."):
                thesis = thesis_service.generate_thesis(query, docs)
                st.session_state["generated_thesis"] = thesis

                # Embed the query for episodic recall: it is stored on the job 
                # and used to rank related past runs.
                query_embedding = None
                try:
                    query_embedding = (
                        container.get_embedding_model().get_embeddings().embed_query(query)
                    )
                except Exception:
                    logger.exception("Failed to embed query for recall")

                # Store retrieved documents for refinement
                st.session_state["retrieved_docs"] = docs

                # Initialize refinement state
                st.session_state["refinement_state"] = {
                    "topic": query,
                    "refinement_count": 0,
                    "status": "refining",
                    "feedback_history": [],
                    "refinement_supported": True,  # Assume supported until proven otherwise
                }
                st.session_state["refinement_history"] = []
                st.session_state["execution_log"] = []
                st.session_state["approved_at"] = None  # fresh thesis starts unapproved

                # Increment counter so the toggle gets a new unique key (resets automatically)
                st.session_state["thesis_count"] = st.session_state.get("thesis_count", 0) + 1

                # Create and checkpoint the job (no-op if Supabase isn't configured)
                job_id = None
                if job_manager:
                    try:
                        job_id = job_manager.create_job(query).id
                    except Exception:
                        logger.exception("Failed to create Supabase job")
                st.session_state["job_id"] = job_id
                if job_id:
                    st.query_params["job_id"] = job_id
                _checkpoint(
                    job_id,
                    status=JobStatus.COMPLETED,
                    thesis=thesis,
                    retrieved_docs=docs,
                    refinement_count=0,
                    refinement_status="N/A",  # not in a refinement session yet
                    feedback_history=[],
                    execution_log=[],
                    query_embedding=query_embedding,
                )

                # Episodic recall: surface past runs on similar queries.
                st.session_state["related_theses"] = _compute_related(
                    query_embedding, job_id
                )

        except Exception as exc:
            logger.exception("Error while generating thesis")
            st.error(f"An unexpected error occurred: {str(exc)}")

# Step 5: Display the articles used for context
if "retrieved_docs" in st.session_state:
    docs = st.session_state["retrieved_docs"]
    # Show each article once in relevance order.
    seen_urls = set()
    sources = []
    for doc in docs:
        url = doc.metadata.get("url", "")
        if url and url in seen_urls:
            continue
        seen_urls.add(url)
        sources.append(doc.metadata)

    # Timeframe = published-date span of the source articles
    dates = []
    for meta in sources:
        raw = meta.get("published_at")
        if raw:
            try:
                dates.append(datetime.fromisoformat(raw))
            except ValueError:
                pass
    label = "Source Articles"
    if dates:
        lo, hi = min(dates), max(dates)
        if lo.date() == hi.date():
            label += f" ({lo.strftime('%b %d, %Y')})"
        else:
            label += f" ({lo.strftime('%b %d, %Y')} - {hi.strftime('%b %d, %Y')})"

    with st.expander(label):
        for meta in sources:
            title = meta.get("title", "Untitled")
            url = meta.get("url", "")
            if url:
                st.markdown(f"• [{title}]({url})")
            else:
                st.markdown(f"• {title}")

# Step 6: Display results (outside button block so toggle doesn't disappear)
if "generated_thesis" in st.session_state:
    thesis = st.session_state["generated_thesis"]

    if thesis.raw_output:
        st.subheader("Raw Summary")
        st.code(thesis.raw_output, language="json")

    if thesis.key_themes:
        st.success("Structured thesis generated successfully")
        display_structured_thesis(thesis)

        # Related past theses (episodic recall over similar past queries)
        show_related_theses(st.session_state.get("related_theses", []))

        # Display refinement panel if thesis is available
        display_refinement_panel()

        # Display hallucination analysis if available
        if "hallucination_analysis" in st.session_state:
            st.divider()
            show_hallucination_analysis(st.session_state["hallucination_analysis"])

        # Display execution trace
        if "execution_log" in st.session_state:
            show_execution_trace(st.session_state["execution_log"])
    else:
        st.warning("Could not parse structured output. See raw output above.")
else:
    st.info("Enter a query and click 'Generate Thesis' to start.")

st.markdown("---")
