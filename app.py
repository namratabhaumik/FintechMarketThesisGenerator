# app.py
"""
Streamlit frontend for FinThesis (Refactored with SOLID principles)
Uses dependency injection to manage all dependencies.
"""

import logging
import os

import streamlit as st
from dotenv import load_dotenv

from config.settings import AppConfig
from core.agents.hallucination_detector import HallucinationDetector
from core.models.thesis import StructuredThesis
from core.utils.logging import setup_logging
from dependency_injection.container import ServiceContainer

# Load environment variables from .env
load_dotenv()

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
    if "vectorstore_built" in st.session_state:
        del st.session_state["vectorstore_built"]
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


# === UI Helper Functions ===


def show_approval_toggle(thesis_count: int) -> bool:
    """Display approval toggle and return its state.

    Args:
        thesis_count: Unique counter for widget key.

    Returns:
        Boolean indicating if thesis is approved.
    """
    col_toggle, col_spacer = st.columns([1, 2])
    with col_toggle:
        return st.toggle(
            "✅ Approved",
            value=False,
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

    # Show approval toggle
    is_approved = show_approval_toggle(thesis_count)

    # If approved, show confirmation and history (if supported), then exit
    if is_approved:
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

    # Get the LangGraph refinement graph
    try:
        graph = container.get_refinement_graph()
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
            # Invoke the graph for one step
            result_state = graph.invoke(langgraph_state)
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
            ingestion_service = container.get_ingestion_service()
            retrieval_service = container.get_retrieval_service()
            thesis_service = container.get_thesis_service()

            # Step 1: Fetch articles
            with st.spinner("Fetching latest fintech news from RSS feeds..."):
                articles = ingestion_service.fetch_articles(query="fintech", limit=5)

                if not articles:
                    st.warning("No articles found. Please try again later.")
                    st.stop()

                # Store articles in session state for display outside button block
                st.session_state["articles"] = articles

            # Step 2: Build vectorstore (cache in session)
            if "vectorstore_built" not in st.session_state:
                with st.spinner("Building FAISS vectorstore (one-time)..."):
                    documents = ingestion_service.convert_to_documents(articles)
                    retrieval_service.build_vectorstore(documents)
                    st.session_state["vectorstore_built"] = True
                    logger.info("Vectorstore built and cached in session")

            # Step 3: Retrieve relevant documents
            with st.spinner("Retrieving relevant context from vectorstore..."):
                docs = retrieval_service.retrieve(query, k=5)

                if not docs:
                    st.warning("No relevant documents found for this query.")
                    st.stop()

            # Step 4: Generate thesis
            with st.spinner("Generating market thesis with Gemini..."):
                thesis = thesis_service.generate_thesis(query, docs)
                st.session_state["generated_thesis"] = thesis

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

                # Increment counter so the toggle gets a new unique key (resets automatically)
                st.session_state["thesis_count"] = st.session_state.get("thesis_count", 0) + 1

        except Exception as exc:
            logger.exception("Error while generating thesis")
            st.error(f"An unexpected error occurred: {str(exc)}")

# Step 5: Display articles (outside button block so they persist across reruns)
if "articles" in st.session_state:
    st.subheader("Latest Fintech Articles")
    articles = st.session_state["articles"]
    for article in articles:
        if article.url:
            st.markdown(f"• [{article.title}]({article.url})")
        else:
            st.markdown(f"• {article.title}")

# Step 6: Display results (outside button block so toggle doesn't disappear)
if "generated_thesis" in st.session_state:
    thesis = st.session_state["generated_thesis"]

    if thesis.raw_output:
        st.subheader("Raw Summary")
        st.code(thesis.raw_output, language="json")

    if thesis.key_themes:
        st.success("Structured thesis generated successfully")
        display_structured_thesis(thesis)

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
