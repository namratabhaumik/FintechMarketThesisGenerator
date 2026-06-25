"""LangGraph-based iterative thesis refinement with real tool calling."""

import json
import logging
import os
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from core.agents.execution_tracker import ExecutionTracker  # noqa: F401 (kept for compatibility)
from core.agents.thesis_tools import create_thesis_tools
from core.models.thesis import StructuredThesis
from core.services.thesis_generator_service import ThesisGeneratorService

logger = logging.getLogger(__name__)

MAX_REFINEMENTS = 3


class ThesisRefinementState(TypedDict):
    """State for the thesis refinement graph.

    Attributes:
        topic: Original market topic.
        documents: Source documents for context.
        current_thesis: Current StructuredThesis object.
        feedback_history: List of feedback rounds.
        refinement_count: Number of refinements completed [0, MAX_REFINEMENTS).
        status: Current status ("refining" | "escalated").
        execution_log: Tool execution events for the UI trace.
        messages: LLM conversation messages for tool calling.
    """

    topic: str
    documents: List[Document]
    current_thesis: StructuredThesis
    feedback_history: List[List[str]]
    refinement_count: int
    status: str
    execution_log: List[Dict[str, Any]]
    messages: Annotated[List[BaseMessage], add_messages]


def _create_langfuse_handler() -> Optional[object]:
    """Create a Langfuse callback handler if credentials are configured."""
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        logger.info("Langfuse credentials not set - tracing disabled")
        return None

    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler

    # v4: the CallbackHandler only binds to a global Langfuse client singleton, so it
    # must be initialized first. Langfuse() reads keys + LANGFUSE_BASE_URL from env.
    Langfuse()
    logger.info("Langfuse tracing enabled")
    return CallbackHandler()


def _make_planner_node(llm_with_tools):
    """Return a planner node that asks the LLM which tool to invoke."""

    def planner_node(state: ThesisRefinementState) -> dict:
        latest_feedback = (
            state["feedback_history"][-1] if state["feedback_history"] else []
        )
        feedback_str = "\n".join(f"- {f}" for f in latest_feedback)

        prompt = (
            f"You are a fintech market analyst assistant. A user reviewed an investment "
            f"thesis and provided feedback. Choose the right tool to address it.\n\n"
            f"CURRENT THESIS SCORE: {state['current_thesis'].opportunity_score}/5\n"
            f"RECOMMENDATION: {state['current_thesis'].recommendation}\n\n"
            f"USER FEEDBACK:\n{feedback_str}\n\n"
            f"Tool selection guide:\n"
            f"- refine_thesis: rewrite the thesis to address the feedback - missing "
            f"trends, vague signals, too many risks, too broad, weak evidence, or a "
            f"score the user disputes (strengthen signals / trim weak risks). The "
            f"opportunity score is recomputed automatically from the revised content."
        )

        messages = state.get("messages", []) + [HumanMessage(content=prompt)]
        try:
            response = llm_with_tools.invoke(messages)
        except Exception as e:
            logger.error(f"Planner LLM call failed: {e}")
            raise

        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_names = [tc["name"] for tc in response.tool_calls]
            logger.info(f"Planner chose tools: {tool_names}")
        else:
            logger.info("Planner made no tool call — will pass through to assemble")

        return {"messages": messages + [response]}

    return planner_node


def _resolve_components(tool_name: str, result: dict, current: StructuredThesis):
    """Resolve thesis components for a recognized tool, or None if unknown.

    refine_thesis takes its new content from the tool result; scoring itself
    is handled downstream by _score_and_build.

    Returns:
        Tuple (key_themes, risks, investment_signals, sources, raw_output),
        or None when the tool name is not recognized.
    """
    if tool_name == "refine_thesis":
        return (
            result.get("key_themes", current.key_themes),
            result.get("risks", current.risks),
            result.get("investment_signals", current.investment_signals),
            result.get("sources", current.sources),
            result.get("raw_output", current.raw_output),
        )
    return None


def _score_and_build(components, current: StructuredThesis) -> StructuredThesis:
    """Re-assemble a StructuredThesis after refinement.

    Every number is FROZEN - score, recommendation, confidence and the
    confidence as-of date all reflect the retrieved evidence and the Gold
    snapshot, neither of which a refinement changes, so they are carried forward
    from the current thesis. Only the refined content (narrative + displayed
    tags) is swapped in; key risks follow the refined risk list.

    components is (key_themes, risks, investment_signals, sources, raw_output).
    """
    key_themes, risks, investment_signals, sources, raw_output = components
    return StructuredThesis(
        key_themes=key_themes,
        risks=risks,
        investment_signals=investment_signals,
        sources=sources,
        raw_output=raw_output,
        opportunity_score=current.opportunity_score,
        confidence_level=current.confidence_level,
        confidence_as_of=current.confidence_as_of,
        recommendation=current.recommendation,
        key_risk_factors=risks[:min(3, len(risks))],
    )


def _diff_thesis(current: StructuredThesis, new: StructuredThesis) -> List[str]:
    """Short, plain lines describing what a refinement changed.

    The numbers are frozen by design, so this surfaces the narrative rewrite and
    the per-dimension added/removed tags, then states plainly that the numbers
    held - so a reader who asked to "raise the score" can see why it did not move.
    """
    changes: List[str] = []
    if (new.raw_output or "") != (current.raw_output or ""):
        changes.append("Narrative rewritten")
    for label, before, after in (
        ("Themes", current.key_themes, new.key_themes),
        ("Risks", current.risks, new.risks),
        ("Signals", current.investment_signals, new.investment_signals),
    ):
        added = [t for t in after if t not in before]
        removed = [t for t in before if t not in after]
        parts = []
        if added:
            parts.append("+" + ", ".join(added))
        if removed:
            parts.append("-" + ", ".join(removed))
        if parts:
            changes.append(f"{label}: " + "  ".join(parts))
    changes.append("Score, confidence, recommendation unchanged")
    return changes


def _make_assemble_node():
    """Return an assemble node that rebuilds StructuredThesis from tool output."""

    def assemble_node(state: ThesisRefinementState) -> dict:
        current = state["current_thesis"]
        execution_log = list(state.get("execution_log", []))
        new_refinement_count = state["refinement_count"] + 1

        def skip(tool_name: str, status: str) -> dict:
            """Log a non-update and return state with the thesis unchanged."""
            execution_log.append({
                "tool_name": tool_name,
                "status": status,
                "refinement_number": new_refinement_count,
            })
            return {
                "refinement_count": new_refinement_count,
                "status": "refining",
                "execution_log": execution_log,
            }

        tool_messages = [m for m in state["messages"] if isinstance(m, ToolMessage)]
        if not tool_messages:
            logger.warning("assemble_node: no ToolMessage found, keeping current thesis")
            return skip("no_tool", "skipped")

        try:
            result = json.loads(tool_messages[-1].content)
        except (json.JSONDecodeError, TypeError):
            logger.error(
                f"assemble_node: could not parse tool result as JSON: "
                f"{tool_messages[-1].content[:200]}"
            )
            return skip("unknown", "parse_error")

        tool_name = result.get("tool", "unknown")
        components = _resolve_components(tool_name, result, current)
        if components is None:
            logger.warning(f"assemble_node: unknown tool '{tool_name}', keeping current thesis")
            return skip(tool_name, "skipped")

        # Re-assemble the refined thesis. All numbers (score, recommendation,
        # confidence, as-of) are frozen - carried forward from `current`, since
        # they reflect the unchanged retrieved evidence and Gold snapshot. Only
        # the narrative and displayed tags move.
        new_thesis = _score_and_build(components, current)

        execution_log.append({
            "tool_name": tool_name,
            "status": "executed",
            "refinement_number": new_refinement_count,
            "changes": _diff_thesis(current, new_thesis),
        })

        logger.info(
            f"assemble_node: thesis updated via {tool_name} "
            f"(refinement {new_refinement_count}/{MAX_REFINEMENTS})"
        )

        return {
            "current_thesis": new_thesis,
            "refinement_count": new_refinement_count,
            "status": "refining",
            "execution_log": execution_log,
        }

    return assemble_node


def _escalate_node(state: ThesisRefinementState) -> dict:
    """Terminal node when max refinements reached."""
    logger.info("Max refinements reached. Escalating.")
    execution_log = list(state.get("execution_log", []))
    execution_log.append({
        "tool_name": "escalate",
        "status": "executed",
        "reason": "max_refinements_reached",
    })
    return {"status": "escalated", "execution_log": execution_log}


def _route_entry(state: ThesisRefinementState) -> str:
    if state["refinement_count"] >= MAX_REFINEMENTS:
        logger.info(f"refinement_count={state['refinement_count']} >= MAX, escalating")
        return "escalate"
    return "planner"


def _route_after_planner(state: ThesisRefinementState) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "assemble"


def build_refinement_graph(
    thesis_service: ThesisGeneratorService,
    gemini_api_key: str,
    model_name: str = "gemini-2.5-flash",
) -> object:
    """Build and compile the LangGraph refinement graph with real tool calling.

    Graph flow:
        START → route_entry → planner (LLM picks tool) → ToolNode (executes)
                                                        → assemble (rebuilds thesis) → END
                            → escalate → END

    Args:
        thesis_service: For LLM-driven thesis rewriting.
        gemini_api_key: API key for the planner LLM.
        model_name: Gemini model to use for tool-call decisions.

    Returns:
        Tuple of (compiled graph, langfuse callback handler or None).
    """
    tools = create_thesis_tools(thesis_service)

    planner_llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0,
        google_api_key=gemini_api_key,
    ).bind_tools(tools)

    graph = StateGraph(ThesisRefinementState)

    graph.add_node("planner", _make_planner_node(planner_llm))
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("assemble", _make_assemble_node())
    graph.add_node("escalate", _escalate_node)

    graph.set_conditional_entry_point(
        _route_entry,
        {"planner": "planner", "escalate": "escalate"},
    )

    graph.add_conditional_edges(
        "planner",
        _route_after_planner,
        {"tools": "tools", "assemble": "assemble"},
    )

    graph.add_edge("tools", "assemble")
    graph.add_edge("assemble", END)
    graph.add_edge("escalate", END)

    compiled = graph.compile()
    langfuse_handler = _create_langfuse_handler()
    logger.info(
        f"Refinement graph compiled with real tool calling, "
        f"MAX_REFINEMENTS={MAX_REFINEMENTS}, model={model_name}"
    )
    return compiled, langfuse_handler
