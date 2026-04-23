"""LangGraph-based iterative thesis refinement with real tool calling."""

import json
import logging
import os
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from core.agents.execution_tracker import ExecutionTracker  # noqa: F401 (kept for compatibility)
from core.agents.thesis_tools import create_thesis_tools
from core.interfaces.thesis_structurer import IThesisStructurer
from core.models.thesis import StructuredThesis
from finthesis_internal.opportunity_scoring_service import OpportunityScoringService
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
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if not secret_key or not public_key:
        logger.info("Langfuse credentials not set — tracing disabled")
        return None

    from langfuse.callback import CallbackHandler

    handler = CallbackHandler(
        secret_key=secret_key,
        public_key=public_key,
        host=host,
    )
    logger.info("Langfuse tracing enabled")
    return handler


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
            f"- refine_thesis: content needs changing (missing trends, vague signals, "
            f"too many risks, too broad, weak evidence)\n"
            f"- score_opportunity: ONLY if user explicitly says score is wrong\n"
            f"- structure_thesis: ONLY if themes/risks/signals need reorganising"
        )

        messages = state.get("messages", []) + [HumanMessage(content=prompt)]
        response = llm_with_tools.invoke(messages)

        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_names = [tc["name"] for tc in response.tool_calls]
            logger.info(f"Planner chose tools: {tool_names}")
        else:
            logger.info("Planner made no tool call — will pass through to assemble")

        return {"messages": messages + [response]}

    return planner_node


def _make_assemble_node(scoring_service: OpportunityScoringService):
    """Return an assemble node that rebuilds StructuredThesis from tool output."""

    def assemble_node(state: ThesisRefinementState) -> dict:
        # Find the last ToolMessage
        tool_messages = [m for m in state["messages"] if isinstance(m, ToolMessage)]

        current = state["current_thesis"]
        execution_log = list(state.get("execution_log", []))
        new_refinement_count = state["refinement_count"] + 1

        if not tool_messages:
            logger.warning("assemble_node: no ToolMessage found, keeping current thesis")
            execution_log.append({
                "tool_name": "no_tool",
                "status": "skipped",
                "refinement_number": new_refinement_count,
            })
            return {
                "refinement_count": new_refinement_count,
                "status": "refining",
                "execution_log": execution_log,
            }

        last_tool_msg = tool_messages[-1]
        try:
            result = json.loads(last_tool_msg.content)
        except (json.JSONDecodeError, TypeError):
            logger.error(f"assemble_node: could not parse tool result as JSON: {last_tool_msg.content[:200]}")
            execution_log.append({
                "tool_name": "unknown",
                "status": "parse_error",
                "refinement_number": new_refinement_count,
            })
            return {
                "refinement_count": new_refinement_count,
                "status": "refining",
                "execution_log": execution_log,
            }

        tool_name = result.get("tool", "unknown")

        if tool_name == "refine_thesis":
            new_thesis = StructuredThesis(
                key_themes=result.get("key_themes", current.key_themes),
                risks=result.get("risks", current.risks),
                investment_signals=result.get("investment_signals", current.investment_signals),
                sources=result.get("sources", current.sources),
                raw_output=result.get("raw_output", current.raw_output),
                opportunity_score=result.get("opportunity_score", current.opportunity_score),
                confidence_level=result.get("confidence_level", current.confidence_level),
                recommendation=result.get("recommendation", current.recommendation),
                key_risk_factors=result.get("key_risk_factors", current.key_risk_factors),
            )

        elif tool_name == "score_opportunity":
            # Keep content, update score fields only
            new_thesis = StructuredThesis(
                key_themes=current.key_themes,
                risks=current.risks,
                investment_signals=current.investment_signals,
                sources=current.sources,
                raw_output=current.raw_output,
                opportunity_score=result.get("opportunity_score", current.opportunity_score),
                confidence_level=result.get("confidence_level", current.confidence_level),
                recommendation=result.get("recommendation", current.recommendation),
                key_risk_factors=result.get("key_risk_factors", current.key_risk_factors),
            )

        elif tool_name == "structure_thesis":
            # Update structure, re-score with new components
            key_themes = result.get("key_themes", current.key_themes)
            risks = result.get("risks", current.risks)
            investment_signals = result.get("investment_signals", current.investment_signals)
            score_result = scoring_service.score_opportunity(
                key_themes=key_themes,
                risks=risks,
                investment_signals=investment_signals,
                sources=current.sources,
            )
            new_thesis = StructuredThesis(
                key_themes=key_themes,
                risks=risks,
                investment_signals=investment_signals,
                sources=current.sources,
                raw_output=current.raw_output,
                opportunity_score=score_result["score"],
                confidence_level=score_result["confidence_level"],
                recommendation=score_result["recommendation"],
                key_risk_factors=score_result["key_risks"],
            )

        else:
            logger.warning(f"assemble_node: unknown tool '{tool_name}', keeping current thesis")
            new_thesis = current

        execution_log.append({
            "tool_name": tool_name,
            "status": "executed",
            "refinement_number": new_refinement_count,
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
    structuring_service: IThesisStructurer,
    scoring_service: OpportunityScoringService,
    gemini_api_key: str,
    model_name: str = "gemini-2.0-flash",
) -> object:
    """Build and compile the LangGraph refinement graph with real tool calling.

    Graph flow:
        START → route_entry → planner (LLM picks tool) → ToolNode (executes)
                                                        → assemble (rebuilds thesis) → END
                            → escalate → END

    Args:
        thesis_service: For LLM-driven thesis rewriting.
        structuring_service: For keyword-based re-structuring.
        scoring_service: For rule-based re-scoring.
        gemini_api_key: API key for the planner LLM.
        model_name: Gemini model to use for tool-call decisions.

    Returns:
        Tuple of (compiled graph, langfuse callback handler or None).
    """
    tools = create_thesis_tools(thesis_service, structuring_service, scoring_service)

    planner_llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0,
        google_api_key=gemini_api_key,
    ).bind_tools(tools)

    graph = StateGraph(ThesisRefinementState)

    graph.add_node("planner", _make_planner_node(planner_llm))
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("assemble", _make_assemble_node(scoring_service))
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
