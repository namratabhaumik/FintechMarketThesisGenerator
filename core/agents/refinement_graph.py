"""LangGraph-based iterative thesis refinement agent."""

import logging
from functools import partial
from typing import List, Optional, TypedDict

from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END

from core.models.thesis import StructuredThesis
from core.services.thesis_generator_service import ThesisGeneratorService

logger = logging.getLogger(__name__)

# Maximum number of refinements allowed (refinement_count ∈ [0, 3))
MAX_REFINEMENTS = 3


class ThesisRefinementState(TypedDict):
    """State for thesis refinement graph.

    Attributes:
        topic: Original market topic.
        documents: List of source documents for context.
        current_thesis: Current StructuredThesis object.
        feedback_history: List of feedback rounds, each containing feedback items selected.
        refinement_count: Number of refinements completed so far [0, 1, 2].
        status: Current status ("refining", "escalated", "done").
    """

    topic: str
    documents: List[Document]
    current_thesis: StructuredThesis
    feedback_history: List[List[str]]  # Each entry is one round's selected feedback items
    refinement_count: int  # Range: [0, 3)
    status: str  # "refining" | "escalated" | "done"


def refine_node(
    state: ThesisRefinementState,
    thesis_service: ThesisGeneratorService,
) -> ThesisRefinementState:
    """Execute one refinement iteration.

    Calls thesis_service.refine_thesis() with the latest feedback,
    updates the thesis, and increments refinement_count.

    Args:
        state: Current refinement state.
        thesis_service: Service for thesis refinement.

    Returns:
        Updated state with refined thesis and incremented count.
    """
    current_feedback = state["feedback_history"][-1]  # Latest feedback items

    logger.info(
        f"Refining thesis (refinement {state['refinement_count'] + 1}/{MAX_REFINEMENTS})"
    )

    refined_thesis = thesis_service.refine_thesis(
        topic=state["topic"],
        documents=state["documents"],
        current_thesis=state["current_thesis"],
        feedback_items=current_feedback,
    )

    new_refinement_count = state["refinement_count"] + 1

    logger.info(f"Refinement complete. New count: {new_refinement_count}")

    return {
        **state,
        "current_thesis": refined_thesis,
        "refinement_count": new_refinement_count,
        "status": "refining",  # Keep as refining, let router decide escalation
    }


def escalate_node(state: ThesisRefinementState) -> ThesisRefinementState:
    """Terminal node when max refinements reached.

    Sets status to "escalated" to signal the UI that further refinements
    are not allowed.

    Args:
        state: Current refinement state.

    Returns:
        State with status set to "escalated".
    """
    logger.info("Max refinements reached. Escalating.")

    return {
        **state,
        "status": "escalated",
    }


def route_entry(state: ThesisRefinementState) -> str:
    """Conditional router at graph entry.

    Checks if refinement_count has already reached MAX_REFINEMENTS.
    If so, go to escalate_node. Otherwise, go to refine_node.

    Args:
        state: Current refinement state.

    Returns:
        Route name: "refine" or "escalate".
    """
    if state["refinement_count"] >= MAX_REFINEMENTS:
        logger.info(
            f"Refinement count {state['refinement_count']} >= MAX {MAX_REFINEMENTS}, escalating"
        )
        return "escalate"
    return "refine"


def build_refinement_graph(
    thesis_service: ThesisGeneratorService,
) -> object:
    """Build and compile the LangGraph refinement state machine.

    Defines nodes and edges:
    - Entry: route_entry conditional (checks refinement_count)
    - refine_node: Executes one refinement iteration
    - escalate_node: Terminal node when max reached
    - Edges: START → route_entry → {refine | escalate} → END

    Args:
        thesis_service: Service for thesis refinement operations.

    Returns:
        Compiled LangGraph graph ready for invocation.
    """
    graph = StateGraph(ThesisRefinementState)

    # Add nodes
    graph.add_node("refine", partial(refine_node, thesis_service=thesis_service))
    graph.add_node("escalate", escalate_node)

    # Conditional routing from entry point
    graph.set_conditional_entry_point(
        route_entry,
        {
            "refine": "refine",
            "escalate": "escalate",
        },
    )

    # Both paths lead to END
    graph.add_edge("refine", END)
    graph.add_edge("escalate", END)

    # Compile and return
    compiled_graph = graph.compile()
    logger.info(
        f"Refinement graph built with MAX_REFINEMENTS={MAX_REFINEMENTS}, compiled and ready"
    )
    return compiled_graph
