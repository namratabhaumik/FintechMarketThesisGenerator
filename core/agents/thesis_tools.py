"""LangChain tools for thesis refinement with injected graph state."""

import json
import logging
from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from core.services.thesis_generator_service import ThesisGeneratorService

logger = logging.getLogger(__name__)


def create_thesis_tools(
    thesis_service: ThesisGeneratorService,
) -> list:
    """Create LangChain tools bound to services via closure.

    Tools receive state automatically via InjectedState — the LLM only
    specifies the lightweight arguments it reasons about.

    Args:
        thesis_service: For LLM-based thesis refinement.

    Returns:
        List of LangChain tool functions ready for binding to an LLM.
    """

    @tool
    def refine_thesis(
        feedback_focus: str,
        state: Annotated[dict, InjectedState],
    ) -> str:
        """Rewrite the investment thesis content based on user feedback.

        Use this when content needs changing: missing market trends, vague
        investment signals, too many risks listed, analysis too broad, or
        needs stronger evidence for key themes.

        Args:
            feedback_focus: Brief description of what aspect to focus on.
        """
        documents = state["documents"]
        current_thesis = state["current_thesis"]
        topic = state["topic"]
        feedback_items = (
            state["feedback_history"][-1]
            if state["feedback_history"]
            else [feedback_focus]
        )

        logger.info(f"Tool refine_thesis: focus='{feedback_focus}'")

        refined = thesis_service.refine_thesis(
            topic=topic,
            documents=documents,
            current_thesis=current_thesis,
            feedback_items=feedback_items,
        )

        return json.dumps({
            "tool": "refine_thesis",
            "key_themes": refined.key_themes,
            "risks": refined.risks,
            "investment_signals": refined.investment_signals,
            "sources": refined.sources,
            "raw_output": refined.raw_output,
        })

    return [refine_thesis]
