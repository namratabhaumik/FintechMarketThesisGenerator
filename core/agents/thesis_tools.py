"""LangChain tools for thesis refinement with injected graph state."""

import json
import logging
from typing import Annotated

from langchain_core.tools import tool
from pydantic import Field
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
    async def refine_thesis(
        feedback_focus: str,
        theme_delta: Annotated[int, Field(ge=-1, le=1)],
        risk_delta: Annotated[int, Field(ge=-1, le=1)],
        signal_delta: Annotated[int, Field(ge=-1, le=1)],
        state: Annotated[dict, InjectedState],
    ) -> str:
        """Rewrite the investment thesis content based on user feedback.

        Use this when content needs changing: missing market trends, vague
        investment signals, too many risks listed, analysis too broad, or
        needs stronger evidence for key themes.

        Args:
            feedback_focus: Brief description of what aspect to focus on.
            theme_delta: How many more/fewer themes to surface, from this
                feedback (eg: +1 if thesis is too broad/generic or is 
                missing recent trends; -1 if themes are repetitive; 0 otherwise)
            risk_delta: Same idea for risks (eg: +1 if feedback wants risks
                covered more thoroughly; -1 if it says there are too many
                risks listed; 0 otherwise)
            signal_delta: Same idea for investment signals (eg: +1 if signals 
            are vague or the opportunity score seems too low; -1 if too many 
            signals; 0 otherwise)
        """
        documents = state["documents"]
        # The diverse subset the LLM rewrites from; older jobs (pre-split) have
        # no summary_documents, so fall back to the full retrieved set.
        summary_documents = state.get("summary_documents") or documents
        current_thesis = state["current_thesis"]
        topic = state["topic"]
        # Split the feedback trajectory: this round's feedback drives the evidence
        # lens and the rewrite; earlier rounds are passed as constraints to
        # preserve (prior_feedback), so a new round does not regress them.
        history = state["feedback_history"]
        feedback_items = history[-1] if history else [feedback_focus]
        prior_feedback = history[:-1] if history else []

        logger.info(
            f"Tool refine_thesis: focus='{feedback_focus}' "
            f"deltas=(theme={theme_delta}, risk={risk_delta}, signal={signal_delta})"
        )

        refined = await thesis_service.refine_thesis(
            topic=topic,
            documents=documents,
            current_thesis=current_thesis,
            feedback_items=feedback_items,
            summary_documents=summary_documents,
            prior_feedback=prior_feedback,
            theme_delta=theme_delta,
            risk_delta=risk_delta,
            signal_delta=signal_delta,
        )

        return json.dumps({
            "tool": "refine_thesis",
            "key_themes": refined.key_themes,
            "risks": refined.risks,
            "investment_signals": refined.investment_signals,
            "sources": refined.sources,
            "raw_output": refined.raw_output,
            "summary_status": refined.summary_status,
            "refusal_reason": refined.refusal_reason,
        })

    return [refine_thesis]
