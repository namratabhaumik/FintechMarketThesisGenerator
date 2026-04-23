"""LangChain tools for thesis refinement with injected graph state."""

import json
import logging
from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from core.interfaces.thesis_structurer import IThesisStructurer
from finthesis_internal.opportunity_scoring_service import OpportunityScoringService
from core.services.thesis_generator_service import ThesisGeneratorService

logger = logging.getLogger(__name__)


def create_thesis_tools(
    thesis_service: ThesisGeneratorService,
    structuring_service: IThesisStructurer,
    scoring_service: OpportunityScoringService,
) -> list:
    """Create LangChain tools bound to services via closure.

    Tools receive state automatically via InjectedState — the LLM only
    specifies the lightweight arguments it reasons about.

    Args:
        thesis_service: For LLM-based thesis refinement.
        structuring_service: For keyword-based re-structuring.
        scoring_service: For rule-based re-scoring.

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
            "opportunity_score": refined.opportunity_score,
            "confidence_level": refined.confidence_level,
            "recommendation": refined.recommendation,
            "key_risk_factors": refined.key_risk_factors,
        })

    @tool
    def score_opportunity(
        reason: str,
        state: Annotated[dict, InjectedState],
    ) -> str:
        """Re-calculate the investment opportunity score and recommendation.

        Use ONLY when the score or recommendation needs updating without
        changing the thesis content — e.g. user says score seems too low.

        Args:
            reason: Brief explanation of why rescoring is needed.
        """
        current_thesis = state["current_thesis"]

        logger.info(f"Tool score_opportunity: reason='{reason}'")

        result = scoring_service.score_opportunity(
            key_themes=current_thesis.key_themes,
            risks=current_thesis.risks,
            investment_signals=current_thesis.investment_signals,
            sources=current_thesis.sources,
        )

        return json.dumps({
            "tool": "score_opportunity",
            "opportunity_score": result["score"],
            "confidence_level": result["confidence_level"],
            "recommendation": result["recommendation"],
            "key_risk_factors": result["key_risks"],
        })

    @tool
    def structure_thesis(
        reason: str,
        state: Annotated[dict, InjectedState],
    ) -> str:
        """Re-extract themes, risks, and investment signals from existing thesis text.

        Use ONLY when the thesis structure needs reorganising without changing
        the underlying content or score.

        Args:
            reason: Brief explanation of why re-structuring is needed.
        """
        current_thesis = state["current_thesis"]
        raw_text = current_thesis.raw_output or ""

        logger.info(f"Tool structure_thesis: reason='{reason}'")

        result = structuring_service.structure_thesis(raw_text)

        return json.dumps({
            "tool": "structure_thesis",
            "key_themes": result.get("key_themes", []),
            "risks": result.get("risks", []),
            "investment_signals": result.get("investment_signals", []),
        })

    return [refine_thesis, score_opportunity, structure_thesis]
