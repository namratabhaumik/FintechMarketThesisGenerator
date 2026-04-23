"""Thesis generator service."""

import logging
from typing import List

from langchain_core.documents import Document

from core.interfaces.llm import ILanguageModel
from core.interfaces.thesis_structurer import IThesisStructurer
from core.models.thesis import StructuredThesis
from finthesis_internal.opportunity_scoring_service import OpportunityScoringService

logger = logging.getLogger(__name__)


class ThesisGeneratorService:
    """Service for generating market theses.

    Single Responsibility: Thesis generation orchestration.
    Implements Dependency Inversion: Depends on abstractions (ILanguageModel,
    IThesisStructurer) rather than concrete implementations.
    """

    def __init__(
        self,
        llm: ILanguageModel,
        structuring_service: IThesisStructurer,
        scoring_service: OpportunityScoringService,
    ):
        """Initialize with dependencies.

        Args:
            llm: Injected LLM implementation (for summarization).
            structuring_service: Injected thesis structurer implementation.
            scoring_service: Injected opportunity scoring service.
        """
        self._llm = llm
        self._structuring_service = structuring_service
        self._scoring_service = scoring_service

    def generate_thesis(
        self,
        topic: str,
        documents: List[Document]
    ) -> StructuredThesis:
        """Generate structured thesis from documents.

        Args:
            topic: Market topic for analysis.
            documents: Retrieved context documents.

        Returns:
            StructuredThesis object with analysis results.
        """
        logger.info(f"Generating thesis for topic: {topic}")

        # Step 1: Summarize documents (LLM only)
        logger.info("Step 1: Summarizing retrieved documents...")
        summary = self._llm.summarize(documents)

        if not summary:
            logger.error("Empty summary returned by LLM")
            raise RuntimeError("Failed to generate summary")

        # Step 2: Structure summary into thesis (Python only - no LLM)
        logger.info("Step 2: Structuring thesis from summary...")
        result = self._structuring_service.structure_thesis(summary)

        # Step 3: Extract sources from document metadata
        logger.info("Step 3: Extracting sources from documents...")
        sources = [doc.metadata["url"] for doc in documents if doc.metadata.get("url")]

        # Step 4: Score opportunity (rule-based, no LLM)
        logger.info("Step 4: Scoring opportunity...")
        key_themes = result.get("key_themes", [])
        risks = result.get("risks", [])
        investment_signals = result.get("investment_signals", [])

        score_result = self._scoring_service.score_opportunity(
            key_themes=key_themes,
            risks=risks,
            investment_signals=investment_signals,
            sources=sources
        )

        logger.info("Successfully generated structured thesis with scoring")
        return StructuredThesis(
            key_themes=key_themes,
            risks=risks,
            investment_signals=investment_signals,
            sources=sources,
            raw_output=summary,
            opportunity_score=score_result["score"],
            confidence_level=score_result["confidence_level"],
            recommendation=score_result["recommendation"],
            key_risk_factors=score_result["key_risks"]
        )

    def refine_thesis(
        self,
        topic: str,
        documents: List[Document],
        current_thesis: StructuredThesis,
        feedback_items: List[str],
    ) -> StructuredThesis:
        """Refine an existing thesis based on user feedback.

        Calls LLM with feedback-aware prompt, then runs same structuring
        and scoring pipeline as generate_thesis.

        Args:
            topic: Original market topic for analysis.
            documents: Retrieved context documents.
            current_thesis: The thesis to refine.
            feedback_items: Predefined feedback strings selected by user.

        Returns:
            New StructuredThesis with refinements applied.
        """
        logger.info(f"Refining thesis for topic: {topic} based on {len(feedback_items)} feedback items")

        # Step 1: Get refined summary from LLM using feedback
        current_thesis_text = current_thesis.raw_output or ""
        logger.info("Step 1: Refining thesis with LLM feedback...")
        refined_summary = self._llm.refine(documents, current_thesis_text, feedback_items)

        if not refined_summary:
            logger.error("Empty refined summary returned by LLM")
            raise RuntimeError("Failed to refine thesis")

        # Step 2: Structure refined summary (same as generate_thesis)
        logger.info("Step 2: Structuring refined thesis...")
        result = self._structuring_service.structure_thesis(refined_summary)

        # Step 3: Extract sources from document metadata (same as generate_thesis)
        logger.info("Step 3: Extracting sources from documents...")
        sources = [doc.metadata["url"] for doc in documents if doc.metadata.get("url")]

        # Step 4: Score opportunity (same as generate_thesis)
        logger.info("Step 4: Scoring refined opportunity...")
        key_themes = result.get("key_themes", [])
        risks = result.get("risks", [])
        investment_signals = result.get("investment_signals", [])

        score_result = self._scoring_service.score_opportunity(
            key_themes=key_themes,
            risks=risks,
            investment_signals=investment_signals,
            sources=sources
        )

        logger.info("Successfully refined thesis with updated scoring")
        return StructuredThesis(
            key_themes=key_themes,
            risks=risks,
            investment_signals=investment_signals,
            sources=sources,
            raw_output=refined_summary,
            opportunity_score=score_result["score"],
            confidence_level=score_result["confidence_level"],
            recommendation=score_result["recommendation"],
            key_risk_factors=score_result["key_risks"]
        )
