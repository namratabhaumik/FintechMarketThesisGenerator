"""Thesis generator service."""

import logging
from typing import List

from langchain.docstore.document import Document

from core.interfaces.llm import ILanguageModel
from core.models.thesis import StructuredThesis

logger = logging.getLogger(__name__)


class ThesisGeneratorService:
    """Service for generating market theses.

    Single Responsibility: Thesis generation orchestration.
    Implements Dependency Inversion: Depends on ILanguageModel abstraction.
    """

    def __init__(self, llm: ILanguageModel):
        """Initialize with language model.

        Args:
            llm: Injected LLM implementation.
        """
        self._llm = llm

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

        # Step 1: Summarize documents
        logger.info("Step 1: Summarizing retrieved documents...")
        summary = self._llm.summarize(documents)

        if not summary:
            logger.error("Empty summary returned by LLM")
            raise RuntimeError("Failed to generate summary")

        # Step 2: Generate structured output
        logger.info("Step 2: Generating structured thesis...")
        prompt = self._build_thesis_prompt(topic, summary)
        result = self._llm.generate_structured_output(prompt)

        # Step 3: Parse into structured thesis
        if result.get("json"):
            logger.info("Successfully generated structured thesis")
            return StructuredThesis(
                key_themes=result["json"].get("key_themes", []),
                risks=result["json"].get("risks", []),
                investment_signals=result["json"].get("investment_signals", []),
                sources=result["json"].get("sources", []),
                raw_output=result["raw"]
            )
        else:
            logger.warning("Failed to parse JSON output, returning partial thesis")
            return StructuredThesis(
                key_themes=[],
                risks=[],
                investment_signals=[],
                sources=[],
                raw_output=result.get("raw")
            )

    def _build_thesis_prompt(self, topic: str, summary: str) -> str:
        """Build prompt for structured thesis generation.

        Args:
            topic: Market topic.
            summary: Summarized context.

        Returns:
            Prompt string for the LLM.
        """
        return f"""
You are an expert VC analyst. Based on this summarized evidence about "{topic}":

{summary}

Return a JSON object with keys:
- key_themes: list of 3 concise themes
- risks: list of 3 concise risks
- investment_signals: list of 3 startup focus areas
- sources: list of source titles or URLs
Only output valid JSON.
"""
