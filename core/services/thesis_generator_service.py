"""Thesis generator service."""

import logging
from typing import List

from langchain.docstore.document import Document

from core.interfaces.llm import ILanguageModel
from core.models.thesis import StructuredThesis
from core.services.thesis_structuring_service import ThesisStructuringService

logger = logging.getLogger(__name__)


class ThesisGeneratorService:
    """Service for generating market theses.

    Single Responsibility: Thesis generation orchestration.
    Implements Dependency Inversion: Depends on ILanguageModel abstraction.
    """

    def __init__(self, llm: ILanguageModel):
        """Initialize with language model.

        Args:
            llm: Injected LLM implementation (only for summarization).
        """
        self._llm = llm
        self._structuring_service = ThesisStructuringService()

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
        sources = [
            doc.metadata["url"]
            for doc in documents
            if doc.metadata.get("url")
        ]

        logger.info("Successfully generated structured thesis")
        return StructuredThesis(
            key_themes=result.get("key_themes", []),
            risks=result.get("risks", []),
            investment_signals=result.get("investment_signals", []),
            sources=sources,
            raw_output=summary
        )
