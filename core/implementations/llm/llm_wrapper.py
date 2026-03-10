"""LLM wrapper with retry logic and fallback."""

import logging
import time
from typing import List

from langchain_core.documents import Document

from core.interfaces.llm import ILanguageModel

logger = logging.getLogger(__name__)


class LLMWrapper(ILanguageModel):
    """Wraps an LLM with retry logic and automatic fallback.

    Implements pattern: Try primary LLM with retries, then fall back to secondary.
    """

    def __init__(
        self,
        primary_llm: ILanguageModel,
        fallback_llm: ILanguageModel,
        max_retries: int = 2,
        initial_delay_seconds: float = 1.0,
    ):
        """Initialize with primary and fallback LLMs.

        Args:
            primary_llm: Primary LLM (e.g., Gemini). Tried first with retries.
            fallback_llm: Fallback LLM (e.g., Local). Used if primary exhausts retries.
            max_retries: Number of times to retry primary before fallback.
            initial_delay_seconds: Initial delay between retries (exponential backoff).
        """
        self._primary_llm = primary_llm
        self._fallback_llm = fallback_llm
        self._max_retries = max_retries
        self._initial_delay_seconds = initial_delay_seconds
        logger.info(
            f"LLMWrapper initialized: "
            f"primary={primary_llm.get_model_name()}, "
            f"fallback={fallback_llm.get_model_name()}, "
            f"max_retries={max_retries}"
        )

    def summarize(self, documents: List[Document]) -> str:
        """Summarize documents with retry logic and fallback.

        Attempts to use primary LLM with exponential backoff retries.
        If primary fails after all retries, falls back to secondary LLM.

        Args:
            documents: List of LangChain Document objects.

        Returns:
            Summarized text from either primary or fallback LLM.
        """
        logger.info(
            f"Attempting summarization with primary LLM ({self._primary_llm.get_model_name()})"
        )

        delay = self._initial_delay_seconds
        for attempt in range(self._max_retries + 1):
            try:
                logger.info(
                    f"Attempt {attempt + 1}/{self._max_retries + 1} "
                    f"with {self._primary_llm.get_model_name()}"
                )
                result = self._primary_llm.summarize(documents)
                logger.info("Primary LLM summarization succeeded")
                return result

            except Exception as e:
                logger.warning(
                    f"Attempt {attempt + 1} failed: {e}. "
                    f"Retries remaining: {self._max_retries - attempt}"
                )

                if attempt < self._max_retries:
                    logger.info(f"Waiting {delay}s before retry...")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff

        # All retries exhausted, use fallback
        logger.warning(
            f"Primary LLM ({self._primary_llm.get_model_name()}) exhausted. "
            f"Falling back to {self._fallback_llm.get_model_name()}"
        )
        try:
            result = self._fallback_llm.summarize(documents)
            logger.info("Fallback LLM summarization succeeded")
            return result
        except Exception as e:
            logger.error(f"Fallback LLM also failed: {e}")
            raise

    def get_model_name(self) -> str:
        """Return model identifier showing both primary and fallback."""
        return f"{self._primary_llm.get_model_name()}[+{self._fallback_llm.get_model_name()}]"

    def refine(
        self,
        documents: List[Document],
        current_thesis_text: str,
        feedback_items: List[str],
    ) -> str:
        """Refine thesis with retry logic and fallback.

        Attempts to use primary LLM with exponential backoff retries.
        If primary fails, falls back to secondary LLM.
        Re-raises NotImplementedError immediately (no retry, no fallback).

        Args:
            documents: Source documents for context.
            current_thesis_text: Original thesis to refine.
            feedback_items: Feedback constraints from user.

        Returns:
            Refined thesis text.
        """
        logger.info(
            f"Attempting refinement with primary LLM ({self._primary_llm.get_model_name()})"
        )

        delay = self._initial_delay_seconds
        for attempt in range(self._max_retries + 1):
            try:
                logger.info(
                    f"Attempt {attempt + 1}/{self._max_retries + 1} "
                    f"with {self._primary_llm.get_model_name()}"
                )
                result = self._primary_llm.refine(
                    documents, current_thesis_text, feedback_items
                )
                logger.info("Primary LLM refinement succeeded")
                return result

            except NotImplementedError:
                logger.error(
                    f"{self._primary_llm.get_model_name()} does not support refinement"
                )
                raise  # Do not retry or fall back on not-implemented

            except Exception as e:
                logger.warning(
                    f"Attempt {attempt + 1} failed: {e}. "
                    f"Retries remaining: {self._max_retries - attempt}"
                )

                if attempt < self._max_retries:
                    logger.info(f"Waiting {delay}s before retry...")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff

        # All retries exhausted, use fallback
        logger.warning(
            f"Primary LLM ({self._primary_llm.get_model_name()}) exhausted. "
            f"Falling back to {self._fallback_llm.get_model_name()}"
        )
        try:
            result = self._fallback_llm.refine(
                documents, current_thesis_text, feedback_items
            )
            logger.info("Fallback LLM refinement succeeded")
            return result
        except Exception as e:
            logger.error(f"Fallback LLM also failed: {e}")
            raise
