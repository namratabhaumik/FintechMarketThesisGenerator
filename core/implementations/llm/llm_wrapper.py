"""LLM wrapper with retry logic and fallback."""

import logging
from typing import List, Optional

from langchain_core.documents import Document
from tenacity import (
    AsyncRetrying,
    before_sleep_log,
    retry_if_not_exception_type,
    stop_after_attempt,
    stop_after_delay,
    wait_exponential,
)

from core.interfaces.llm import ILanguageModel

logger = logging.getLogger(__name__)


class LLMWrapper(ILanguageModel):
    """Wraps an LLM with retry logic and automatic fallback.

    Implements pattern: Try primary LLM with retries, then fall back to
    secondary. Fallback applies to summarize only; refine propagates after
    retries (see refine's docstring).
    """

    def __init__(
        self,
        primary_llm: ILanguageModel,
        fallback_llm: ILanguageModel,
        max_retries: int = 2,
        initial_delay_seconds: float = 1.0,
        retry_budget_seconds: float = 60.0,
    ):
        """Initialize with primary and fallback LLMs.

        Args:
            primary_llm: Primary LLM (e.g., Gemini). Tried first with retries.
            fallback_llm: Fallback LLM (e.g., Local). Used if primary exhausts retries.
            max_retries: Number of times to retry primary before fallback.
            initial_delay_seconds: Initial delay between retries (exponential backoff).
            retry_budget_seconds: Total wall-clock ceiling across ALL attempts of
                one call. Retrying stops once it is exceeded even if attempts
                remain, so slow hangs (per-attempt timeouts) cannot stack past a
                platform gateway timeout: budget + one in-flight attempt is the true worst case.
        """
        self._primary_llm = primary_llm
        self._fallback_llm = fallback_llm
        self._max_retries = max_retries
        self._initial_delay_seconds = initial_delay_seconds
        self._retry_budget_seconds = retry_budget_seconds
        logger.info(
            f"LLMWrapper initialized: "
            f"primary={primary_llm.get_model_name()}, "
            f"fallback={fallback_llm.get_model_name()}, "
            f"max_retries={max_retries}, "
            f"retry_budget={retry_budget_seconds}s"
        )

    def _make_retrying(self, **kwargs) -> AsyncRetrying:
        return AsyncRetrying(
            # Whichever binds first: the attempt cap (fast failures) or the
            # wall-clock budget (slow hangs at the per-attempt timeout).
            stop=(
                stop_after_attempt(self._max_retries + 1)
                | stop_after_delay(self._retry_budget_seconds)
            ),
            wait=wait_exponential(
                multiplier=self._initial_delay_seconds,
                min=self._initial_delay_seconds,
                max=30,
            ),
            reraise=True,
            before_sleep=before_sleep_log(logger, logging.WARNING),
            **kwargs,
        )

    async def summarize(self, documents: List[Document], topic: str = "") -> str:
        """Summarize documents with retry logic and fallback.

        Args:
            documents: List of LangChain Document objects.
            topic: The user's query; passed through to whichever LLM serves.

        Returns:
            Summarized text from either primary or fallback LLM.
        """
        logger.info(
            f"Attempting summarization with primary LLM ({self._primary_llm.get_model_name()})"
        )
        try:
            async for attempt in self._make_retrying():
                with attempt:
                    result = await self._primary_llm.summarize(documents, topic)
            logger.info("Primary LLM summarization succeeded")
            return result
        except Exception:
            logger.warning(
                f"Primary LLM ({self._primary_llm.get_model_name()}) exhausted. "
                f"Falling back to {self._fallback_llm.get_model_name()}"
            )
            try:
                result = await self._fallback_llm.summarize(documents, topic)
                logger.info("Fallback LLM summarization succeeded")
                return result
            except Exception as e:
                logger.error(f"Fallback LLM also failed: {e}")
                raise

    def get_model_name(self) -> str:
        """Return the primary model identifier (the billable model on the success path)."""
        return self._primary_llm.get_model_name()

    async def refine(
        self,
        documents: List[Document],
        current_thesis_text: str,
        feedback_items: List[str],
        prior_feedback: Optional[List[List[str]]] = None,
    ) -> str:
        """Refine thesis with retry logic on the primary LLM only.

        Unlike summarize, refine has NO fallback: the local summarizer cannot
        rewrite a thesis (its refine raises NotImplementedError), so once the
        primary exhausts its retries the error propagates to the caller, which
        surfaces it and persists nothing.

        NotImplementedError is re-raised immediately — no retry.

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
        try:
            async for attempt in self._make_retrying(
                retry=retry_if_not_exception_type(NotImplementedError)
            ):
                with attempt:
                    result = await self._primary_llm.refine(
                        documents, current_thesis_text, feedback_items, prior_feedback
                    )
            logger.info("Primary LLM refinement succeeded")
            return result
        except NotImplementedError:
            logger.error(
                f"{self._primary_llm.get_model_name()} does not support refinement"
            )
            raise
        except Exception as e:
            logger.error(
                f"Primary LLM ({self._primary_llm.get_model_name()}) exhausted "
                f"retries; refinement failed (no fallback): {e}"
            )
            raise
