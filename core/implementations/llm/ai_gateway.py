"""AI Gateway for LLM routing with caching and a daily call-budget guardrail.

Dollar cost is tracked by Langfuse (accurate, self-maintaining per-model
pricing), not here. The gateway confines itself to routing (size- and
budget-based), caching, and lightweight usage counting.
"""

import asyncio
import logging
import time
from typing import List, Dict

from langchain_core.documents import Document

from core.interfaces.cache import ICacheManager
from core.interfaces.llm import (
    SOURCE_LLM,
    SOURCE_LOCAL,
    ILanguageModel,
    summary_source_var,
)
from core.implementations.llm.usage_tracker import UsageTracker
from core.implementations.llm.routing_strategy import ROUTE_PRIMARY, get_strategy
from core.models.cache_entry import FALLBACK_MODEL
from config.settings import AIGatewayConfig

logger = logging.getLogger(__name__)


class AIGateway(ILanguageModel):
    """AI Gateway wrapper for LLM routing.

    Provides:
    - Response caching to reduce API calls
    - Usage counting (calls/tokens; dollars live in Langfuse)
    - Intelligent provider routing (size- and budget-based)
    - Seamless fallback to the local summarizer
    """

    def __init__(
        self,
        primary_llm: ILanguageModel,
        fallback_llm: ILanguageModel,
        config: AIGatewayConfig,
        cache_manager: ICacheManager,
        usage_tracker: UsageTracker,
    ):
        """Initialize AI Gateway.

        Args:
            primary_llm: Primary LLM implementation.
            fallback_llm: Fallback LLM implementation.
            config: AI Gateway configuration.
            cache_manager: Cache manager instance.
            usage_tracker: Usage tracker (call counts / tokens).
        """
        self._primary_llm = primary_llm
        self._fallback_llm = fallback_llm
        self._config = config
        self._cache_manager = cache_manager
        self._usage_tracker = usage_tracker
        self._routing_strategy = get_strategy(config.strategy)

        logger.info(
            f"AI Gateway initialized: strategy={config.strategy}, "
            f"cache_enabled={config.cache_enabled}, "
            f"call_budget_daily={config.call_budget_daily}"
        )

    def _get_documents_text(self, documents: List[Document]) -> str:
        """Concatenate document contents for cache key generation.

        Args:
            documents: List of documents.

        Returns:
            Concatenated text.
        """
        return "".join(doc.page_content for doc in documents)

    async def _cache_fallback_summary(
        self, docs_text: str, topic: str, result: str
    ) -> None:
        """Persist a degraded local-fallback summary under a short-lived entry.

        Written with FALLBACK_MODEL so the cache manager ages it out on a short
        TTL: rapid identical repeats during an outage / budget window are
        served from cache instead of re-running the extractor, but the real LLM
        is retried again soon after it recovers.
        """
        if not self._config.cache_enabled:
            return
        cache_key = self._cache_manager.generate_key(docs_text, topic, "combined")
        await asyncio.to_thread(
            self._cache_manager.set,
            key=cache_key,
            response=result,
            model=FALLBACK_MODEL,
            input_tokens=0,
            output_tokens=0,
        )

    def _select_provider(self, documents: List[Document], topic: str) -> tuple[str, ILanguageModel]:
        """Use the routing strategy to pick a route and resolve the LLM.

        The strategy decides a role (primary vs fallback), not a vendor; the
        label returned here is that role ("primary" / "local"), and callers
        log the resolved llm's own model name - so logs and usage records stay
        truthful whichever provider LLM_PROVIDER wires in as primary.

        Args:
            documents: List of documents.
            topic: Query topic.

        Returns:
            Tuple of (provider_label, llm_instance).
        """
        daily_calls = self._usage_tracker.get_daily_calls()

        route = self._routing_strategy.select_route(
            documents=documents,
            topic=topic,
            daily_calls=daily_calls,
            call_budget=self._config.call_budget_daily,
        )

        if route == ROUTE_PRIMARY:
            return "primary", self._primary_llm
        return "local", self._fallback_llm

    async def summarize(self, documents: List[Document], topic: str = "") -> str:
        """Summarize documents with caching, cost optimization, and routing.

        Flow:
        1. Check cache for identical documents + topic → return if hit
        2. Enforce the daily call budget (fall back to local when reached)
        3. Use routing strategy to select provider
        4. Call selected provider
        5. Cache result
        6. Track usage
        7. Return summary

        Args:
            documents: List of LangChain Document objects.
            topic: The user's query.

        Returns:
            Summarized text.

        Raises:
            Exception: If both primary and fallback LLMs fail.
        """
        start_time = time.time()
        docs_text = self._get_documents_text(documents)

        # Step 1: Check cache
        if self._config.cache_enabled:
            cache_key = self._cache_manager.generate_key(docs_text, topic, "combined")
            cached_entry = await asyncio.to_thread(self._cache_manager.get, cache_key)

            if cached_entry is not None:
                logger.info(f"Cache hit for documents: {cache_key}")
                # Restore provenance from the entry: a cached local-extractive
                # summary must stay marked as such on later hits (no model runs
                # on this path, so nothing else would set it).
                summary_source_var.set(
                    SOURCE_LOCAL if "local" in cached_entry.model else SOURCE_LLM
                )
                # Record cache hit
                if self._config.track_metrics:
                    latency_ms = (time.time() - start_time) * 1000
                    self._usage_tracker.record_call(
                        provider="cache",
                        model="cache",
                        input_tokens=0,
                        output_tokens=0,
                        latency_ms=latency_ms,
                        cache_hit=True,
                    )
                return cached_entry.response

        # Step 2: Enforce the daily call budget
        daily_calls = self._usage_tracker.get_daily_calls()
        if daily_calls >= self._config.call_budget_daily:
            logger.warning(
                f"Daily call budget reached: {daily_calls} >= "
                f"{self._config.call_budget_daily}"
            )
            # Use fallback to stay within budget
            try:
                logger.info("Using fallback LLM due to call budget")
                result = await self._fallback_llm.summarize(documents, topic)
                await self._cache_fallback_summary(docs_text, topic, result)
                latency_ms = (time.time() - start_time) * 1000
                if self._config.track_metrics:
                    self._usage_tracker.record_call(
                        provider="local",
                        model="local-extractor",
                        input_tokens=0,
                        output_tokens=0,
                        latency_ms=latency_ms,
                    )
                return result
            except Exception as e:
                logger.error(f"Fallback LLM failed: {e}")
                raise

        # Step 3: Select provider based on routing strategy
        provider, llm = self._select_provider(documents, topic)

        # Step 4: Call selected provider
        try:
            logger.info(f"Calling {llm.get_model_name()} for summarization ({provider} route)")
            result = await llm.summarize(documents, topic)

            # Step 5: Cache result
            if self._config.cache_enabled:
                cache_key = self._cache_manager.generate_key(docs_text, topic, "combined")
                # Estimate tokens (rough approximation)
                input_tokens = len(docs_text.split()) * 1.3
                output_tokens = len(result.split()) * 1.3
                await asyncio.to_thread(
                    self._cache_manager.set,
                    key=cache_key,
                    response=result,
                    model=llm.get_model_name(),
                    input_tokens=int(input_tokens),
                    output_tokens=int(output_tokens),
                )

            # Step 6: Track usage
            if self._config.track_metrics:
                latency_ms = (time.time() - start_time) * 1000
                input_tokens = len(docs_text.split()) * 1.3
                output_tokens = len(result.split()) * 1.3
                calls = self._usage_tracker.record_call(
                    provider=provider,
                    model=llm.get_model_name(),
                    input_tokens=int(input_tokens),
                    output_tokens=int(output_tokens),
                    latency_ms=latency_ms,
                )
                logger.info(
                    f"{llm.get_model_name()} summarization succeeded "
                    f"(daily primary calls={calls})"
                )

            return result

        except Exception as e:
            logger.error(f"{llm.get_model_name()} failed: {e}")
            # Try fallback
            try:
                logger.info("Attempting fallback LLM after primary failure")
                result = await self._fallback_llm.summarize(documents, topic)
                await self._cache_fallback_summary(docs_text, topic, result)

                if self._config.track_metrics:
                    latency_ms = (time.time() - start_time) * 1000
                    self._usage_tracker.record_call(
                        provider="local",
                        model="local-extractor",
                        input_tokens=0,
                        output_tokens=0,
                        latency_ms=latency_ms,
                    )

                return result
            except Exception as fallback_error:
                logger.error(f"Fallback LLM also failed: {fallback_error}")
                raise

    async def refine(
        self,
        documents: List[Document],
        current_thesis_text: str,
        feedback_items: List[str],
    ) -> str:
        """Refine thesis with caching, cost optimization, and routing.

        Delegates to primary LLM, with caching based on thesis + feedback hash.

        Args:
            documents: Source documents for context.
            current_thesis_text: Original thesis to refine.
            feedback_items: Feedback constraints from user.

        Returns:
            Refined thesis text.
        """
        start_time = time.time()
        topic = "fintech"  # Default topic

        # Generate cache key from thesis + feedback
        feedback_key = "".join(sorted(feedback_items))
        cache_input = current_thesis_text + feedback_key

        # Step 1: Check cache
        if self._config.cache_enabled:
            cache_key = self._cache_manager.generate_key(cache_input, topic, "refine")
            cached_entry = await asyncio.to_thread(self._cache_manager.get, cache_key)

            if cached_entry is not None:
                logger.info(f"Cache hit for refinement: {cache_key}")
                if self._config.track_metrics:
                    latency_ms = (time.time() - start_time) * 1000
                    self._usage_tracker.record_call(
                        provider="cache",
                        model="cache",
                        input_tokens=0,
                        output_tokens=0,
                        latency_ms=latency_ms,
                        cache_hit=True,
                    )
                return cached_entry.response

        # Step 2: Enforce the daily call budget
        daily_calls = self._usage_tracker.get_daily_calls()
        if daily_calls >= self._config.call_budget_daily:
            logger.warning(
                f"Daily call budget reached: {daily_calls} >= "
                f"{self._config.call_budget_daily}"
            )
            # Use fallback to stay within budget
            try:
                logger.info("Using fallback LLM due to call budget")
                result = await self._fallback_llm.refine(
                    documents, current_thesis_text, feedback_items
                )
                latency_ms = (time.time() - start_time) * 1000
                if self._config.track_metrics:
                    self._usage_tracker.record_call(
                        provider="local",
                        model="local-extractor",
                        input_tokens=0,
                        output_tokens=0,
                        latency_ms=latency_ms,
                    )
                return result
            except Exception as e:
                logger.error(f"Fallback LLM failed during refinement: {e}")
                raise

        # Step 3: Select provider
        provider, llm = self._select_provider(documents, topic)

        # Step 4: Call selected provider
        try:
            logger.info(f"Calling {llm.get_model_name()} for refinement ({provider} route)")
            result = await llm.refine(documents, current_thesis_text, feedback_items)

            # Step 5: Cache result
            if self._config.cache_enabled:
                cache_key = self._cache_manager.generate_key(cache_input, topic, "refine")
                input_tokens = len(cache_input.split()) * 1.3
                output_tokens = len(result.split()) * 1.3
                await asyncio.to_thread(
                    self._cache_manager.set,
                    key=cache_key,
                    response=result,
                    model=llm.get_model_name(),
                    input_tokens=int(input_tokens),
                    output_tokens=int(output_tokens),
                )

            # Step 6: Track usage
            if self._config.track_metrics:
                latency_ms = (time.time() - start_time) * 1000
                input_tokens = len(cache_input.split()) * 1.3
                output_tokens = len(result.split()) * 1.3
                calls = self._usage_tracker.record_call(
                    provider=provider,
                    model=llm.get_model_name(),
                    input_tokens=int(input_tokens),
                    output_tokens=int(output_tokens),
                    latency_ms=latency_ms,
                )
                logger.info(
                    f"{llm.get_model_name()} refinement succeeded "
                    f"(daily primary calls={calls})"
                )

            return result

        except Exception as e:
            logger.error(f"{llm.get_model_name()} failed during refinement: {e}")
            # Try fallback
            try:
                logger.info("Attempting fallback LLM after primary failure")
                result = await self._fallback_llm.refine(
                    documents, current_thesis_text, feedback_items
                )

                if self._config.track_metrics:
                    latency_ms = (time.time() - start_time) * 1000
                    self._usage_tracker.record_call(
                        provider="local",
                        model="local-extractor",
                        input_tokens=0,
                        output_tokens=0,
                        latency_ms=latency_ms,
                    )

                return result
            except Exception as fallback_error:
                logger.error(f"Fallback LLM also failed during refinement: {fallback_error}")
                raise

    def get_model_name(self) -> str:
        """Return model identifier with gateway info."""
        return f"AIGateway[{self._primary_llm.get_model_name()}+{self._fallback_llm.get_model_name()}]"

    def get_metrics(self) -> Dict:
        """Get gateway metrics including cache and usage stats.

        Returns:
            Dictionary with metrics.
        """
        cache_metrics = self._cache_manager.get_metrics() if self._config.cache_enabled else {}
        usage_metrics = self._usage_tracker.get_metrics() if self._config.track_metrics else {}

        return {
            "gateway_enabled": True,
            "strategy": self._config.strategy,
            "cache": cache_metrics,
            "usage": usage_metrics,
        }
