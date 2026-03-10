"""AI Gateway for cost-optimized LLM routing with caching and cost tracking."""

import logging
import time
from typing import List, Dict

from langchain_core.documents import Document

from core.interfaces.llm import ILanguageModel
from core.implementations.llm.cache_manager import CacheManager
from core.implementations.llm.cost_tracker import CostTracker
from core.implementations.llm.routing_strategy import get_strategy, RoutingStrategy
from core.models.cache_entry import CacheEntry
from config.settings import AIGatewayConfig

logger = logging.getLogger(__name__)


class AIGateway(ILanguageModel):
    """AI Gateway wrapper for cost-optimized LLM usage.

    Provides:
    - Response caching to reduce API calls
    - Cost tracking and monitoring
    - Intelligent provider routing (cost vs quality tradeoff)
    - Seamless fallback to multiple providers
    """

    def __init__(
        self,
        primary_llm: ILanguageModel,
        fallback_llm: ILanguageModel,
        config: AIGatewayConfig,
        cache_manager: CacheManager,
        cost_tracker: CostTracker,
    ):
        """Initialize AI Gateway.

        Args:
            primary_llm: Primary LLM implementation.
            fallback_llm: Fallback LLM implementation.
            config: AI Gateway configuration.
            cache_manager: Cache manager instance.
            cost_tracker: Cost tracker instance.
        """
        self._primary_llm = primary_llm
        self._fallback_llm = fallback_llm
        self._config = config
        self._cache_manager = cache_manager
        self._cost_tracker = cost_tracker
        self._routing_strategy = get_strategy(config.strategy)

        logger.info(
            f"AI Gateway initialized: strategy={config.strategy}, "
            f"cache_enabled={config.cache_enabled}, "
            f"cost_limit_daily=${config.cost_limit_daily}"
        )

    def _get_documents_text(self, documents: List[Document]) -> str:
        """Concatenate document contents for cache key generation.

        Args:
            documents: List of documents.

        Returns:
            Concatenated text.
        """
        return "".join(doc.page_content for doc in documents)

    def _select_provider(self, documents: List[Document], topic: str) -> tuple[str, ILanguageModel]:
        """Use routing strategy to select provider and get LLM instance.

        Args:
            documents: List of documents.
            topic: Query topic.

        Returns:
            Tuple of (provider_name, llm_instance).
        """
        daily_spend = self._cost_tracker.get_daily_spend()

        provider, model = self._routing_strategy.select_provider(
            documents=documents,
            topic=topic,
            daily_spend=daily_spend,
            daily_limit=self._config.cost_limit_daily,
        )

        # Map provider to LLM instance
        if provider == "gemini":
            return provider, self._primary_llm
        else:
            return provider, self._fallback_llm

    def summarize(self, documents: List[Document]) -> str:
        """Summarize documents with caching, cost optimization, and routing.

        Flow:
        1. Check cache for identical documents → return if hit
        2. Use routing strategy to select provider
        3. Check cost limits
        4. Call selected provider
        5. Cache result
        6. Track cost
        7. Return summary

        Args:
            documents: List of LangChain Document objects.

        Returns:
            Summarized text.

        Raises:
            ValueError: If cost limit exceeded.
            Exception: If both primary and fallback LLMs fail.
        """
        start_time = time.time()
        docs_text = self._get_documents_text(documents)
        topic = "fintech"  # Default topic since not passed in

        # Step 1: Check cache
        if self._config.cache_enabled:
            cache_key = self._cache_manager.generate_key(docs_text, topic, "combined")
            cached_entry = self._cache_manager.get(cache_key)

            if cached_entry is not None:
                logger.info(f"Cache hit for documents: {cache_key}")
                # Record cache hit
                if self._config.track_metrics:
                    latency_ms = (time.time() - start_time) * 1000
                    self._cost_tracker.record_call(
                        provider="cache",
                        model="cache",
                        input_tokens=0,
                        output_tokens=0,
                        latency_ms=latency_ms,
                        cache_hit=True,
                    )
                return cached_entry.response

        # Step 2: Check cost limit
        daily_spend = self._cost_tracker.get_daily_spend()
        if daily_spend >= self._config.cost_limit_daily:
            logger.warning(f"Daily cost limit reached: ${daily_spend:.2f} >= ${self._config.cost_limit_daily}")
            # Use fallback to avoid charges
            try:
                logger.info("Using fallback LLM due to cost limit")
                result = self._fallback_llm.summarize(documents)
                latency_ms = (time.time() - start_time) * 1000
                if self._config.track_metrics:
                    self._cost_tracker.record_call(
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
            logger.info(f"Calling {provider} LLM for summarization")
            result = llm.summarize(documents)

            # Step 5: Cache result
            if self._config.cache_enabled:
                cache_key = self._cache_manager.generate_key(docs_text, topic, "combined")
                # Estimate tokens (rough approximation)
                input_tokens = len(docs_text.split()) * 1.3
                output_tokens = len(result.split()) * 1.3
                self._cache_manager.set(
                    key=cache_key,
                    response=result,
                    model=llm.get_model_name(),
                    input_tokens=int(input_tokens),
                    output_tokens=int(output_tokens),
                )

            # Step 6: Track cost
            if self._config.track_metrics:
                latency_ms = (time.time() - start_time) * 1000
                input_tokens = len(docs_text.split()) * 1.3
                output_tokens = len(result.split()) * 1.3
                cost = self._cost_tracker.record_call(
                    provider=provider,
                    model=llm.get_model_name(),
                    input_tokens=int(input_tokens),
                    output_tokens=int(output_tokens),
                    latency_ms=latency_ms,
                )
                logger.info(f"{provider} summarization succeeded - cost: ${cost:.6f}")

            return result

        except Exception as e:
            logger.error(f"{provider} LLM failed: {e}")
            # Try fallback
            try:
                logger.info("Attempting fallback LLM after primary failure")
                result = self._fallback_llm.summarize(documents)

                if self._config.track_metrics:
                    latency_ms = (time.time() - start_time) * 1000
                    self._cost_tracker.record_call(
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

    def refine(
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
        docs_text = self._get_documents_text(documents)
        topic = "fintech"  # Default topic

        # Generate cache key from thesis + feedback
        feedback_key = "".join(sorted(feedback_items))
        cache_input = current_thesis_text + feedback_key

        # Step 1: Check cache
        if self._config.cache_enabled:
            cache_key = self._cache_manager.generate_key(cache_input, topic, "refine")
            cached_entry = self._cache_manager.get(cache_key)

            if cached_entry is not None:
                logger.info(f"Cache hit for refinement: {cache_key}")
                if self._config.track_metrics:
                    latency_ms = (time.time() - start_time) * 1000
                    self._cost_tracker.record_call(
                        provider="cache",
                        model="cache",
                        input_tokens=0,
                        output_tokens=0,
                        latency_ms=latency_ms,
                        cache_hit=True,
                    )
                return cached_entry.response

        # Step 2: Check cost limit
        daily_spend = self._cost_tracker.get_daily_spend()
        if daily_spend >= self._config.cost_limit_daily:
            logger.warning(
                f"Daily cost limit reached: ${daily_spend:.2f} >= ${self._config.cost_limit_daily}"
            )
            # Use fallback to avoid charges
            try:
                logger.info("Using fallback LLM due to cost limit")
                result = self._fallback_llm.refine(
                    documents, current_thesis_text, feedback_items
                )
                latency_ms = (time.time() - start_time) * 1000
                if self._config.track_metrics:
                    self._cost_tracker.record_call(
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
            logger.info(f"Calling {provider} LLM for refinement")
            result = llm.refine(documents, current_thesis_text, feedback_items)

            # Step 5: Cache result
            if self._config.cache_enabled:
                cache_key = self._cache_manager.generate_key(cache_input, topic, "refine")
                input_tokens = len(cache_input.split()) * 1.3
                output_tokens = len(result.split()) * 1.3
                self._cache_manager.set(
                    key=cache_key,
                    response=result,
                    model=llm.get_model_name(),
                    input_tokens=int(input_tokens),
                    output_tokens=int(output_tokens),
                )

            # Step 6: Track cost
            if self._config.track_metrics:
                latency_ms = (time.time() - start_time) * 1000
                input_tokens = len(cache_input.split()) * 1.3
                output_tokens = len(result.split()) * 1.3
                cost = self._cost_tracker.record_call(
                    provider=provider,
                    model=llm.get_model_name(),
                    input_tokens=int(input_tokens),
                    output_tokens=int(output_tokens),
                    latency_ms=latency_ms,
                )
                logger.info(f"{provider} refinement succeeded - cost: ${cost:.6f}")

            return result

        except Exception as e:
            logger.error(f"{provider} LLM failed during refinement: {e}")
            # Try fallback
            try:
                logger.info("Attempting fallback LLM after primary failure")
                result = self._fallback_llm.refine(
                    documents, current_thesis_text, feedback_items
                )

                if self._config.track_metrics:
                    latency_ms = (time.time() - start_time) * 1000
                    self._cost_tracker.record_call(
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
        """Get gateway metrics including cache and cost stats.

        Returns:
            Dictionary with metrics.
        """
        cache_metrics = self._cache_manager.get_metrics() if self._config.cache_enabled else {}
        cost_metrics = self._cost_tracker.get_metrics() if self._config.track_metrics else {}

        return {
            "gateway_enabled": True,
            "strategy": self._config.strategy,
            "cache": cache_metrics,
            "costs": cost_metrics,
        }
