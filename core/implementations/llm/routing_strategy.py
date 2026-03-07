"""Routing strategies for cost-optimized LLM provider selection."""

import logging
from abc import ABC, abstractmethod
from typing import List, Tuple

from langchain.docstore.document import Document

logger = logging.getLogger(__name__)


class RoutingStrategy(ABC):
    """Abstract base class for routing strategies."""

    @abstractmethod
    def select_provider(
        self,
        documents: List[Document],
        topic: str,
        daily_spend: float,
        daily_limit: float,
    ) -> Tuple[str, str]:
        """Select which provider and model to use.

        Args:
            documents: List of documents to summarize.
            topic: The query topic.
            daily_spend: Current daily spend in USD.
            daily_limit: Daily spend limit in USD.

        Returns:
            Tuple of (provider, model) to use.
        """


class CostOptimizedStrategy(RoutingStrategy):
    """Route to minimize cost while maintaining acceptable quality.

    Uses local summarizer for large documents or when cost limit approaching.
    Falls back to Gemini only for small, critical queries.
    """

    def __init__(self, size_threshold_tokens: int = 5000):
        """Initialize strategy.

        Args:
            size_threshold_tokens: Token count threshold for document size.
        """
        self._size_threshold = size_threshold_tokens

    def _estimate_tokens(self, documents: List[Document]) -> int:
        """Rough estimate of total tokens in documents (words * 1.3).

        Args:
            documents: List of documents.

        Returns:
            Estimated token count.
        """
        total_words = sum(len(doc.page_content.split()) for doc in documents)
        return int(total_words * 1.3)

    def select_provider(
        self,
        documents: List[Document],
        topic: str,
        daily_spend: float,
        daily_limit: float,
    ) -> Tuple[str, str]:
        """Select provider based on document size and cost limits.

        Strategy:
        1. If documents are large (>5000 tokens) → use local (free)
        2. If daily spend > 80% of limit → use local (cost containment)
        3. Otherwise → use gemini (better quality for small docs)

        Args:
            documents: List of documents.
            topic: Query topic.
            daily_spend: Current daily spend.
            daily_limit: Daily limit.

        Returns:
            Tuple of (provider, model).
        """
        estimated_tokens = self._estimate_tokens(documents)
        cost_ratio = daily_spend / daily_limit if daily_limit > 0 else 0

        if estimated_tokens > self._size_threshold:
            logger.info(f"Document size {estimated_tokens} > {self._size_threshold}: using local")
            return ("local", "local-extractor")

        if cost_ratio > 0.8:
            logger.info(f"Cost ratio {cost_ratio:.2%} > 80%: using local for cost containment")
            return ("local", "local-extractor")

        logger.info("Using gemini for cost-optimized routing")
        return ("gemini", "gemini-2.0-flash")


class QualityFirstStrategy(RoutingStrategy):
    """Route to maximize quality, cost is secondary concern."""

    def select_provider(
        self,
        documents: List[Document],
        topic: str,
        daily_spend: float,
        daily_limit: float,
    ) -> Tuple[str, str]:
        """Always prefer Gemini for best quality.

        Args:
            documents: List of documents.
            topic: Query topic.
            daily_spend: Current daily spend.
            daily_limit: Daily limit.

        Returns:
            Tuple of (provider, model).
        """
        logger.info("Using gemini for quality-first routing")
        return ("gemini", "gemini-2.0-flash")


class HybridStrategy(RoutingStrategy):
    """Balance quality and cost with intelligent decision-making.

    Uses gemini for small documents, local for large ones.
    Respects cost limits as hard constraint.
    """

    def __init__(self, size_threshold_tokens: int = 5000):
        """Initialize strategy.

        Args:
            size_threshold_tokens: Token count threshold for document size.
        """
        self._size_threshold = size_threshold_tokens

    def _estimate_tokens(self, documents: List[Document]) -> int:
        """Rough estimate of total tokens (words * 1.3).

        Args:
            documents: List of documents.

        Returns:
            Estimated token count.
        """
        total_words = sum(len(doc.page_content.split()) for doc in documents)
        return int(total_words * 1.3)

    def select_provider(
        self,
        documents: List[Document],
        topic: str,
        daily_spend: float,
        daily_limit: float,
    ) -> Tuple[str, str]:
        """Select provider balancing quality and cost.

        Strategy:
        1. If daily spend would exceed limit → use local (hard constraint)
        2. If documents are large (>5000 tokens) → use local (no benefit to API)
        3. If documents are small → use gemini (better quality on small summaries)

        Args:
            documents: List of documents.
            topic: Query topic.
            daily_spend: Current daily spend.
            daily_limit: Daily limit.

        Returns:
            Tuple of (provider, model).
        """
        estimated_tokens = self._estimate_tokens(documents)

        # Hard cost limit
        if daily_spend >= daily_limit:
            logger.info("Daily cost limit reached: using local")
            return ("local", "local-extractor")

        # Large documents → local (free, adequate quality)
        if estimated_tokens > self._size_threshold:
            logger.info(f"Document size {estimated_tokens} > {self._size_threshold}: using local")
            return ("local", "local-extractor")

        # Small documents → gemini (better quality)
        logger.info(f"Document size {estimated_tokens} <= {self._size_threshold}: using gemini")
        return ("gemini", "gemini-2.0-flash")


# Strategy registry
STRATEGY_REGISTRY = {
    "cost_optimized": CostOptimizedStrategy,
    "quality_first": QualityFirstStrategy,
    "hybrid": HybridStrategy,
}


def get_strategy(strategy_name: str, **kwargs) -> RoutingStrategy:
    """Get a routing strategy by name.

    Args:
        strategy_name: Name of strategy (cost_optimized, quality_first, hybrid).
        **kwargs: Additional arguments for strategy initialization.

    Returns:
        Initialized routing strategy.

    Raises:
        ValueError: If strategy name is unknown.
    """
    if strategy_name not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown routing strategy: {strategy_name}. "
            f"Supported: {list(STRATEGY_REGISTRY.keys())}"
        )

    strategy_class = STRATEGY_REGISTRY[strategy_name]
    return strategy_class(**kwargs)
