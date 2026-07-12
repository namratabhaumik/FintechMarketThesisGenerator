"""Routing strategies for cost-optimized LLM route selection.

A strategy decides a ROUTE, not a vendor: "primary" (the configured LLM,
whatever LLM_PROVIDER names) or "fallback" (the free local extractive
summarizer). The gateway resolves the route to an actual model, so nothing
here needs to know which provider is wired in.
"""

import logging
from abc import ABC, abstractmethod
from typing import List

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Route names the strategies can return.
ROUTE_PRIMARY = "primary"
ROUTE_FALLBACK = "fallback"


class RoutingStrategy(ABC):
    """Abstract base class for routing strategies."""

    @abstractmethod
    def select_route(
        self,
        documents: List[Document],
        topic: str,
        daily_spend: float,
        daily_limit: float,
    ) -> str:
        """Select which route to use.

        Args:
            documents: List of documents to summarize.
            topic: The query topic.
            daily_spend: Current daily spend in USD.
            daily_limit: Daily spend limit in USD.

        Returns:
            ROUTE_PRIMARY or ROUTE_FALLBACK.
        """


def _estimate_tokens(documents: List[Document]) -> int:
    """Rough estimate of total tokens in documents (words * 1.3)."""
    total_words = sum(len(doc.page_content.split()) for doc in documents)
    return int(total_words * 1.3)


class CostOptimizedStrategy(RoutingStrategy):
    """Route to minimize cost while maintaining acceptable quality.

    Uses the local summarizer for large documents or when the cost limit is
    approaching; the primary LLM only for small, critical queries.
    """

    def __init__(self, size_threshold_tokens: int = 5000):
        """Initialize strategy.

        Args:
            size_threshold_tokens: Token count threshold for document size.
        """
        self._size_threshold = size_threshold_tokens

    def select_route(
        self,
        documents: List[Document],
        topic: str,
        daily_spend: float,
        daily_limit: float,
    ) -> str:
        """Select route based on document size and cost limits.

        Strategy:
        1. If documents are large (>threshold tokens) -> fallback (free)
        2. If daily spend > 80% of limit -> fallback (cost containment)
        3. Otherwise -> primary (better quality for small docs)
        """
        estimated_tokens = _estimate_tokens(documents)
        cost_ratio = daily_spend / daily_limit if daily_limit > 0 else 0

        if estimated_tokens > self._size_threshold:
            logger.info(
                f"Document size {estimated_tokens} > {self._size_threshold}: "
                f"routing to the local summarizer"
            )
            return ROUTE_FALLBACK

        if cost_ratio > 0.8:
            logger.info(
                f"Cost ratio {cost_ratio:.2%} > 80%: routing to the local "
                f"summarizer for cost containment"
            )
            return ROUTE_FALLBACK

        logger.info("Cost-optimized routing: using the primary LLM")
        return ROUTE_PRIMARY


class QualityFirstStrategy(RoutingStrategy):
    """Route to maximize quality, cost is secondary concern."""

    def select_route(
        self,
        documents: List[Document],
        topic: str,
        daily_spend: float,
        daily_limit: float,
    ) -> str:
        """Always prefer the primary LLM for best quality."""
        logger.info("Quality-first routing: using the primary LLM")
        return ROUTE_PRIMARY


class HybridStrategy(RoutingStrategy):
    """Balance quality and cost with intelligent decision-making.

    Uses the primary LLM for small documents, the local summarizer for large
    ones. Respects cost limits as a hard constraint.
    """

    def __init__(self, size_threshold_tokens: int = 5000):
        """Initialize strategy.

        Args:
            size_threshold_tokens: Token count threshold for document size.
        """
        self._size_threshold = size_threshold_tokens

    def select_route(
        self,
        documents: List[Document],
        topic: str,
        daily_spend: float,
        daily_limit: float,
    ) -> str:
        """Select route balancing quality and cost.

        Strategy:
        1. If daily spend would exceed limit -> fallback (hard constraint)
        2. If documents are large (>threshold) -> fallback (no benefit to API)
        3. If documents are small -> primary (better quality)
        """
        estimated_tokens = _estimate_tokens(documents)

        # Hard cost limit
        if daily_spend >= daily_limit:
            logger.info("Daily cost limit reached: routing to the local summarizer")
            return ROUTE_FALLBACK

        # Large documents -> local summarizer (free, adequate quality)
        if estimated_tokens > self._size_threshold:
            logger.info(
                f"Document size {estimated_tokens} > {self._size_threshold}: "
                f"routing to the local summarizer"
            )
            return ROUTE_FALLBACK

        # Small documents -> primary LLM (better quality)
        logger.info(
            f"Document size {estimated_tokens} <= {self._size_threshold}: "
            f"routing to the primary LLM"
        )
        return ROUTE_PRIMARY


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
