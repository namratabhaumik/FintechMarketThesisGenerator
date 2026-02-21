"""Thesis structuring service - maps summary keywords to fintech category names."""

import logging
from typing import Dict, List

from core.services.category_mappings import ThemeMappings, RiskMappings, SignalMappings
from core.interfaces.scoring_strategy import IScoringStrategy
from core.interfaces.thesis_structurer import IThesisStructurer

logger = logging.getLogger(__name__)


class ThesisStructuringService(IThesisStructurer):
    """Lightweight service for structuring thesis data from summaries.

    Single Responsibility: Orchestrate keyword-to-category matching.
    Open/Closed: Category mappings and scoring strategy are injected.

    Uses keyword-to-category mappings with injectable scoring strategy.
    The categories with the highest scores are returned.
    """

    def __init__(
        self,
        scoring_strategy: IScoringStrategy,
        max_results: int = 3
    ):
        """Initialize structuring service.

        Args:
            scoring_strategy: Injected strategy for scoring categories.
            max_results: Maximum number of categories to return per type.
        """
        self._scoring_strategy = scoring_strategy
        self._max_results = max_results
        self._theme_map = ThemeMappings.get_mapping()
        self._risk_map = RiskMappings.get_mapping()
        self._signal_map = SignalMappings.get_mapping()

    def structure_thesis(self, summary: str) -> dict:
        """Map summary to structured category labels.

        Args:
            summary: Summarized text from documents.

        Returns:
            Dictionary with key_themes, risks, investment_signals.
        """
        logger.info("Structuring thesis from summary using category mapping")
        text_lower = summary.lower()

        return {
            "key_themes": self._match_categories(text_lower, self._theme_map.categories),
            "risks": self._match_categories(text_lower, self._risk_map.categories),
            "investment_signals": self._match_categories(text_lower, self._signal_map.categories),
        }

    def get_structurer_name(self) -> str:
        """Return the name of this structurer.

        Returns:
            Name of the structurer implementation.
        """
        return "KeywordMappingStructurer"

    def _match_categories(
        self,
        text_lower: str,
        category_map: Dict[str, List[str]]
    ) -> List[str]:
        """Score and rank categories based on keyword matches.

        Args:
            text_lower: Lowercased summary text.
            category_map: Dict of {label: [trigger_keywords]}.

        Returns:
            Up to max_results category labels with at least one match,
            sorted by score descending.
        """
        # Use injected scoring strategy
        scored = self._scoring_strategy.score(text_lower, category_map)

        # Filter out zero-score categories, sort by score descending,
        # return top N labels
        return [
            label
            for label, score in sorted(scored.items(), key=lambda x: x[1], reverse=True)
            if score > 0
        ][:self._max_results]

