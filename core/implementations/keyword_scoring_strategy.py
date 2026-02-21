"""Keyword counting scoring strategy for category matching."""

from typing import Dict, List

from core.interfaces.scoring_strategy import IScoringStrategy


class KeywordCountScoringStrategy(IScoringStrategy):
    """Scores categories based on keyword match count.

    """

    def score(
        self,
        text: str,
        category_keywords: Dict[str, List[str]]
    ) -> Dict[str, int]:
        """Score each category by counting keyword matches.

        Args:
            text: Lowercased text to search.
            category_keywords: Dict mapping category labels to keyword lists.

        Returns:
            Dict mapping category labels to match counts.
        """
        return {
            label: sum(1 for kw in keywords if kw in text)
            for label, keywords in category_keywords.items()
        }
