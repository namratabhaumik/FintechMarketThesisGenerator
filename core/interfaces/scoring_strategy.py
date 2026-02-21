"""Abstract interface for category scoring strategies."""

from abc import ABC, abstractmethod
from typing import Dict, List


class IScoringStrategy(ABC):
    """Protocol for scoring categories based on keyword matches.

    """

    @abstractmethod
    def score(
        self,
        text: str,
        category_keywords: Dict[str, List[str]]
    ) -> Dict[str, int]:
        """Score each category based on text content.

        Args:
            text: Lowercased text to search.
            category_keywords: Dict mapping category labels to keyword lists.

        Returns:
            Dict mapping category labels to numeric scores.
        """
        pass
