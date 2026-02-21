"""Abstract interface for thesis structurers."""

from abc import ABC, abstractmethod
from typing import Dict, List


class IThesisStructurer(ABC):
    """Protocol for structuring thesis data from summaries.

    """

    @abstractmethod
    def structure_thesis(self, summary: str) -> Dict[str, List[str]]:
        """Map summary to structured category labels.

        Args:
            summary: Summarized text from documents.

        Returns:
            Dictionary with key_themes, risks, investment_signals keys.
        """
        pass

    @abstractmethod
    def get_structurer_name(self) -> str:
        """Return the name of this structurer."""
        pass
