"""Abstract interface for the Silver verdict store"""

from abc import ABC, abstractmethod
from typing import List, Set

from core.models.silver_record import SilverVerdict


class ISilverRepository(ABC):
    """Records which Bronze articles have been processed into Silver.

    Holds one verdict per processed URL (fintech-relevant or not) so the Silver
    build skips already-decided articles and never re-runs classification on
    them. This is separate from the embedded vector store, which only holds the
    accepted (fintech) subset.
    """

    @abstractmethod
    def processed_urls(self) -> Set[str]:
        """Return the set of URLs already processed (any verdict)."""
        pass

    @abstractmethod
    def record(self, verdicts: List[SilverVerdict]) -> int:
        """Persist verdicts, skipping URLs already recorded.

        Returns:
            The number of verdicts newly recorded.
        """
        pass