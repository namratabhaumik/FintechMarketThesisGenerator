"""Abstract interface for the Silver dead-letter / quarantine store."""

from abc import ABC, abstractmethod
from typing import List, Set

from core.models.quarantine_record import QuarantineRecord


class IQuarantineRepository(ABC):
    """Stores Bronze articles that failed Silver enrichment.

    Quarantined URLs are excluded from Silver processing, so they neither
    pollute the corpus nor get retried every run. A replay removes the row,
    after which the next Silver run re-attempts the URL.
    """

    @abstractmethod
    def add(self, records: List[QuarantineRecord]) -> int:
        """Park records, skipping any URL already quarantined.

        Returns:
            The number of records newly added.
        """
        pass

    @abstractmethod
    def quarantined_urls(self) -> Set[str]:
        """Return the set of currently quarantined URLs."""
        pass
