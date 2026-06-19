"""Abstract interface for the Silver dead-letter / quarantine store."""

from abc import ABC, abstractmethod
from typing import List, Set

from core.models.quarantine_record import QuarantineRecord


class IQuarantineRepository(ABC):
    """Silver layer: the dead-letter store for articles that failed enrichment.

    When Silver tries to enrich a Bronze article (scrape, validate, embed) and
    something goes wrong --> the article is parked here instead of silently
    disappearing or being retried forever.

    Effect on later runs: Silver checks this store and skips any quarantined URL
    --> the broken article does not pollute the corpus and is not retried every
    run. To give a URL another chance --> remove its row (a "replay") --> the
    next Silver run will attempt it again.
    """

    @abstractmethod
    def add(self, records: List[QuarantineRecord]) -> int:
        """Park failed records, skipping any URL already quarantined.

        for each record --> if its URL is not already quarantined, add it
        --> otherwise skip it.

        Returns:
            How many records were newly added.
        """
        pass

    @abstractmethod
    def quarantined_urls(self) -> Set[str]:
        """Return the set of URLs currently quarantined.

        Silver uses this set to know which articles to skip.
        """
        pass
