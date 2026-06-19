"""Abstract interface for the Silver verdict store"""

from abc import ABC, abstractmethod
from typing import Dict, List, Set

from core.models.silver_record import SilverVerdict


class ISilverRepository(ABC):
    """Silver layer: the verdict log of which articles have been decided.

    For every Bronze article Silver processes, it records exactly one frozen
    verdict here: "fintech-relevant" or "not". That verdict is permanent.

    Silver checks this log first --> if a URL already has a
    verdict, skip it --> classification never runs twice on the same article.
    Note this is just the decision log; the actual embedded vectors live in a
    separate store that only holds the accepted (fintech) subset.
    """

    @abstractmethod
    def processed_urls(self) -> Set[str]:
        """Return the set of URLs that already have a verdict (any outcome).

        Silver uses this to skip articles it has already decided on.
        """
        pass

    @abstractmethod
    def fintech_themes(self) -> Dict[str, List[str]]:
        """Return {url: themes} for the accepted (fintech-relevant) articles.

        Themes were assigned from the full scraped text when Silver ran. The
        Gold layer reads this mapping directly to count trends. An accepted
        article that matched no theme maps to an empty list.
        """
        pass

    @abstractmethod
    def record(self, verdicts: List[SilverVerdict]) -> int:
        """Save verdicts, skipping any URL that already has one.

        for each verdict --> if the URL is undecided, record it --> if it
        already has a verdict, leave the existing (frozen) one untouched.

        Returns:
            How many verdicts were newly recorded.
        """
        pass