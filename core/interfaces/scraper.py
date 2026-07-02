"""Abstract interface for web scraping."""

from abc import ABC, abstractmethod


class IWebScraper(ABC):
    """Protocol for scraping article text from URLs."""

    @abstractmethod
    def scrape(self, url: str) -> str:
        """Scrape and return main text content from URL.

        Returns "" on a terminal failure (404/403, unparseable page) so the
        caller can quarantine. Raises TransientScrapeError on a retryable
        failure (timeout, connection, 5xx, 429) so the caller can leave the
        URL pending for a later run.
        """
        pass
