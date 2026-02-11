"""Abstract interface for web scraping."""

from abc import ABC, abstractmethod


class IWebScraper(ABC):
    """Protocol for scraping article text from URLs."""

    @abstractmethod
    def scrape(self, url: str) -> str:
        """Scrape and return main text content from URL."""
        pass
