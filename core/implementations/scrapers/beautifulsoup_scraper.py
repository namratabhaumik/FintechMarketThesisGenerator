"""BeautifulSoup-based web scraper implementation."""

import logging

import requests
from bs4 import BeautifulSoup

from config.settings import ScraperConfig
from core.interfaces.scraper import IWebScraper

logger = logging.getLogger(__name__)


class BeautifulSoupScraper(IWebScraper):
    """Web scraper using BeautifulSoup4."""

    def __init__(self, config: ScraperConfig):
        """Initialize scraper with configuration.

        Args:
            config: Scraper configuration with timeout and user-agent.
        """
        self._config = config

    def scrape(self, url: str) -> str:
        """Scrape article text from URL.

        Args:
            url: URL to scrape.

        Returns:
            Extracted text content from the URL.
        """
        try:
            resp = requests.get(
                url,
                timeout=self._config.timeout,
                headers={"User-Agent": self._config.user_agent}
            )
            resp.raise_for_status()

            soup = BeautifulSoup(resp.text, "html.parser")

            # Remove unwanted elements
            for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
                tag.decompose()

            # Extract paragraphs
            paragraphs = [
                p.get_text(strip=True)
                for p in soup.find_all("p")
                if len(p.get_text(strip=True)) > 40
            ]

            text = " ".join(paragraphs).strip()
            logger.debug(f"Successfully scraped {len(paragraphs)} paragraphs from {url}")
            return text

        except Exception as e:
            logger.warning(f"Failed to scrape {url}: {e}")
            return ""
