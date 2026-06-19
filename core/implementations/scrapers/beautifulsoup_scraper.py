"""BeautifulSoup-based web scraper implementation."""

import logging

import requests
from bs4 import BeautifulSoup, Tag

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

            # Strip known non-article blocks that TechCrunch renders inline. 
            for noise in soup.select(
                "[class*=wp-block-techcrunch-event-cta],"
                "[class*=wp-block-techcrunch-most-popular-posts],"
                "[class*=wp-block-techcrunch-post-authors],"
                "[class*=affiliate-disclaimer-text]"
            ):
                noise.decompose()

            # Scope extraction to the article body when present, so we don't
            # pull in sidebars ("Most Popular"), related-article lists, author
            # bios, or topic/affiliate footers. Try selectors most-specific
            # first: a comma-separated select_one returns the first match in
            # *document* order, which on TechCrunch is the broad <main> wrapper
            # enclosing all that chrome. Iterating preserves priority so the
            # tight post-content container wins. Fall back to the whole document.
            root: Tag = soup
            for selector in (
                "div.wp-block-post-content",
                "div.entry-content",
                "article",
                "main",
            ):
                match = soup.select_one(selector)
                if match:
                    root = match
                    break

            # Capture all body text blocks, not just <p>: subheadings, list
            # items, and blockquotes are part of the article too. Keep the
            # length threshold low so short-but-real sentences (e.g. ledes)
            # aren't dropped.
            blocks = [
                el.get_text(strip=True)
                for el in root.find_all(["p", "h2", "h3", "li", "blockquote"])
                if len(el.get_text(strip=True)) > 1
            ]

            # Join with newlines, not spaces: downstream clean_article_text
            # strips boilerplate line-by-line (its patterns anchor on \n). A
            # space-joined blob collapses to one line, so a single boilerplate
            # match would delete everything to the end of the article.
            text = "\n".join(blocks).strip()
            logger.debug(f"Successfully scraped {len(blocks)} text blocks from {url}")
            return text

        except Exception as e:
            logger.warning(f"Failed to scrape {url}: {e}")
            return ""
