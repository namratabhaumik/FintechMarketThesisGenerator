"""BeautifulSoup-based web scraper implementation."""

import logging
import re

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
            Extracted text content from the URL, or "" if anything fails.

        fetch the page --> parse HTML --> strip noise --> narrow to
        the article body --> collect text blocks --> join with newlines.
        """
        try:
            # Fetch the page. A custom User-Agent avoids bot blocks; the timeout
            # keeps one slow site from stalling the whole run.
            resp = requests.get(
                url,
                timeout=self._config.timeout,
                headers={"User-Agent": self._config.user_agent}
            )
            # Turn any 4xx/5xx HTTP status into an exception (caught below).
            resp.raise_for_status()

            # Parse the raw HTML into a navigable tree we can query and prune.
            soup = BeautifulSoup(resp.text, "html.parser")

            # Remove unwanted elements: scripts, styling, and page chrome (nav,
            # header, footer, sidebars) are not article body --> delete them.
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
            # root: the subtree we will read text from. Start at the whole
            # document, then try to narrow it to the tightest article container.
            root: Tag = soup
            for selector in (
                "div.wp-block-post-content",
                "div.entry-content",
                "article",
                "main",
            ):
                # First selector that matches wins (most-specific first) -->
                # narrow root to it and stop --> if none match, root stays the
                # whole document.
                match = soup.select_one(selector)
                if match:
                    root = match
                    break

            # Capture all body text blocks, not just <p>: subheadings, list
            # items, and blockquotes are part of the article too. Keep the
            # length threshold low so short-but-real sentences (e.g. ledes)
            # aren't dropped.
            # blocks: the cleaned text of each body element inside root. We walk
            # paragraphs, subheadings, list items, and quotes (all article
            # content) and keep each element's trimmed text, dropping anything
            # 1 char or shorter (stray markup, bullets). separator=" " puts a
            # space between adjacent inline elements (links, spans) so their text
            # isn't fused ("Digital Bank" + "Grasshopper" -> "Digital
            # BankGrasshopper"); \s+ then collapses any doubles it introduces.
            blocks = []
            for el in root.find_all(["p", "h2", "h3", "li", "blockquote"]):
                block = re.sub(r"\s+", " ", el.get_text(separator=" ", strip=True)).strip()
                if len(block) > 1:
                    blocks.append(block)

            # Join with newlines, not spaces: downstream clean_article_text
            # strips boilerplate line-by-line (its patterns anchor on \n). A
            # space-joined blob collapses to one line, so a single boilerplate
            # match would delete everything to the end of the article.
            text = "\n".join(blocks).strip()
            logger.debug(f"Successfully scraped {len(blocks)} text blocks from {url}")
            return text

        except Exception as e:
            # Any failure (network error, bad status, parse issue) --> log and
            # return "" so the caller can fall back to the RSS description.
            logger.warning(f"Failed to scrape {url}: {e}")
            return ""
