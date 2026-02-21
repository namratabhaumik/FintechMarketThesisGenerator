"""RSS feed article source implementation."""

import logging
from typing import List
from urllib.parse import urlparse

import feedparser

from config.settings import RSSFeedConfig
from core.interfaces.article_source import IArticleSource
from core.interfaces.scraper import IWebScraper
from core.models.article import Article
from core.utils.text_utils import clean_article_text

logger = logging.getLogger(__name__)


class RSSArticleSource(IArticleSource):
    """Fetches articles from RSS feeds."""

    def __init__(self, feeds: List[RSSFeedConfig], scraper: IWebScraper):
        """Initialize with RSS feed configurations and web scraper.

        Args:
            feeds: List of RSS feed configurations.
            scraper: Web scraper implementation (dependency injection).
        """
        self._feeds = feeds
        self._scraper = scraper

    def fetch_articles(self, query: str, limit: int) -> List[Article]:
        """Fetch articles from configured RSS feeds.

        Args:
            query: Search query (currently unused, feeds are pre-filtered).
            limit: Maximum number of articles to fetch.

        Returns:
            List of Article objects.
        """
        articles = []

        for feed_config in self._feeds:
            if not feed_config.enabled:
                continue

            if len(articles) >= limit:
                break

            try:
                logger.info(f"Fetching from: {feed_config.name}")
                feed = feedparser.parse(feed_config.url)

                if not feed.entries:
                    logger.warning(f"No entries found in feed: {feed_config.name}")
                    continue

                for entry in feed.entries[:limit - len(articles)]:
                    article = self._process_entry(entry)
                    if article:
                        articles.append(article)

                    if len(articles) >= limit:
                        break

            except Exception as e:
                logger.warning(f"Error fetching from {feed_config.name}: {e}")
                continue

        logger.info(f"Fetched {len(articles)} articles from RSS feeds")
        return articles

    def _process_entry(self, entry) -> Article:
        """Process a single RSS entry into an Article.

        Args:
            entry: feedparser entry object.

        Returns:
            Article object or None if entry is invalid.
        """
        title = entry.get("title", "Untitled")
        url = entry.get("link", "")

        if not url:
            return None

        try:
            # Use injected scraper to get article text
            text = self._scraper.scrape(url)
            if not text:
                # Fallback to summary if scraping fails
                text = entry.get("summary", "No content available")

            # Clean text before storing (removes ads, normalizes whitespace)
            text = clean_article_text(text)

            source = urlparse(url).netloc

            article = Article(
                title=title,
                text=text[:4000],  # Limit for embeddings
                source=source,
                url=url
            )
            logger.info(f"Successfully processed article: {title[:60]}...")
            return article

        except ValueError as e:
            logger.warning(f"Invalid article skipped: {e}")
            return None

    def get_source_name(self) -> str:
        """Return the name of this article source."""
        return "RSS Feeds"
