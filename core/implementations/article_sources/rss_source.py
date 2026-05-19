"""RSS feed article source implementation."""

import asyncio
import concurrent.futures
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
        self._feeds = feeds
        self._scraper = scraper

    def fetch_articles(self, query: str, limit: int) -> List[Article]:
        """Fetch up to `limit` articles from configured RSS feeds.

        Entries are collected from all feeds first, then all URLs are scraped
        concurrently via asyncio so network latency is paid once (in parallel)
        rather than serially per article.
        """
        entries = self._collect_entries(limit)
        if not entries:
            logger.warning("No RSS entries found across all feeds")
            return []

        urls = [e.get("link", "") for e in entries]
        summaries = [e.get("summary", "No content available") for e in entries]

        texts = self._scrape_concurrently(urls)

        articles = []
        for entry, url, text, summary in zip(entries, urls, texts, summaries):
            if not url:
                continue
            if isinstance(text, Exception) or not text:
                text = summary
            article = self._build_article(entry, url, text)
            if article:
                articles.append(article)

        logger.info(f"Fetched {len(articles)} articles from RSS feeds")
        return articles

    def _collect_entries(self, limit: int) -> list:
        """Parse all enabled feeds and return up to `limit` entries."""
        entries = []
        for feed_config in self._feeds:
            if not feed_config.enabled or len(entries) >= limit:
                break
            try:
                logger.info(f"Parsing feed: {feed_config.name}")
                feed = feedparser.parse(feed_config.url)
                if not feed.entries:
                    logger.warning(f"No entries in feed: {feed_config.name}")
                    continue
                remaining = limit - len(entries)
                entries.extend(feed.entries[:remaining])
            except Exception as e:
                logger.warning(f"Error parsing {feed_config.name}: {e}")
        return entries

    def _scrape_concurrently(self, urls: List[str]) -> List[str]:
        """Scrape all URLs concurrently.

        Uses asyncio.gather + asyncio.to_thread when no event loop is running
        (FastAPI background task context). Falls back to a ThreadPoolExecutor
        when called from within an existing event loop (e.g. Streamlit).
        """
        try:
            asyncio.get_running_loop()
            # Already inside a running loop — delegate blocking I/O to threads directly
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(urls) or 1) as ex:
                return list(ex.map(self._scraper.scrape, urls))
        except RuntimeError:
            return asyncio.run(self._scrape_all_async(urls))

    async def _scrape_all_async(self, urls: List[str]) -> List[str]:
        tasks = [asyncio.to_thread(self._scraper.scrape, url) for url in urls]
        return await asyncio.gather(*tasks, return_exceptions=True)

    def _build_article(self, entry, url: str, text: str) -> Article:
        title = entry.get("title", "Untitled")
        try:
            text = clean_article_text(text)
            return Article(
                title=title,
                text=text[:4000],
                source=urlparse(url).netloc,
                url=url,
            )
        except ValueError as e:
            logger.warning(f"Invalid article skipped: {e}")
            return None

    def get_source_name(self) -> str:
        """Return the name of this article source."""
        return "RSS Feeds"
