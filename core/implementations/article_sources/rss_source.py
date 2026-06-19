"""RSS feed article source implementation."""

import logging
from datetime import datetime, timezone
from typing import List, Optional
from urllib.parse import urlparse

import feedparser

from config.settings import RSSFeedConfig
from core.exceptions import NoArticlesFetchedError, NoRelevantArticlesError
from core.interfaces.article_source import IArticleSource
from core.interfaces.relevance_classifier import IRelevanceClassifier
from core.interfaces.scraper import IWebScraper
from core.models.article import Article
from core.models.raw_article import RawArticle
from core.utils.data_quality import check_ingestion
from core.utils.text_utils import clean_article_text

logger = logging.getLogger(__name__)


class RSSArticleSource(IArticleSource):
    """Fetches articles from RSS feeds."""

    def __init__(
        self,
        feeds: List[RSSFeedConfig],
        scraper: IWebScraper,
        classifier: Optional[IRelevanceClassifier] = None,
    ):
        self._feeds = feeds
        self._scraper = scraper
        self._classifier = classifier

    def fetch_articles(self, query: str, limit: int) -> List[Article]:
        """Fetch up to `limit` articles from configured RSS feeds.

        Each entry is first classified by its RSS <title> + <description> (when
        a classifier is configured); only relevant entries get scraped and kept.

        Raises:
            NoArticlesFetchedError: The feeds returned no usable entries (empty
                or unreachable).
            NoRelevantArticlesError: Entries were fetched but the classifier
                rejected every one of them.
        """
        entries = self._collect_entries(limit)
        if not entries:
            raise NoArticlesFetchedError(
                "No articles returned from RSS feeds (feed empty or unreachable)."
            )

        articles = []
        relevant = 0
        for entry in entries:
            title = entry.get("title", "")
            description = entry.get("description", "")

            if self._classifier is not None and not self._classifier.is_relevant(title, description):
                continue
            relevant += 1

            url = entry.get("link", "")
            if not url:
                continue

            text = self._scraper.scrape(url) or description or "No content available"
            article = self._build_article(entry, url, text)
            if article:
                articles.append(article)

        if self._classifier is not None:
            logger.info(f"Fintech classifier kept {relevant}/{len(entries)} entries")

        if not articles:
            if self._classifier is not None and relevant == 0:
                raise NoRelevantArticlesError(
                    f"Fetched {len(entries)} articles but none were classified as fintech."
                )
            raise NoArticlesFetchedError(
                "No valid articles could be built from the fetched entries."
            )

        logger.info(f"Fetched {len(articles)} articles from RSS feeds")
        return articles

    def collect_raw(self, limit: int) -> List[RawArticle]:
        """Collect feed entries.

        Unlike fetch_articles, this does NOT classify or scrape: it lands the
        raw feed entry (title, summary, link, <pubDate>, feed name) so the
        corpus accumulates cheaply.
        """
        entries = self._collect_entries(limit)
        raw_articles = []
        for entry in entries:
            url = entry.get("link", "")
            published_at = self._parse_published(entry)
            if not url or published_at is None:
                continue
            try:
                raw_articles.append(
                    RawArticle(
                        title=entry.get("title", "Untitled"),
                        url=url,
                        published_at=published_at,
                        summary=entry.get("description", ""),
                        source=urlparse(url).netloc,
                        feed_name=entry.get("_feed_name", ""),
                    )
                )
            except ValueError as e:
                logger.warning(f"Invalid raw article skipped: {e}")
        check_ingestion(seen=len(entries), landed=len(raw_articles))
        return raw_articles

    def _collect_entries(self, limit: int) -> list:
        """Parse all enabled feeds and return their entries, deduped by link.

        Each feed contributes up to `limit` (new) entries; entries already seen
        from an earlier feed are skipped so overlapping feeds (e.g. the general
        feed and the fintech category/tag feeds) don't yield duplicates.

        feedparser does not raise on network/DNS/timeout failures; it flags
        `feed.bozo` and returns no entries. Each failure mode is logged
        distinctly so an unreachable feed is not mistaken for an empty one.
        """
        seen_links = set()
        entries = []
        for feed_config in self._feeds:
            if not feed_config.enabled:
                continue

            try:
                logger.info(f"Parsing feed: {feed_config.name}")
                feed = feedparser.parse(feed_config.url)
            except Exception as e:
                logger.warning(f"Error parsing {feed_config.name}: {e}")
                continue

            if feed.bozo and not feed.entries:
                logger.warning(
                    f"Feed error for {feed_config.name} (unreachable or malformed): "
                    f"{feed.get('bozo_exception')}"
                )
                continue

            if not feed.entries:
                logger.warning(f"No entries in feed: {feed_config.name}")
                continue

            added = 0
            for entry in feed.entries:
                if added >= limit:
                    break
                link = entry.get("link", "")
                if link and link in seen_links:
                    continue
                if link:
                    seen_links.add(link)
                # Stash provenance so collect_raw can record which feed an
                # entry came from (fetch_articles ignores this key).
                entry["_feed_name"] = feed_config.name
                entries.append(entry)
                added += 1
            logger.info(f"Collected {added} new entries from {feed_config.name}")
        return entries

    def _build_article(self, entry, url: str, text: str) -> Optional[Article]:
        title = entry.get("title", "Untitled")
        published_at = self._parse_published(entry)
        if published_at is None:
            logger.warning(f"Skipping article without a <pubDate>: {url}")
            return None
        try:
            text = clean_article_text(text)
            return Article(
                title=title,
                text=text[:4000],
                source=urlparse(url).netloc,
                url=url,
                published_at=published_at,
            )
        except ValueError as e:
            logger.warning(f"Invalid article skipped: {e}")
            return None

    @staticmethod
    def _parse_published(entry) -> Optional[datetime]:
        """Extract the feed entry's <pubDate> as a UTC datetime.

        feedparser pre-parses <pubDate> into `published_parsed` (a
        time.struct_time in UTC). Returns None when it is absent or unparseable
        so the caller skips the entry rather than placing it on the time axis
        with a fabricated date.
        """
        parsed = entry.get("published_parsed")
        if not parsed:
            return None
        try:
            year, month, day, hour, minute, second = parsed[:6]
            return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
        except (TypeError, ValueError):
            return None

    def get_source_name(self) -> str:
        """Return the name of this article source."""
        return "RSS Feeds"
