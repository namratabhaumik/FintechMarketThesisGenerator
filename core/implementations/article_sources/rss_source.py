"""RSS feed article source implementation."""

import logging
from datetime import datetime, timezone
from typing import List, Optional
from urllib.parse import urlparse

import feedparser

from config.settings import RSSFeedConfig
from core.models.raw_article import RawArticle
from core.utils.data_quality import check_ingestion

logger = logging.getLogger(__name__)


class RSSArticleSource:
    """Reads the configured RSS feeds and lands raw entries verbatim (Bronze).

    No classifying or scraping happens here - that is Silver's job, reading
    the Bronze rows this class produced.
    """

    def __init__(self, feeds: List[RSSFeedConfig]):
        # The feeds to read (each has a URL, a display name, and an enabled flag).
        self._feeds = feeds

    def collect_raw(self, limit: int) -> List[RawArticle]:
        """Collect feed entries.

        Lands the raw feed entry (title, summary, link, <pubDate>, feed name)
        so the corpus accumulates cheaply. Returns [] when the feeds are empty
        or unreachable; the caller reports the landed count.
        """
        entries = self._collect_entries(limit)
        # raw_articles: the Bronze records we successfully landed.
        raw_articles = []
        # Turn each deduped feed entry into a RawArticle.
        for entry in entries:
            url = entry.get("link", "")
            published_at = self._parse_published(entry)
            # No link or no usable <pubDate> --> skip.
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
                # RawArticle validation rejected it (e.g. bad field) --> log and
                # drop it.
                logger.warning(f"Invalid raw article skipped: {e}")
        # Record how many entries we saw vs how many actually landed, so a
        # silent drop in yield is visible in the data-quality logs.
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
        # seen_links: every entry link we've already taken, across all feeds.
        #   This is the dedup key: overlapping feeds (general + fintech category)
        # entries: the deduped feed entries we return.
        seen_links = set()
        entries = []
        # Read each configured feed in turn.
        for feed_config in self._feeds:
            # Disabled in config --> skip this feed entirely.
            if not feed_config.enabled:
                continue

            try:
                logger.info(f"Parsing feed: {feed_config.name}")
                feed = feedparser.parse(feed_config.url)
            except Exception as e:
                logger.warning(f"Error parsing {feed_config.name}: {e}")
                continue

            # feedparser does not raise on network/DNS/timeout problems: it sets
            # feed.bozo and returns no entries. So bozo + no entries --> treat as
            # unreachable/malformed and log that distinctly (not "empty").
            if feed.bozo and not feed.entries:
                logger.warning(
                    f"Feed error for {feed_config.name} (unreachable or malformed): "
                    f"{feed.get('bozo_exception')}"
                )
                continue

            # Reachable but genuinely had nothing --> log as empty and skip.
            if not feed.entries:
                logger.warning(f"No entries in feed: {feed_config.name}")
                continue

            # added: how many NEW (unseen) entries this feed contributed.
            # Each feed is capped at `limit` so one feed can't crowd out others.
            added = 0
            for entry in feed.entries:
                # Hit this feed's quota --> stop reading it.
                if added >= limit:
                    break
                link = entry.get("link", "")
                # Already saw this link in an earlier feed --> duplicate, skip it.
                if link and link in seen_links:
                    continue
                # Remember the link so later feeds won't re-add the same story.
                if link:
                    seen_links.add(link)
                # Stash provenance so collect_raw can record which feed an
                # entry came from.
                entry["_feed_name"] = feed_config.name
                entries.append(entry)
                added += 1
            logger.info(f"Collected {added} new entries from {feed_config.name}")
        return entries

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
