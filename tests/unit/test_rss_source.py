"""Unit tests for RSSArticleSource fintech filtering."""

from types import SimpleNamespace

import pytest

from core.exceptions import NoArticlesFetchedError, NoRelevantArticlesError
from core.implementations.article_sources.rss_source import RSSArticleSource
from core.interfaces.relevance_classifier import IRelevanceClassifier
from core.interfaces.scraper import IWebScraper


# feedparser pre-parses <pubDate> into a time.struct_time-like 9-tuple (UTC).
PUB_PARSED = (2026, 1, 1, 12, 0, 0, 0, 1, 0)


class StubScraper(IWebScraper):
    def scrape(self, url: str) -> str:
        return f"body of {url}"


class TitleClassifier(IRelevanceClassifier):
    """Relevant only when the title contains 'fintech'."""

    def is_relevant(self, title: str, description: str) -> bool:
        return "fintech" in title.lower()


class _FakeFeed(dict):
    """Dict that also supports attribute access, like feedparser's result."""
    __getattr__ = dict.get


def _patch_feed(monkeypatch, entries, bozo=False, bozo_exception=None):
    feed = _FakeFeed(entries=entries, bozo=bozo, bozo_exception=bozo_exception)
    monkeypatch.setattr(
        "core.implementations.article_sources.rss_source.feedparser.parse",
        lambda url: feed,
    )


def _patch_feeds_by_url(monkeypatch, url_to_entries):
    """Patch feedparser.parse to return different entries per feed URL."""
    feeds = {
        url: _FakeFeed(entries=entries, bozo=False, bozo_exception=None)
        for url, entries in url_to_entries.items()
    }
    monkeypatch.setattr(
        "core.implementations.article_sources.rss_source.feedparser.parse",
        lambda url: feeds[url],
    )


def _feed_config(name="TechCrunch", url="https://techcrunch.com/feed/"):
    return SimpleNamespace(name=name, url=url, enabled=True)


def test_filters_out_non_fintech_entries(monkeypatch):
    """Only entries the classifier approves are scraped and returned."""
    _patch_feed(
        monkeypatch,
        [
            {"title": "A fintech payments startup", "description": "payments", "link": "https://x/1", "published_parsed": PUB_PARSED},
            {"title": "A space rocket launch", "description": "space", "link": "https://x/2", "published_parsed": PUB_PARSED},
            {"title": "Another fintech bank", "description": "banking", "link": "https://x/3", "published_parsed": PUB_PARSED},
        ],
    )
    source = RSSArticleSource([_feed_config()], StubScraper(), classifier=TitleClassifier())

    articles = source.fetch_articles("fintech", limit=10)

    titles = [a.title for a in articles]
    assert titles == ["A fintech payments startup", "Another fintech bank"]


def test_no_classifier_keeps_everything(monkeypatch):
    """Without a classifier, all entries pass through (back-compat)."""
    _patch_feed(
        monkeypatch,
        [
            {"title": "A space rocket launch", "description": "space", "link": "https://x/2", "published_parsed": PUB_PARSED},
        ],
    )
    source = RSSArticleSource([_feed_config()], StubScraper(), classifier=None)

    articles = source.fetch_articles("fintech", limit=10)

    assert len(articles) == 1


def test_dedupes_overlapping_entries_across_feeds(monkeypatch):
    """An article appearing in two feeds is collected (and scraped) only once."""
    shared = {"title": "A fintech bank launches", "description": "banking", "link": "https://x/shared", "published_parsed": PUB_PARSED}
    _patch_feeds_by_url(
        monkeypatch,
        {
            "https://techcrunch.com/feed/": [
                shared,
                {"title": "A space launch", "description": "space", "link": "https://x/space", "published_parsed": PUB_PARSED},
            ],
            "https://techcrunch.com/category/fintech/feed/": [
                shared,  # same link -> should be deduped
                {"title": "A fintech wallet app", "description": "payments", "link": "https://x/wallet", "published_parsed": PUB_PARSED},
            ],
        },
    )
    feeds = [
        _feed_config("General", "https://techcrunch.com/feed/"),
        _feed_config("Fintech", "https://techcrunch.com/category/fintech/feed/"),
    ]
    source = RSSArticleSource(feeds, StubScraper(), classifier=TitleClassifier())

    articles = source.fetch_articles("fintech", limit=10)

    urls = [a.url for a in articles]
    assert urls.count("https://x/shared") == 1
    assert set(urls) == {"https://x/shared", "https://x/wallet"}


def test_parses_published_date(monkeypatch):
    """The entry's <pubDate> is parsed into the Article's published_at (UTC)."""
    _patch_feed(
        monkeypatch,
        [
            {"title": "A fintech bank", "description": "banking", "link": "https://x/1", "published_parsed": PUB_PARSED},
        ],
    )
    source = RSSArticleSource([_feed_config()], StubScraper(), classifier=TitleClassifier())

    articles = source.fetch_articles("fintech", limit=10)

    assert len(articles) == 1
    published = articles[0].published_at
    assert (published.year, published.month, published.day) == (2026, 1, 1)
    assert published.tzinfo is not None


def test_entry_without_pubdate_is_skipped(monkeypatch):
    """An entry with no <pubDate> has no place on the time axis and is dropped."""
    _patch_feed(
        monkeypatch,
        [
            {"title": "A fintech bank", "description": "banking", "link": "https://x/1"},  # no published_parsed
        ],
    )
    source = RSSArticleSource([_feed_config()], StubScraper(), classifier=TitleClassifier())

    with pytest.raises(NoArticlesFetchedError):
        source.fetch_articles("fintech", limit=10)


def test_all_non_fintech_raises_no_relevant(monkeypatch):
    """When the classifier rejects every entry, NoRelevantArticlesError is raised."""
    _patch_feed(
        monkeypatch,
        [
            {"title": "A space rocket launch", "description": "space", "link": "https://x/1"},
            {"title": "A new gadget review", "description": "hardware", "link": "https://x/2"},
        ],
    )
    source = RSSArticleSource([_feed_config()], StubScraper(), classifier=TitleClassifier())

    with pytest.raises(NoRelevantArticlesError):
        source.fetch_articles("fintech", limit=10)


def test_empty_feed_raises_no_articles(monkeypatch):
    """An empty feed raises NoArticlesFetchedError."""
    _patch_feed(monkeypatch, [])
    source = RSSArticleSource([_feed_config()], StubScraper(), classifier=TitleClassifier())

    with pytest.raises(NoArticlesFetchedError):
        source.fetch_articles("fintech", limit=10)


def test_unreachable_feed_raises_no_articles(monkeypatch):
    """A bozo (unreachable/malformed) feed with no entries raises NoArticlesFetchedError."""
    _patch_feed(monkeypatch, [], bozo=True, bozo_exception=OSError("DNS failure"))
    source = RSSArticleSource([_feed_config()], StubScraper(), classifier=TitleClassifier())

    with pytest.raises(NoArticlesFetchedError):
        source.fetch_articles("fintech", limit=10)
