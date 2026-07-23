"""Unit tests for RSSArticleSource (Bronze raw collection)."""

from types import SimpleNamespace

from core.implementations.article_sources.rss_source import RSSArticleSource


# feedparser pre-parses <pubDate> into a time.struct_time-like 9-tuple (UTC).
PUB_PARSED = (2026, 1, 1, 12, 0, 0, 0, 1, 0)


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


def test_collect_raw_lands_entries_verbatim(monkeypatch):
    """collect_raw keeps every dated entry (no classification) with provenance."""
    _patch_feed(
        monkeypatch,
        [
            {"title": "A fintech bank", "description": "banking", "link": "https://x/1", "published_parsed": PUB_PARSED},
            {"title": "A space launch", "description": "space", "link": "https://x/2", "published_parsed": PUB_PARSED},
        ],
    )
    source = RSSArticleSource([_feed_config(name="TechCrunch")])

    raw = source.collect_raw(limit=10)

    assert [r.title for r in raw] == ["A fintech bank", "A space launch"]
    assert raw[0].summary == "banking"
    assert raw[0].source == "x"  # netloc of https://x/1
    assert raw[0].feed_name == "TechCrunch"
    assert raw[0].published_at.year == 2026


def test_collect_raw_parses_published_date(monkeypatch):
    """The entry's <pubDate> lands as a timezone-aware UTC published_at."""
    _patch_feed(
        monkeypatch,
        [
            {"title": "A fintech bank", "description": "banking", "link": "https://x/1", "published_parsed": PUB_PARSED},
        ],
    )
    source = RSSArticleSource([_feed_config()])

    raw = source.collect_raw(limit=10)

    assert len(raw) == 1
    published = raw[0].published_at
    assert (published.year, published.month, published.day) == (2026, 1, 1)
    assert published.tzinfo is not None


def test_collect_raw_skips_dateless_entries(monkeypatch):
    """An entry without <pubDate> has no time-axis slot and is dropped."""
    _patch_feed(
        monkeypatch,
        [
            {"title": "no date", "description": "d", "link": "https://x/1"},
            {"title": "has date", "description": "d", "link": "https://x/2", "published_parsed": PUB_PARSED},
        ],
    )
    source = RSSArticleSource([_feed_config()])

    raw = source.collect_raw(limit=10)

    assert [r.title for r in raw] == ["has date"]


def test_collect_raw_dedupes_overlapping_entries_across_feeds(monkeypatch):
    """An article appearing in two feeds is landed only once."""
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
    source = RSSArticleSource(feeds)

    raw = source.collect_raw(limit=10)

    urls = [r.url for r in raw]
    assert urls.count("https://x/shared") == 1
    assert set(urls) == {"https://x/shared", "https://x/space", "https://x/wallet"}


def test_collect_raw_empty_feed_returns_nothing(monkeypatch):
    """An empty feed lands zero rows (the cron caller reports the count)."""
    _patch_feed(monkeypatch, [])
    source = RSSArticleSource([_feed_config()])

    assert source.collect_raw(limit=10) == []


def test_collect_raw_unreachable_feed_returns_nothing(monkeypatch):
    """A bozo (unreachable/malformed) feed with no entries lands zero rows."""
    _patch_feed(monkeypatch, [], bozo=True, bozo_exception=OSError("DNS failure"))
    source = RSSArticleSource([_feed_config()])

    assert source.collect_raw(limit=10) == []
