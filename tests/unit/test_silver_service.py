"""Unit tests for SilverService (Bronze -> classify -> scrape -> embed)."""

from datetime import datetime, timezone

from core.interfaces.relevance_classifier import IRelevanceClassifier
from core.interfaces.scraper import IWebScraper
from core.models.raw_article import RawArticle
from core.services.silver_service import SilverService
from finthesis_internal.keyword_scoring_strategy import KeywordCountScoringStrategy

PUB = datetime(2026, 1, 1, tzinfo=timezone.utc)
THEMES = {"Payments": ["payment"], "Crypto": ["crypto"]}


class _FakeRepo:
    def __init__(self, articles):
        self._articles = articles

    def fetch_all(self):
        return self._articles


class _FakeSilverRepo:
    def __init__(self, processed=None):
        self._processed = set(processed or [])
        self.recorded = []

    def processed_urls(self):
        return self._processed

    def record(self, verdicts):
        self.recorded.extend(verdicts)
        return len(verdicts)


class _FakeQuarantineRepo:
    def __init__(self, quarantined=None):
        self._quarantined = set(quarantined or [])
        self.added = []

    def quarantined_urls(self):
        return self._quarantined

    def add(self, records):
        self.added.extend(records)
        return len(records)


class _TitleClassifier(IRelevanceClassifier):
    """Relevant only when the title contains 'fintech'."""

    def is_relevant(self, title: str, description: str) -> bool:
        return "fintech" in title.lower()


class _StubScraper(IWebScraper):
    """Returns a body that mentions 'payment' - a theme keyword not in titles."""

    def scrape(self, url: str) -> str:
        return f"full body discussing payment infrastructure for {url}"


class _FailingScraper(IWebScraper):
    """Mimics a scrape failure: the real scraper returns '' on any error."""

    def scrape(self, url: str) -> str:
        return ""


class _FakeVectorStore:
    def __init__(self):
        self.built = []

    def build(self, documents):
        self.built.extend(documents)
        return object()


def _raw(title, url, source="x.com"):
    return RawArticle(
        title=title, url=url, published_at=PUB, summary="desc", source=source
    )


def _service(articles, silver_repo, vs, quarantine_repo=None, scraper=None):
    return SilverService(
        _FakeRepo(articles),
        silver_repo,
        quarantine_repo or _FakeQuarantineRepo(),
        _TitleClassifier(),
        scraper or _StubScraper(),
        KeywordCountScoringStrategy(),
        THEMES,
        vs,
    )


def test_embeds_fintech_records_verdicts_with_themes_from_full_text():
    raw = [
        _raw("Fintech A", "https://x/1"),    # new + fintech    -> embed, themed
        _raw("Space B", "https://x/2"),       # new, not fintech -> verdict False
        _raw("Fintech C", "https://x/3"),     # already processed -> skipped
    ]
    silver_repo = _FakeSilverRepo(processed={"https://x/3"})
    quarantine_repo = _FakeQuarantineRepo()
    vs = _FakeVectorStore()

    embedded = _service(raw, silver_repo, vs, quarantine_repo).build()

    assert embedded == 1
    assert vs.built[0].metadata["url"] == "https://x/1"
    by_url = {v.url: v for v in silver_repo.recorded}
    assert by_url["https://x/1"].themes == ["Payments"]  # from scraped body
    assert by_url["https://x/2"].fintech_relevant is False
    assert quarantine_repo.added == []


def test_scrape_failure_is_quarantined_not_embedded():
    raw = [_raw("Fintech A", "https://x/1")]
    silver_repo = _FakeSilverRepo()
    quarantine_repo = _FakeQuarantineRepo()
    vs = _FakeVectorStore()

    embedded = _service(raw, silver_repo, vs, quarantine_repo, scraper=_FailingScraper()).build()

    assert embedded == 0
    assert vs.built == []
    assert silver_repo.recorded == []  # no verdict -> replayable after fix
    assert [(r.url, r.reason) for r in quarantine_repo.added] == [
        ("https://x/1", "scrape_failed")
    ]


def test_invalid_article_is_quarantined():
    # Empty source makes Article validation fail (scrape itself succeeds).
    raw = [_raw("Fintech A", "https://x/1", source="")]
    silver_repo = _FakeSilverRepo()
    quarantine_repo = _FakeQuarantineRepo()
    vs = _FakeVectorStore()

    embedded = _service(raw, silver_repo, vs, quarantine_repo).build()

    assert embedded == 0
    assert silver_repo.recorded == []
    assert [(r.url, r.reason) for r in quarantine_repo.added] == [
        ("https://x/1", "invalid_article")
    ]


def test_quarantined_urls_are_skipped():
    raw = [_raw("Fintech A", "https://x/1")]
    silver_repo = _FakeSilverRepo()
    quarantine_repo = _FakeQuarantineRepo(quarantined={"https://x/1"})
    vs = _FakeVectorStore()

    embedded = _service(raw, silver_repo, vs, quarantine_repo).build()

    assert embedded == 0
    assert vs.built == []
    assert silver_repo.recorded == []
    assert quarantine_repo.added == []  # not reprocessed


def test_no_new_articles_does_not_call_build():
    raw = [_raw("Fintech A", "https://x/1")]
    silver_repo = _FakeSilverRepo(processed={"https://x/1"})
    vs = _FakeVectorStore()

    assert _service(raw, silver_repo, vs).build() == 0
    assert vs.built == []
    assert silver_repo.recorded == []
