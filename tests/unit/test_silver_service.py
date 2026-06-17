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


class _TitleClassifier(IRelevanceClassifier):
    """Relevant only when the title contains 'fintech'."""

    def is_relevant(self, title: str, description: str) -> bool:
        return "fintech" in title.lower()


class _StubScraper(IWebScraper):
    """Returns a body that mentions 'payment' - a theme keyword not in titles."""

    def scrape(self, url: str) -> str:
        return f"full body discussing payment infrastructure for {url}"


class _FakeVectorStore:
    def __init__(self):
        self.built = []

    def build(self, documents):
        self.built.extend(documents)
        return object()


def _raw(title, url):
    return RawArticle(
        title=title, url=url, published_at=PUB, summary="desc", source="x.com"
    )


def _service(articles, silver_repo, vs):
    return SilverService(
        _FakeRepo(articles),
        silver_repo,
        _TitleClassifier(),
        _StubScraper(),
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
    vs = _FakeVectorStore()

    embedded = _service(raw, silver_repo, vs).build()

    assert embedded == 1
    assert len(vs.built) == 1
    assert vs.built[0].metadata["url"] == "https://x/1"

    by_url = {v.url: v for v in silver_repo.recorded}
    assert set(by_url) == {"https://x/1", "https://x/2"}
    # Theme came from the scraped body ("payment"), not the title "Fintech A".
    assert by_url["https://x/1"].fintech_relevant is True
    assert by_url["https://x/1"].themes == ["Payments"]
    # Rejected article: verdict recorded, no themes.
    assert by_url["https://x/2"].fintech_relevant is False
    assert by_url["https://x/2"].themes == []


def test_no_new_articles_does_not_call_build():
    raw = [_raw("Fintech A", "https://x/1")]
    silver_repo = _FakeSilverRepo(processed={"https://x/1"})
    vs = _FakeVectorStore()

    assert _service(raw, silver_repo, vs).build() == 0
    assert vs.built == []
    assert silver_repo.recorded == []
