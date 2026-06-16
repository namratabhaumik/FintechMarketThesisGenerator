"""Unit tests for SilverService (Bronze -> classify -> scrape -> embed)."""

from datetime import datetime, timezone

from core.interfaces.relevance_classifier import IRelevanceClassifier
from core.interfaces.scraper import IWebScraper
from core.models.raw_article import RawArticle
from core.services.silver_service import SilverService

PUB = datetime(2026, 1, 1, tzinfo=timezone.utc)


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
    def scrape(self, url: str) -> str:
        return f"full scraped body for {url}"


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


def test_embeds_fintech_records_all_verdicts_and_skips_processed():
    raw = [
        _raw("Fintech A", "https://x/1"),    # new + fintech    -> embed + verdict True
        _raw("Space B", "https://x/2"),       # new, not fintech -> verdict False, no embed
        _raw("Fintech C", "https://x/3"),     # already processed -> skipped entirely
    ]
    silver_repo = _FakeSilverRepo(processed={"https://x/3"})
    vs = _FakeVectorStore()
    svc = SilverService(_FakeRepo(raw), silver_repo, _TitleClassifier(), _StubScraper(), vs)

    embedded = svc.build()

    assert embedded == 1
    assert len(vs.built) == 1
    doc = vs.built[0]
    assert doc.metadata["url"] == "https://x/1"
    assert doc.metadata["published_at"] == PUB.isoformat()
    assert "full scraped body" in doc.page_content

    # A verdict is recorded for both decided articles (the rejected one too),
    # so neither is re-classified on a later run. The processed one is untouched.
    recorded = {v.url: v.fintech_relevant for v in silver_repo.recorded}
    assert recorded == {"https://x/1": True, "https://x/2": False}


def test_no_new_articles_does_not_call_build():
    raw = [_raw("Fintech A", "https://x/1")]
    silver_repo = _FakeSilverRepo(processed={"https://x/1"})
    vs = _FakeVectorStore()
    svc = SilverService(_FakeRepo(raw), silver_repo, _TitleClassifier(), _StubScraper(), vs)

    assert svc.build() == 0
    assert vs.built == []
    assert silver_repo.recorded == []
