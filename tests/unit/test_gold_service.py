"""Unit tests for GoldService (fintech corpus -> per-(dimension, category) weekly trends)."""

from datetime import date, datetime, timezone

from core.models.article import Article
from core.services.gold_service import GoldService

# Jan 2026: Jan 5 and Jan 12 are Mondays.


class _FakeContentRepo:
    def __init__(self, articles):
        self._articles = articles

    def fetch_all(self):
        return self._articles


class _FakeSilverRepo:
    """Returns pre-computed tags per fintech URL (assigned at Silver time)."""

    def __init__(self, tags_by_url):
        self._tags_by_url = dict(tags_by_url)

    def fintech_tags(self):
        return self._tags_by_url


class _FakeTrendRepo:
    def __init__(self):
        self.upserted = []

    def upsert(self, metrics):
        self.upserted = list(metrics)
        return len(metrics)


class _FakeUntaggedRepo:
    def __init__(self):
        self.saved = []

    def save(self, articles):
        self.saved = list(articles)
        return len(articles)


def _raw(url, day):
    return Article(
        title="t",
        text="body",
        url=url,
        published_at=datetime(2026, 1, day, tzinfo=timezone.utc),
        source="x.com",
    )


def _tags(themes=None, risks=None, signals=None):
    return {"themes": themes or [], "risks": risks or [], "signals": signals or []}


def _service(articles, tags_by_url, trend_repo, untagged_repo=None):
    return GoldService(
        content_repository=_FakeContentRepo(articles),
        silver_repository=_FakeSilverRepo(tags_by_url),
        trend_repository=trend_repo,
        untagged_repository=untagged_repo or _FakeUntaggedRepo(),
    )


def _by_key(metrics):
    return {(m.week_start, m.dimension, m.category): m.article_count for m in metrics}


def test_buckets_by_week_dimension_and_category_excluding_non_fintech():
    articles = [_raw("https://x/1", 7), _raw("https://x/2", 8), _raw("https://x/3", 14), _raw("https://x/4", 7)]
    tags = {
        "https://x/1": _tags(themes=["Payments"], risks=["Regulatory Risk"]),       # wk Jan 5
        "https://x/2": _tags(themes=["Crypto"], signals=["Crypto & Web3 Opportunity"]),  # wk Jan 5
        "https://x/3": _tags(themes=["Payments", "Crypto"]),                        # wk Jan 12, both themes
        # x/4 absent from tags -> not fintech -> excluded
    }
    trend_repo = _FakeTrendRepo()
    written = _service(articles, tags, trend_repo).build()

    assert _by_key(trend_repo.upserted) == {
        (date(2026, 1, 5), "theme", "Payments"): 1,
        (date(2026, 1, 5), "theme", "Crypto"): 1,
        (date(2026, 1, 5), "risk", "Regulatory Risk"): 1,
        (date(2026, 1, 5), "signal", "Crypto & Web3 Opportunity"): 1,
        (date(2026, 1, 12), "theme", "Payments"): 1,
        (date(2026, 1, 12), "theme", "Crypto"): 1,
    }
    assert written == len(trend_repo.upserted)


def test_same_week_dimension_category_accumulates():
    articles = [_raw("https://x/1", 5), _raw("https://x/2", 7)]
    tags = {
        "https://x/1": _tags(themes=["Payments"], risks=["Regulatory Risk"]),
        "https://x/2": _tags(themes=["Payments"], risks=["Regulatory Risk"]),
    }
    trend_repo = _FakeTrendRepo()
    _service(articles, tags, trend_repo).build()

    assert _by_key(trend_repo.upserted) == {
        (date(2026, 1, 5), "theme", "Payments"): 2,
        (date(2026, 1, 5), "risk", "Regulatory Risk"): 2,
    }


def test_no_theme_is_untagged_but_risks_and_signals_still_counted():
    # Accepted (fintech) but matched no theme, yet carries a risk and a signal.
    articles = [_raw("https://x/1", 7)]
    tags = {"https://x/1": _tags(risks=["Cybersecurity Risk"], signals=["Payment Infrastructure"])}
    trend_repo = _FakeTrendRepo()
    untagged_repo = _FakeUntaggedRepo()
    written = _service(articles, tags, trend_repo, untagged_repo).build()

    # No theme -> captured as a taxonomy gap...
    assert [a.url for a in untagged_repo.saved] == ["https://x/1"]
    # ...but its risk and signal are still tallied (independent of themes).
    assert _by_key(trend_repo.upserted) == {
        (date(2026, 1, 5), "risk", "Cybersecurity Risk"): 1,
        (date(2026, 1, 5), "signal", "Payment Infrastructure"): 1,
    }
    assert written == 2
