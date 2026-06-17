"""Unit tests for GoldService (fintech corpus -> per-theme weekly trends)."""

from datetime import date, datetime, timezone

from core.models.raw_article import RawArticle
from core.services.gold_service import GoldService

# Jan 2026: Jan 5 and Jan 12 are Mondays.


class _FakeArticleRepo:
    def __init__(self, articles):
        self._articles = articles

    def fetch_all(self):
        return self._articles


class _FakeSilverRepo:
    """Returns pre-computed themes per fintech URL (assigned at Silver time)."""

    def __init__(self, themes_by_url):
        self._themes_by_url = dict(themes_by_url)

    def fintech_themes(self):
        return self._themes_by_url


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
    return RawArticle(
        title="t",
        url=url,
        published_at=datetime(2026, 1, day, tzinfo=timezone.utc),
        summary="",
        source="x.com",
    )


def _service(articles, themes_by_url, trend_repo, untagged_repo=None):
    return GoldService(
        article_repository=_FakeArticleRepo(articles),
        silver_repository=_FakeSilverRepo(themes_by_url),
        trend_repository=trend_repo,
        untagged_repository=untagged_repo or _FakeUntaggedRepo(),
    )


def test_buckets_by_week_and_theme_excluding_non_fintech():
    articles = [_raw("https://x/1", 7), _raw("https://x/2", 8), _raw("https://x/3", 14), _raw("https://x/4", 7)]
    themes = {
        "https://x/1": ["Payments"],            # wk Jan 5
        "https://x/2": ["Crypto"],              # wk Jan 5
        "https://x/3": ["Payments", "Crypto"],  # wk Jan 12, both
        # x/4 absent -> not fintech -> excluded
    }
    trend_repo = _FakeTrendRepo()
    written = _service(articles, themes, trend_repo).build()

    assert written == 4
    got = {(m.week_start, m.theme): m.article_count for m in trend_repo.upserted}
    assert got == {
        (date(2026, 1, 5), "Payments"): 1,
        (date(2026, 1, 5), "Crypto"): 1,
        (date(2026, 1, 12), "Payments"): 1,
        (date(2026, 1, 12), "Crypto"): 1,
    }


def test_same_week_same_theme_accumulates():
    articles = [_raw("https://x/1", 5), _raw("https://x/2", 7)]
    themes = {"https://x/1": ["Payments"], "https://x/2": ["Payments"]}
    trend_repo = _FakeTrendRepo()
    _service(articles, themes, trend_repo).build()

    got = {(m.week_start, m.theme): m.article_count for m in trend_repo.upserted}
    assert got == {(date(2026, 1, 5), "Payments"): 2}


def test_fintech_article_with_no_theme_is_recorded_as_untagged():
    articles = [_raw("https://x/1", 7)]
    themes = {"https://x/1": []}  # accepted but matched no theme
    trend_repo = _FakeTrendRepo()
    untagged_repo = _FakeUntaggedRepo()
    written = _service(articles, themes, trend_repo, untagged_repo).build()

    assert written == 0
    assert trend_repo.upserted == []
    assert [a.url for a in untagged_repo.saved] == ["https://x/1"]
