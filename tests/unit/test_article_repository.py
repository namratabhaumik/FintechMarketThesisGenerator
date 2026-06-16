"""Unit tests for the Bronze SupabaseArticleRepository (URL dedup)."""

from datetime import datetime, timezone

from core.implementations.repositories.supabase_article_repository import (
    SupabaseArticleRepository,
)
from core.models.raw_article import RawArticle

PUB = datetime(2026, 1, 1, tzinfo=timezone.utc)


class _FakeResp:
    def __init__(self, data=None, count=None):
        self.data = data
        self.count = count


class _FakeTable:
    """Mimics supabase-py's table builder for upsert/select used by the repo."""

    def __init__(self, store: dict):
        self._store = store
        self._op = None
        self._payload = None
        self._count = None

    def upsert(self, rows, on_conflict=None, ignore_duplicates=False):
        # Emulate UNIQUE(url) + ignore_duplicates: only never-seen URLs insert.
        new = [r for r in rows if r["url"] not in self._store]
        for r in new:
            self._store[r["url"]] = r
        self._op, self._payload = "upsert", new
        return self

    def select(self, *args, count=None):
        self._op = "select"
        self._count = count
        return self

    def order(self, column, desc=False):
        return self

    def execute(self):
        if self._op == "upsert":
            return _FakeResp(data=self._payload)
        if self._count is not None:
            return _FakeResp(count=len(self._store))
        return _FakeResp(data=list(self._store.values()))


class _FakeClient:
    def __init__(self):
        self.store: dict = {}

    def table(self, name):
        return _FakeTable(self.store)


def _raw(url, title="Title"):
    return RawArticle(title=title, url=url, published_at=PUB, summary="s")


def test_save_inserts_and_counts():
    repo = SupabaseArticleRepository(_FakeClient())
    inserted = repo.save([_raw("https://x/1"), _raw("https://x/2")])
    assert inserted == 2
    assert repo.count() == 2


def test_save_dedupes_by_url():
    repo = SupabaseArticleRepository(_FakeClient())
    repo.save([_raw("https://x/1"), _raw("https://x/2")])

    # Re-saving an existing URL inserts nothing new.
    inserted = repo.save([_raw("https://x/1"), _raw("https://x/3")])
    assert inserted == 1
    assert repo.count() == 3


def test_save_empty_is_noop():
    repo = SupabaseArticleRepository(_FakeClient())
    assert repo.save([]) == 0
    assert repo.count() == 0


def test_fetch_all_round_trips_to_raw_articles():
    repo = SupabaseArticleRepository(_FakeClient())
    repo.save([_raw("https://x/1", title="A"), _raw("https://x/2", title="B")])

    out = repo.fetch_all()

    assert {a.url for a in out} == {"https://x/1", "https://x/2"}
    assert all(isinstance(a.published_at, datetime) for a in out)
    assert all(a.published_at == PUB for a in out)
