"""Unit tests for SupabaseUntaggedRepository"""

from datetime import datetime, timezone

from core.implementations.repositories.supabase_untagged_repository import (
    SupabaseUntaggedRepository,
)
from core.models.raw_article import RawArticle

PUB = datetime(2026, 1, 1, tzinfo=timezone.utc)


class _FakeResp:
    def __init__(self, data=None):
        self.data = data


class _FakeTable:
    def __init__(self, store: dict):
        self._store = store
        self._payload = None

    def upsert(self, rows, on_conflict=None, ignore_duplicates=False):
        new = [r for r in rows if r["url"] not in self._store]
        for r in new:
            self._store[r["url"]] = r
        self._payload = new
        return self

    def execute(self):
        return _FakeResp(data=self._payload)


class _FakeClient:
    def __init__(self):
        self.store: dict = {}

    def table(self, name):
        return _FakeTable(self.store)


def _raw(url):
    return RawArticle(title="T", url=url, published_at=PUB, summary="s", source="x.com")


def test_save_records_and_dedupes():
    repo = SupabaseUntaggedRepository(_FakeClient())
    assert repo.save([_raw("https://x/1"), _raw("https://x/2")]) == 2

    # Re-saving an existing URL records nothing new.
    assert repo.save([_raw("https://x/1"), _raw("https://x/3")]) == 1


def test_save_empty_is_noop():
    repo = SupabaseUntaggedRepository(_FakeClient())
    assert repo.save([]) == 0