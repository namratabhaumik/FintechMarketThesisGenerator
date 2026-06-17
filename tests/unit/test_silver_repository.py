"""Unit tests for the Silver SupabaseSilverRepository (verdict dedup)."""

from core.implementations.repositories.supabase_silver_repository import (
    SupabaseSilverRepository,
)
from core.models.silver_record import SilverVerdict


class _FakeResp:
    def __init__(self, data=None):
        self.data = data


class _FakeTable:
    def __init__(self, store: dict):
        self._store = store  # url -> full row dict
        self._op = None
        self._payload = None
        self._filter = None

    def upsert(self, rows, on_conflict=None, ignore_duplicates=False):
        new = [r for r in rows if r["url"] not in self._store]
        for r in new:
            self._store[r["url"]] = r
        self._op, self._payload = "upsert", new
        return self

    def select(self, *args):
        self._op = "select"
        return self

    def eq(self, column, value):
        self._filter = (column, value)
        return self

    def execute(self):
        if self._op == "upsert":
            return _FakeResp(data=self._payload)
        rows = list(self._store.values())
        if self._filter:
            col, val = self._filter
            rows = [r for r in rows if r.get(col) == val]
        return _FakeResp(data=rows)


class _FakeClient:
    def __init__(self):
        self.store: dict = {}

    def table(self, name):
        return _FakeTable(self.store)


def test_record_and_processed_urls():
    repo = SupabaseSilverRepository(_FakeClient())
    recorded = repo.record(
        [
            SilverVerdict(url="https://x/1", fintech_relevant=True),
            SilverVerdict(url="https://x/2", fintech_relevant=False),
        ]
    )
    assert recorded == 2
    assert repo.processed_urls() == {"https://x/1", "https://x/2"}


def test_fintech_themes_returns_themes_for_relevant_only():
    repo = SupabaseSilverRepository(_FakeClient())
    repo.record(
        [
            SilverVerdict(url="https://x/1", fintech_relevant=True, themes=["Payments"]),
            SilverVerdict(url="https://x/2", fintech_relevant=False),
            SilverVerdict(url="https://x/3", fintech_relevant=True, themes=["Crypto", "Payments"]),
        ]
    )
    assert repo.fintech_themes() == {
        "https://x/1": ["Payments"],
        "https://x/3": ["Crypto", "Payments"],
    }
    assert repo.processed_urls() == {"https://x/1", "https://x/2", "https://x/3"}


def test_record_dedupes_by_url():
    repo = SupabaseSilverRepository(_FakeClient())
    repo.record([SilverVerdict(url="https://x/1", fintech_relevant=True)])

    again = repo.record(
        [
            SilverVerdict(url="https://x/1", fintech_relevant=True),
            SilverVerdict(url="https://x/2", fintech_relevant=False),
        ]
    )
    assert again == 1
    assert repo.processed_urls() == {"https://x/1", "https://x/2"}


def test_record_empty_is_noop():
    repo = SupabaseSilverRepository(_FakeClient())
    assert repo.record([]) == 0
    assert repo.processed_urls() == set()
