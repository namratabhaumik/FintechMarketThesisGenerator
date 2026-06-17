"""Unit tests for SupabaseQuarantineRepository (URL dedup)."""

from core.implementations.repositories.supabase_quarantine_repository import (
    SupabaseQuarantineRepository,
)
from core.models.quarantine_record import (
    INVALID_ARTICLE,
    SCRAPE_FAILED,
    QuarantineRecord,
)


class _FakeResp:
    def __init__(self, data=None):
        self.data = data


class _FakeTable:
    def __init__(self, store: dict):
        self._store = store
        self._op = None
        self._payload = None

    def upsert(self, rows, on_conflict=None, ignore_duplicates=False):
        new = [r for r in rows if r["url"] not in self._store]
        for r in new:
            self._store[r["url"]] = r
        self._op, self._payload = "upsert", new
        return self

    def select(self, *args):
        self._op = "select"
        return self

    def execute(self):
        if self._op == "upsert":
            return _FakeResp(data=self._payload)
        return _FakeResp(data=[{"url": u} for u in self._store])


class _FakeClient:
    def __init__(self):
        self.store: dict = {}

    def table(self, name):
        return _FakeTable(self.store)


def test_add_records_and_quarantined_urls():
    repo = SupabaseQuarantineRepository(_FakeClient())
    added = repo.add(
        [
            QuarantineRecord(url="https://x/1", reason=SCRAPE_FAILED),
            QuarantineRecord(url="https://x/2", reason=INVALID_ARTICLE, detail="empty source"),
        ]
    )
    assert added == 2
    assert repo.quarantined_urls() == {"https://x/1", "https://x/2"}


def test_add_dedupes_by_url():
    repo = SupabaseQuarantineRepository(_FakeClient())
    repo.add([QuarantineRecord(url="https://x/1", reason=SCRAPE_FAILED)])

    again = repo.add(
        [
            QuarantineRecord(url="https://x/1", reason=SCRAPE_FAILED),
            QuarantineRecord(url="https://x/2", reason=SCRAPE_FAILED),
        ]
    )
    assert again == 1
    assert repo.quarantined_urls() == {"https://x/1", "https://x/2"}


def test_add_empty_is_noop():
    repo = SupabaseQuarantineRepository(_FakeClient())
    assert repo.add([]) == 0
    assert repo.quarantined_urls() == set()
