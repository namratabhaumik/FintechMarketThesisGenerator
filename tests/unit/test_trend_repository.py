"""Unit tests for the Gold SupabaseTrendRepository (upsert overwrite)."""

from datetime import date

from core.implementations.repositories.supabase_trend_repository import (
    SupabaseTrendRepository,
)
from core.models.trend_metric import TrendMetric


class _FakeResp:
    def __init__(self, data=None):
        self.data = data


class _FakeTable:
    def __init__(self, store: dict):
        self._store = store  # (week_start, theme) -> row
        self._op = None

    def upsert(self, rows, on_conflict=None):
        # Composite key overwrite, mirroring ON CONFLICT DO UPDATE.
        for r in rows:
            self._store[(r["week_start"], r["theme"])] = r
        self._op = "upsert"
        return self

    def select(self, *args):
        self._op = "select"
        return self

    def order(self, column, desc=False):
        return self

    def execute(self):
        return _FakeResp(data=list(self._store.values()))


class _FakeClient:
    def __init__(self):
        self.store: dict = {}

    def table(self, name):
        return _FakeTable(self.store)


def _m(week, theme, count):
    return TrendMetric(week_start=date.fromisoformat(week), theme=theme, article_count=count)


def test_upsert_and_fetch_all():
    repo = SupabaseTrendRepository(_FakeClient())
    written = repo.upsert([_m("2026-01-05", "Payments", 2), _m("2026-01-05", "Crypto", 1)])

    assert written == 2
    got = {(m.week_start, m.theme): m.article_count for m in repo.fetch_all()}
    assert got == {
        (date(2026, 1, 5), "Payments"): 2,
        (date(2026, 1, 5), "Crypto"): 1,
    }


def test_upsert_overwrites_existing_count():
    repo = SupabaseTrendRepository(_FakeClient())
    repo.upsert([_m("2026-01-05", "Payments", 2)])
    repo.upsert([_m("2026-01-05", "Payments", 5)])

    metrics = repo.fetch_all()
    assert len(metrics) == 1
    assert metrics[0].article_count == 5


def test_upsert_empty_is_noop():
    repo = SupabaseTrendRepository(_FakeClient())
    assert repo.upsert([]) == 0
    assert repo.fetch_all() == []
