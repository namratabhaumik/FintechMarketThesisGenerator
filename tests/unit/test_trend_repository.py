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
        self._store = store  # (week_start, dimension, category) -> row
        self._op = None
        self._filters = []  # (column, min_value) for .gte
        self._order = None  # (column, desc)
        self._limit = None

    def upsert(self, rows, on_conflict=None):
        # Composite key overwrite, mirroring ON CONFLICT DO UPDATE.
        for r in rows:
            self._store[(r["week_start"], r["dimension"], r["category"])] = r
        self._op = "upsert"
        return self

    def select(self, *args):
        self._op = "select"
        return self

    def gte(self, column, value):
        self._filters.append((column, value))
        return self

    def order(self, column, desc=False):
        self._order = (column, desc)
        return self

    def limit(self, n):
        self._limit = n
        return self

    def execute(self):
        # week_start rows are ISO strings, so >= and sort compare lexically,
        # which for ISO-8601 dates matches chronological order.
        rows = list(self._store.values())
        for column, value in self._filters:
            rows = [r for r in rows if r[column] >= value]
        if self._order:
            column, desc = self._order
            rows.sort(key=lambda r: r[column], reverse=desc)
        if self._limit is not None:
            rows = rows[: self._limit]
        return _FakeResp(data=rows)


class _FakeClient:
    def __init__(self):
        self.store: dict = {}

    def table(self, name):
        return _FakeTable(self.store)


def _m(week, dimension, category, count):
    return TrendMetric(
        week_start=date.fromisoformat(week),
        dimension=dimension,
        category=category,
        article_count=count,
    )


def test_upsert_and_fetch_all():
    repo = SupabaseTrendRepository(_FakeClient())
    written = repo.upsert([
        _m("2026-01-05", "theme", "Payments", 2),
        _m("2026-01-05", "risk", "Regulatory Risk", 1),
    ])

    assert written == 2
    got = {(m.week_start, m.dimension, m.category): m.article_count for m in repo.fetch_all()}
    assert got == {
        (date(2026, 1, 5), "theme", "Payments"): 2,
        (date(2026, 1, 5), "risk", "Regulatory Risk"): 1,
    }


def test_upsert_overwrites_existing_count():
    repo = SupabaseTrendRepository(_FakeClient())
    repo.upsert([_m("2026-01-05", "theme", "Payments", 2)])
    repo.upsert([_m("2026-01-05", "theme", "Payments", 5)])

    metrics = repo.fetch_all()
    assert len(metrics) == 1
    assert metrics[0].article_count == 5


def test_upsert_empty_is_noop():
    repo = SupabaseTrendRepository(_FakeClient())
    assert repo.upsert([]) == 0
    assert repo.fetch_all() == []


# Four consecutive Mondays ending at as_of, plus one well outside a 4-week window.
_W0, _W1, _W2, _W3 = "2026-06-15", "2026-06-08", "2026-06-01", "2026-05-25"
_OLD = "2026-04-06"


def _seeded_repo():
    repo = SupabaseTrendRepository(_FakeClient())
    repo.upsert([
        _m(_W0, "theme", "Payments", 5),
        _m(_W2, "theme", "Payments", 3),
        _m(_W3, "signal", "Infra", 2),
        _m(_OLD, "theme", "Payments", 4),  # outside a 4-week window from as_of
    ])
    return repo


def test_fetch_recent_none_returns_everything():
    # Whole-corpus retrieval (window_weeks None) must read all of Gold.
    repo = _seeded_repo()
    assert {m.week_start for m in repo.fetch_recent(None)} == {
        m.week_start for m in repo.fetch_all()
    }


def test_fetch_recent_scopes_to_window_ending_at_as_of():
    # window_weeks=4 -> the 4 Mondays [as_of-3wk, as_of]; the OLD week is dropped,
    # and as_of (W0) is retained so the window still anchors to the latest week.
    repo = _seeded_repo()
    weeks = {m.week_start.isoformat() for m in repo.fetch_recent(4)}
    assert weeks == {_W0, _W2, _W3}  # W1 has no row; OLD excluded
    assert date.fromisoformat(_OLD) not in {m.week_start for m in repo.fetch_recent(4)}


def test_fetch_recent_window_one_keeps_only_as_of():
    repo = _seeded_repo()
    assert {m.week_start.isoformat() for m in repo.fetch_recent(1)} == {_W0}


def test_fetch_recent_empty_gold_returns_empty():
    repo = SupabaseTrendRepository(_FakeClient())
    assert repo.fetch_recent(52) == []
