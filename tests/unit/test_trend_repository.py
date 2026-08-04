"""Unit tests for the Gold SupabaseTrendRepository (upsert overwrite)."""

from datetime import date

from core.implementations.repositories.supabase_trend_repository import (
    SupabaseTrendRepository,
)
from core.models.tag_base_rate import TagBaseRate
from core.models.trend_metric import TrendMetric
from tests.unit.fake_supabase import FakeClient


def _FakeClient():
    """Gold's two tables are the only ones not keyed by url, so they need their
    real ON CONFLICT targets spelled out for the fake to dedupe like Postgres."""
    return FakeClient(keys={
        "trend_metrics": lambda r: (r["week_start"], r["dimension"], r["category"]),
        "tag_base_rates": lambda r: (r["dimension"], r["category"]),
    })


def _m(week, dimension, category, count, load_ids=None):
    return TrendMetric(
        week_start=date.fromisoformat(week),
        dimension=dimension,
        category=category,
        article_count=count,
        load_ids=load_ids or [],
    )


def test_upsert_and_fetch_all_round_trips_load_ids():
    """Lineage: the contributing Bronze loads are written and read back."""
    repo = SupabaseTrendRepository(_FakeClient())
    repo.upsert([_m("2026-01-05", "theme", "Payments", 3, load_ids=["load-A", "load-B"])])
    assert repo.fetch_all()[0].load_ids == ["load-A", "load-B"]


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


def test_upsert_stamps_computed_at_and_refreshes_it():
    """Every bucket in one recompute shares a computed_at (lineage: which run
    produced the count), and a later recompute overwrites it with a fresh one."""
    client = _FakeClient()
    repo = SupabaseTrendRepository(client)

    repo.upsert([
        _m("2026-01-05", "theme", "Payments", 2),
        _m("2026-01-05", "risk", "Regulatory Risk", 1),
    ])
    stamps = {r["computed_at"] for r in client.store.values()}
    assert len(stamps) == 1  # one timestamp across the whole recompute
    first = stamps.pop()
    assert first is not None

    repo.upsert([_m("2026-01-05", "theme", "Payments", 5)])
    refreshed = client.store[("2026-01-05", "theme", "Payments")]["computed_at"]
    assert refreshed >= first  # recompute refreshed the provenance stamp


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


def _rate(dimension, category, count, total=10):
    return TagBaseRate(
        dimension=dimension, category=category,
        article_count=count, total_articles=total,
    )


def test_base_rates_round_trip():
    repo = SupabaseTrendRepository(_FakeClient())
    written = repo.upsert_base_rates([
        _rate("theme", "Payments", 4),
        _rate("risk", "Regulatory Risk", 2),
    ])

    assert written == 2
    stored = {(r.dimension, r.category): r for r in repo.fetch_base_rates()}
    assert stored[("theme", "Payments")].article_count == 4
    assert stored[("theme", "Payments")].rate == 0.4


def test_recompute_drops_categories_that_left_the_corpus():
    """A stale row would stay a live lift denominator, so anything the current
    recompute did not stamp is deleted rather than left behind."""
    repo = SupabaseTrendRepository(_FakeClient())
    repo.upsert_base_rates([_rate("theme", "Payments", 4), _rate("theme", "Gone", 1)])

    repo.upsert_base_rates([_rate("theme", "Payments", 6, total=12)])

    stored = repo.fetch_base_rates()
    assert [(r.category, r.article_count, r.total_articles) for r in stored] == [
        ("Payments", 6, 12)
    ]


def test_empty_base_rates_write_nothing_and_leave_existing_rows():
    # Guards the delete: an empty recompute must not wipe the table, or a failed
    # Gold run would silently disable lift ranking.
    repo = SupabaseTrendRepository(_FakeClient())
    repo.upsert_base_rates([_rate("theme", "Payments", 4)])

    assert repo.upsert_base_rates([]) == 0
    assert [r.category for r in repo.fetch_base_rates()] == ["Payments"]


def test_fetch_base_rates_empty_when_gold_has_not_run():
    assert SupabaseTrendRepository(_FakeClient()).fetch_base_rates() == []
