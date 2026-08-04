"""Unit tests for the Bronze SupabaseArticleRepository (URL dedup)."""

from datetime import datetime, timezone

from core.implementations.repositories.supabase_article_repository import (
    SupabaseArticleRepository,
)
from core.models.raw_article import RawArticle
from tests.unit.fake_supabase import FakeClient as _FakeClient

PUB = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _raw(url, title="Title", published_at=PUB):
    return RawArticle(title=title, url=url, published_at=published_at, summary="s")


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


def test_save_stamps_one_load_id_per_run():
    """Every row from a single save() shares one load_id (the run that landed
    them); a second run gets a different id, so rows trace back to their load."""
    client = _FakeClient()
    repo = SupabaseArticleRepository(client)

    repo.save([_raw("https://x/1"), _raw("https://x/2")])
    repo.save([_raw("https://x/3")])

    rows = list(client.store.values())
    by_url = {r["url"]: r["load_id"] for r in rows}
    assert by_url["https://x/1"] == by_url["https://x/2"]  # same run -> same id
    assert by_url["https://x/3"] != by_url["https://x/1"]  # new run -> new id
    assert all(r["load_id"] is not None for r in rows)


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


class TestFetchAllPaging:
    """Bronze outgrew PostgREST's max-rows cap, which truncates a read with no
    error. Silver treats fetch_all as the whole of Bronze, so anything left
    behind is never classified and no gate can see the loss.

    Paging mechanics are covered once in test_paging.py; these two check the
    parts specific to this repository - that it pages at all, and that its sort
    is unique enough to page over.
    """

    def _fill(self, client, n, published_at=PUB):
        repo = SupabaseArticleRepository(client)
        repo.save([
            _raw(f"https://x/{i:03d}", published_at=published_at) for i in range(n)
        ])
        return repo

    def test_reads_past_the_max_rows_cap(self):
        client = _FakeClient(max_rows=10)
        repo = self._fill(client, 25)

        out = repo.fetch_all()

        # 25 rows behind a 10-row cap: the old unpaged read stopped at 10.
        assert len(out) == 25
        assert len({a.url for a in out}) == 25

    def test_tied_published_at_does_not_drop_or_repeat_rows(self):
        # Feed entries routinely share a timestamp, and published_at alone is a
        # non-unique sort: tied rows may reorder between requests, so a row can
        # be served twice while another is skipped. The url tiebreaker pins it.
        client = _FakeClient(max_rows=5)
        repo = self._fill(client, 20, published_at=PUB)

        urls = [a.url for a in repo.fetch_all()]

        assert len(urls) == len(set(urls)) == 20
