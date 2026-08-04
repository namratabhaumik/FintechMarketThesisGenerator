"""Unit tests for fetch_paged, the shared Supabase paged read.

The bug this guards against is silent: PostgREST caps a read at its max-rows
setting and returns the short page with no error, so an unpaged repository read
looks successful while hiding rows. In production the Bronze read had already
crossed the cap - 1000 articles returned against 1013 already-processed URLs.
"""

import pytest

from core.implementations.repositories.paging import fetch_paged
from tests.unit.fake_supabase import FakeClient

TABLE = "articles_raw"


def _fill(client, n):
    store = client.store_for(TABLE)
    for i in range(n):
        store[f"https://x/{i:04d}"] = {"url": f"https://x/{i:04d}", "n": i}
    return client


def _query(client):
    return lambda: client.table(TABLE).select("*").order("url")


def test_reads_every_row_past_the_max_rows_cap():
    client = _fill(FakeClient(max_rows=10), 25)

    rows = fetch_paged(_query(client), page_size=10)

    # An unpaged read returns 10 here and reports nothing wrong.
    assert len(rows) == 25
    assert len({r["url"] for r in rows}) == 25


def test_short_page_is_not_mistaken_for_the_end():
    """Server max-rows below our page size makes EVERY page short. Treating a
    short page as the end would silently truncate at the first request."""
    client = _fill(FakeClient(max_rows=4), 25)

    assert len(fetch_paged(_query(client), page_size=10)) == 25


def test_exact_multiple_of_page_size_terminates():
    # Row count divides evenly, so the last full page is followed by an empty
    # one. Stopping needs that empty page, not arithmetic.
    client = _fill(FakeClient(max_rows=10), 20)

    assert len(fetch_paged(_query(client), page_size=10)) == 20


def test_empty_table_makes_one_request_and_stops():
    client = FakeClient(max_rows=10)

    assert fetch_paged(_query(client), page_size=10) == []
    assert client.requests == [(0, 9)]


def test_advances_by_rows_returned_not_by_page_size():
    """With max_rows below page_size the offset must follow what came back;
    stepping by page_size would skip the rows the server held back."""
    client = _fill(FakeClient(max_rows=4), 10)

    rows = fetch_paged(_query(client), page_size=10)

    assert [r["n"] for r in rows] == list(range(10))
    # Each range starts where the previous page actually ended: the cap yields
    # 4, 4, then only 2 rows, so the final probe is at 10 - not at 12, which is
    # where stepping blindly by page_size would have landed (skipping rows 8-9).
    assert client.requests == [(0, 9), (4, 13), (8, 17), (10, 19)]


def test_preserves_query_order_across_pages():
    client = _fill(FakeClient(max_rows=3), 12)

    rows = fetch_paged(
        lambda: client.table(TABLE).select("*").order("n", desc=True), page_size=3
    )

    assert [r["n"] for r in rows] == list(range(11, -1, -1))


def test_builds_a_fresh_query_per_page():
    """A PostgREST builder accumulates state and is spent once executed, so the
    helper must call the factory again for each page rather than reusing one."""
    client = _fill(FakeClient(max_rows=5), 12)
    built = []

    def build():
        built.append(1)
        return client.table(TABLE).select("*").order("url")

    fetch_paged(build, page_size=5)

    assert len(built) == len(client.requests) > 1


def test_filters_are_reapplied_on_every_page():
    client = FakeClient(max_rows=3)
    store = client.store_for(TABLE)
    for i in range(20):
        store[f"u{i:02d}"] = {"url": f"u{i:02d}", "keep": i % 2 == 0}

    rows = fetch_paged(
        lambda: client.table(TABLE).select("*").eq("keep", True).order("url"),
        page_size=3,
    )

    # Rebuilding the query per page must not drop the filter partway through.
    assert len(rows) == 10
    assert all(r["keep"] for r in rows)


@pytest.mark.parametrize("total", [0, 1, 9, 10, 11])
def test_row_count_is_exact_around_the_page_boundary(total):
    client = _fill(FakeClient(max_rows=10), total)

    assert len(fetch_paged(_query(client), page_size=10)) == total
