"""Paged reads for the Supabase repositories.

PostgREST caps any read at its server-side max-rows setting and returns the
short page WITHOUT an error or any marker that more rows exist. Every repository
here used to issue a single unfiltered select and treat the result as the whole
table, so each was one row of corpus growth away from silently losing data.
"""

import logging
from typing import Any, Callable, List

logger = logging.getLogger(__name__)

# Rows requested per page. Only ever a request: the server may return fewer.
FETCH_PAGE = 1000


def fetch_paged(
    build_query: Callable[[], Any], page_size: int = FETCH_PAGE
) -> List[dict]:
    """Read every row a query matches, one page at a time.

    `build_query` must return a FRESH PostgREST builder each call (select,
    filters and ordering applied, no range). A builder accumulates state and is
    spent once executed, so the pages cannot share one.

    The query MUST carry a deterministic total order - sort on a unique column,
    or append one as a tiebreaker. Paging over a non-unique sort key lets tied
    rows reorder between requests, which drops some rows and repeats others.
    Postgres gives no stable order otherwise, so this cannot be checked here; it
    is the caller's job.

    Advances by the rows actually RETURNED and stops only on an empty page,
    rather than treating a short page as the end. A server whose max-rows is
    below `page_size` returns a short page that is NOT the end of the table, so
    stopping there would reintroduce the silent truncation this exists to fix.

    Assumes no concurrent DELETE on a paged table: offset paging shifts rows, and
    only a delete can shift one out of an unread page --> silently skipped. The one
    delete (upsert_base_rates) fits its table in a single request, so it cannot.
    Concurrent inserts only re-serve a row --> harmless. Breaking either --> keyset.
    """
    rows: List[dict] = []
    offset = 0
    while True:
        resp = build_query().range(offset, offset + page_size - 1).execute()
        page = resp.data or []
        if not page:
            return rows
        rows.extend(page)
        offset += len(page)