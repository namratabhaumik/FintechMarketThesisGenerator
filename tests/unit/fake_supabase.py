"""A shared in-memory stand-in for the supabase-py query builder.

Every repository test used to carry its own fake. Once the repositories moved to
paged reads that stopped being viable: each fake would have to re-implement the
one behaviour the paging exists to survive - PostgREST silently truncating a read
at its max-rows cap - and a fake that returns the whole table in a single page
cannot tell a correctly paged read from a truncated one.

So `max_rows` is modelled here on purpose: a select never returns more rows than
the cap, and it reports no error when it holds rows back, exactly like the
server. `requests` records every range asked for so a test can assert on paging
behaviour itself.
"""

from typing import Callable, Optional


class FakeResp:
    def __init__(self, data=None, count=None):
        self.data = data
        self.count = count


class FakeTable:
    """One query. Builder methods chain; `execute` resolves against the store."""

    def __init__(self, store: dict, key: Callable, max_rows: int, requests: list):
        self._store = store          # key -> row dict
        self._key = key              # row -> the store's unique key
        self._max_rows = max_rows
        self._requests = requests
        self._op = None
        self._payload = None
        self._count = None
        self._filters: list = []     # (column, op, value)
        self._neq = None
        self._order: list = []       # (column, desc), applied in call order
        self._limit = None
        self._range = None

    def upsert(self, rows, on_conflict=None, ignore_duplicates=False):
        if ignore_duplicates:
            # UNIQUE(key) + ignore_duplicates: only never-seen keys insert, and
            # only those come back in the response.
            new = [r for r in rows if self._key(r) not in self._store]
        else:
            # ON CONFLICT DO UPDATE: existing rows are overwritten.
            new = list(rows)
        for r in new:
            self._store[self._key(r)] = r
        self._op, self._payload = "upsert", new
        return self

    def delete(self):
        self._op = "delete"
        return self

    def select(self, *args, count=None):
        self._op = "select"
        self._count = count
        return self

    def eq(self, column, value):
        self._filters.append((column, "eq", value))
        return self

    def gte(self, column, value):
        self._filters.append((column, "gte", value))
        return self

    def neq(self, column, value):
        self._neq = (column, value)
        return self

    def order(self, column, desc=False):
        self._order.append((column, desc))
        return self

    def limit(self, n):
        self._limit = n
        return self

    def range(self, start, end):
        self._range = (start, end)
        return self

    def execute(self):
        if self._op == "upsert":
            return FakeResp(data=self._payload)
        if self._op == "delete":
            column, value = self._neq
            for k in [k for k, r in self._store.items() if r.get(column) != value]:
                del self._store[k]
            return FakeResp(data=[])
        if self._count is not None:
            return FakeResp(count=len(self._store))

        rows = list(self._store.values())
        for column, op, value in self._filters:
            if op == "eq":
                rows = [r for r in rows if r.get(column) == value]
            else:  # gte; ISO date strings compare lexically = chronologically
                rows = [r for r in rows if r[column] >= value]
        # Later .order() calls are the weaker sort keys, so apply in reverse.
        for column, desc in reversed(self._order):
            rows.sort(key=lambda r: r[column], reverse=desc)
        if self._limit is not None:
            rows = rows[: self._limit]
        if self._range is not None:
            start, end = self._range
            self._requests.append((start, end))
            rows = rows[start : end + 1]
        # The cap is the last word: however wide the ask, the server returns at
        # most max_rows and says nothing about what it withheld.
        return FakeResp(data=rows[: self._max_rows])


class FakeClient:
    """Routes `table(name)` to a per-table store.

    `keys` maps a table name to its unique key function, mirroring that table's
    real UNIQUE / ON CONFLICT target. Tables not listed are keyed by url, which
    is what every Silver/Bronze table uses.
    """

    def __init__(self, max_rows: int = 1000, keys: Optional[dict] = None):
        self.max_rows = max_rows
        self.requests: list = []
        self._keys = keys or {}
        self.stores: dict = {}

    def store_for(self, name: str) -> dict:
        return self.stores.setdefault(name, {})

    @property
    def store(self) -> dict:
        """The single table's store, for the repositories that only touch one."""
        if len(self.stores) != 1:
            raise AssertionError(f"expected exactly one table, got {list(self.stores)}")
        return next(iter(self.stores.values()))

    def table(self, name):
        key = self._keys.get(name, lambda r: r["url"])
        return FakeTable(self.store_for(name), key, self.max_rows, self.requests)
