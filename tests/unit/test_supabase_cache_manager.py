"""Unit tests for SupabaseCacheManager (fake Supabase client, no network)."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from core.implementations.llm.supabase_cache_manager import SupabaseCacheManager
from core.models.cache_entry import FALLBACK_MODEL, FALLBACK_TTL_SECONDS


class FakeTable:
    """Minimal stand-in for the Supabase query builder over a dict store.

    Records the operation and its terminal filter, then resolves it against the
    shared dict on execute() - enough to exercise get/set/clear/metrics and the
    chained calls the manager makes.
    """

    def __init__(self, store):
        self._store = store
        self._op = None
        self._key = None
        self._row = None
        self._count = False

    def select(self, *cols, count=None):
        self._op = "select"
        self._count = count is not None
        return self

    def eq(self, col, val):
        self._key = val
        return self

    def neq(self, col, val):  # used by clear(); matches every row
        return self

    def limit(self, n):
        return self

    def upsert(self, row):
        self._op = "upsert"
        self._row = row
        return self

    def delete(self):
        self._op = "delete"
        return self

    def execute(self):
        if self._op == "select":
            if self._key is not None:
                data = [self._store[self._key]] if self._key in self._store else []
            else:
                data = list(self._store.values())
            return SimpleNamespace(
                data=data, count=len(self._store) if self._count else None
            )
        if self._op == "upsert":
            self._store[self._row["key"]] = self._row
            return SimpleNamespace(data=[self._row], count=None)
        if self._op == "delete":
            if self._key is not None:
                self._store.pop(self._key, None)
            else:
                self._store.clear()
            return SimpleNamespace(data=[], count=None)
        return SimpleNamespace(data=[], count=None)


class FakeClient:
    def __init__(self):
        self.store = {}

    def table(self, name):
        return FakeTable(self.store)


class BoomClient:
    """Every access raises, to exercise the best-effort error paths."""

    def table(self, name):
        raise RuntimeError("supabase down")


class TestSupabaseCacheManager:
    def test_version_busts_key(self):
        c = FakeClient()
        k1 = SupabaseCacheManager(c, version="v1").generate_key("d", "t", "m")
        k2 = SupabaseCacheManager(c, version="v2").generate_key("d", "t", "m")
        assert k1 != k2

    def test_miss_on_empty(self):
        m = SupabaseCacheManager(FakeClient(), version="v")
        assert m.get(m.generate_key("d", "t", "combined")) is None

    def test_set_get_roundtrip(self):
        m = SupabaseCacheManager(FakeClient(), version="v")
        key = m.generate_key("docs", "fintech", "combined")
        m.set(key, "a summary", "gemini-2.5-flash", 10, 5)
        entry = m.get(key)
        assert entry is not None
        assert entry.response == "a summary"
        assert entry.model == "gemini-2.5-flash"

    def test_expired_entry_is_evicted_and_missed(self):
        client = FakeClient()
        m = SupabaseCacheManager(client, ttl_seconds=3600, version="v")
        key = m.generate_key("docs", "fintech", "combined")
        m.set(key, "stale", "gemini-2.5-flash", 1, 1)
        # Age the stored row past the TTL.
        client.store[key]["created_at"] = (
            datetime.now(timezone.utc) - timedelta(hours=2)
        ).isoformat()
        assert m.get(key) is None
        assert key not in client.store  # lazily evicted

    def test_read_failure_returns_none(self):
        m = SupabaseCacheManager(BoomClient(), version="v")
        assert m.get("any-key") is None

    def test_write_failure_is_swallowed(self):
        m = SupabaseCacheManager(BoomClient(), version="v")
        # Must not raise into the request path.
        m.set("k", "r", "model", 1, 1)

    def test_clear_removes_all(self):
        client = FakeClient()
        m = SupabaseCacheManager(client, version="v")
        m.set(m.generate_key("a", "t", "combined"), "r1", "model", 1, 1)
        m.set(m.generate_key("b", "t", "combined"), "r2", "model", 1, 1)
        m.clear()
        assert client.store == {}

    def test_metrics_track_hits_and_misses(self):
        m = SupabaseCacheManager(FakeClient(), version="v")
        key = m.generate_key("docs", "fintech", "combined")
        m.set(key, "r", "model", 1, 1)
        m.get(key)            # hit
        m.get("missing-key")  # miss
        metrics = m.get_metrics()
        assert metrics["hits"] == 1
        assert metrics["misses"] == 1
        assert metrics["cache_size"] == 1

    def _age(self, client, key, seconds):
        """Backdate the stored row's created_at by `seconds`."""
        client.store[key]["created_at"] = (
            datetime.now(timezone.utc) - timedelta(seconds=seconds)
        ).isoformat()

    def test_fallback_entry_expires_on_short_ttl(self):
        # A degraded local-fallback summary is aged out on FALLBACK_TTL_SECONDS
        # even though the manager's normal TTL is a week away.
        client = FakeClient()
        m = SupabaseCacheManager(client, ttl_seconds=604800, version="v")
        key = m.generate_key("docs", "fintech", "combined")
        m.set(key, "fallback summary", FALLBACK_MODEL, 0, 0)
        self._age(client, key, FALLBACK_TTL_SECONDS + 60)
        assert m.get(key) is None
        assert key not in client.store  # lazily evicted, so the LLM is retried

    def test_fallback_entry_hits_within_short_ttl(self):
        # A rapid identical repeat during an outage is still served from cache.
        client = FakeClient()
        m = SupabaseCacheManager(client, ttl_seconds=604800, version="v")
        key = m.generate_key("docs", "fintech", "combined")
        m.set(key, "fallback summary", FALLBACK_MODEL, 0, 0)
        self._age(client, key, FALLBACK_TTL_SECONDS - 60)
        entry = m.get(key)
        assert entry is not None
        assert entry.response == "fallback summary"

    def test_normal_entry_survives_short_ttl(self):
        # A normal LLM summary at the same age is unaffected by the short TTL -
        # the shortened window applies only to the fallback marker.
        client = FakeClient()
        m = SupabaseCacheManager(client, ttl_seconds=604800, version="v")
        key = m.generate_key("docs", "fintech", "combined")
        m.set(key, "real summary", "gemini-2.5-flash", 10, 5)
        self._age(client, key, FALLBACK_TTL_SECONDS + 60)
        assert m.get(key) is not None
