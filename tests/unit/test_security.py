"""Tests for the API security layer: per-user JWT auth wiring + rate limiting."""

from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from slowapi.errors import RateLimitExceeded

from api.security import _rate_limit_key, limiter, rate_limit_handler


def _client() -> TestClient:
    """TestClient over the real router with auth NOT overridden, so the real
    get_current_user runs and a missing bearer token 401s. get_container is
    mocked because it resolves before the auth dependency (and would otherwise
    raise 'not initialized' before the 401 is reached)."""
    from api.deps import get_container
    from api.routes import router

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_container] = lambda: MagicMock()
    return TestClient(app)


class TestAuthWiredToRoutes:
    """Every job endpoint (reads included) now requires a Supabase JWT: a
    missing bearer token 401s in get_current_user before the handler runs.
    Public meta endpoints stay open."""

    def test_create_thesis_401_without_token(self):
        res = _client().post("/api/theses", json={"query": "digital lending"})
        assert res.status_code == 401
        assert res.json()["detail"]["code"] == "unauthorized"

    def test_refinement_401_without_token(self):
        res = _client().post(
            "/api/theses/x/refinements", json={"feedback": ["Too broad"]})
        assert res.status_code == 401

    def test_approval_401_without_token(self):
        assert _client().put("/api/theses/x/approval").status_code == 401

    def test_list_401_without_token(self):
        assert _client().get("/api/theses").status_code == 401

    def test_get_401_without_token(self):
        assert _client().get("/api/theses/x").status_code == 401

    def test_delete_401_without_token(self):
        assert _client().delete("/api/theses/x").status_code == 401

    def test_public_meta_endpoints_stay_open(self):
        client = _client()
        assert client.get("/api/feedback-options").status_code == 200
        assert client.get("/api/health").status_code == 200


class TestRequireAdmin:
    """The delete endpoint is admin-only: require_admin gates on the JWT's
    app_metadata.role claim before the handler (and RLS) ever run."""

    def _client_as(self, role: str) -> TestClient:
        from api.auth import AuthUser, get_current_user, get_user_job_manager
        from api.deps import get_container
        from api.routes import router

        mock_jm = MagicMock()
        mock_jm.get_job = AsyncMock(return_value=MagicMock(id="x"))
        mock_jm.delete_job = AsyncMock()

        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_current_user] = lambda: AuthUser(
            id="u1", token="t", role=role
        )
        app.dependency_overrides[get_user_job_manager] = lambda: mock_jm
        app.dependency_overrides[get_container] = lambda: MagicMock()
        client = TestClient(app)
        client._mock_jm = mock_jm  # type: ignore[attr-defined]
        return client

    def test_delete_403_for_non_admin(self):
        res = self._client_as("user").delete("/api/theses/x")
        assert res.status_code == 403
        assert res.json()["detail"]["code"] == "forbidden"

    def test_delete_204_for_admin(self):
        client = self._client_as("admin")
        res = client.delete("/api/theses/x")
        assert res.status_code == 204
        client._mock_jm.delete_job.assert_awaited_once_with("x")  # type: ignore[attr-defined]


class TestRateLimitKey:
    """Behind a reverse proxy (e.g. Render) the key is the client IP recorded
    in X-Forwarded-For, so users get separate buckets; without the header
    (local dev, direct access) it falls back to the peer address.

    The entry is read from the RIGHT (the hop we trust wrote it), because
    anything further left may have been supplied by the caller."""

    @staticmethod
    def _request(headers=(), client=("10.0.0.1", 1234)) -> Request:
        scope = {
            "type": "http", "method": "GET", "path": "/", "query_string": b"",
            "headers": list(headers), "client": client,
            "server": ("test", 80), "scheme": "http",
        }
        return Request(scope)

    def test_uses_hop_written_by_the_trusted_proxy(self):
        """Render appended 10.1.2.3; the leftmost entry is the caller's own
        claim and must not become the key."""
        req = self._request([(b"x-forwarded-for", b"203.0.113.7, 10.1.2.3")])
        assert _rate_limit_key(req) == "10.1.2.3"

    def test_single_hop_when_proxy_overwrites_the_header(self):
        req = self._request([(b"x-forwarded-for", b"203.0.113.7")])
        assert _rate_limit_key(req) == "203.0.113.7"

    def test_spoofed_prefix_cannot_split_the_bucket(self):
        """Two requests from one client that vary only in the fabricated
        left-hand entries still share a bucket."""
        a = self._request([(b"x-forwarded-for", b"1.1.1.1, 203.0.113.9")])
        b = self._request([(b"x-forwarded-for", b"2.2.2.2, 203.0.113.9")])
        assert _rate_limit_key(a) == _rate_limit_key(b) == "203.0.113.9"

    def test_falls_back_to_peer_address_without_header(self):
        assert _rate_limit_key(self._request()) == "10.0.0.1"

    def test_falls_back_to_peer_address_on_empty_header(self):
        assert _rate_limit_key(self._request([(b"x-forwarded-for", b"  ")])) == "10.0.0.1"


class TestRateLimit:
    """The configured limiter enforces a limit and renders 429 in the shared
    {detail: {code, message}} envelope so the frontend parses it like any error."""

    def test_limit_exceeded_returns_429_in_error_envelope(self):
        app = FastAPI()
        app.state.limiter = limiter
        app.add_exception_handler(RateLimitExceeded, rate_limit_handler)

        @app.get("/probe")
        @limiter.limit("1/minute")
        def probe(request: Request):
            return {"ok": True}

        limiter.reset()  # isolate from any counts accrued elsewhere
        client = TestClient(app)
        assert client.get("/probe").status_code == 200
        blocked = client.get("/probe")
        assert blocked.status_code == 429
        assert blocked.json()["detail"]["code"] == "rate_limited"
        limiter.reset()
