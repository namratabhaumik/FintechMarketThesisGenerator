"""Tests for the API security layer: per-user JWT auth wiring + rate limiting."""

from unittest.mock import MagicMock

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

    def test_public_meta_endpoints_stay_open(self):
        client = _client()
        assert client.get("/api/feedback-options").status_code == 200
        assert client.get("/api/health").status_code == 200


class TestRateLimitKey:
    """Behind a reverse proxy (e.g. Render) the key is the client IP recorded
    in X-Forwarded-For, so users get separate buckets; without the header
    (local dev, direct access) it falls back to the peer address."""

    @staticmethod
    def _request(headers=(), client=("10.0.0.1", 1234)) -> Request:
        scope = {
            "type": "http", "method": "GET", "path": "/", "query_string": b"",
            "headers": list(headers), "client": client,
            "server": ("test", 80), "scheme": "http",
        }
        return Request(scope)

    def test_prefers_first_forwarded_hop(self):
        req = self._request([(b"x-forwarded-for", b"203.0.113.7, 10.1.2.3")])
        assert _rate_limit_key(req) == "203.0.113.7"

    def test_falls_back_to_peer_address_without_header(self):
        assert _rate_limit_key(self._request()) == "10.0.0.1"


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
