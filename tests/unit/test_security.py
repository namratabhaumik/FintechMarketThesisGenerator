"""Tests for the API security layer: the shared-key gate and rate limiting."""

import pytest
from unittest.mock import MagicMock

from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient
from slowapi.errors import RateLimitExceeded

from api.security import (
    API_KEY_ENV,
    limiter,
    rate_limit_handler,
    require_api_key,
)


class TestApiKeyGate:
    """Unit tests for require_api_key across env states. The key is read at
    call time, so monkeypatching the env before each call exercises the gate."""

    def test_open_when_key_unset(self, monkeypatch):
        """Unset key -> gate is a no-op regardless of the header (local dev)."""
        monkeypatch.delenv(API_KEY_ENV, raising=False)
        assert require_api_key(x_api_key=None) is None
        assert require_api_key(x_api_key="anything") is None

    def test_missing_header_rejected_when_key_set(self, monkeypatch):
        monkeypatch.setenv(API_KEY_ENV, "s3cret")
        with pytest.raises(HTTPException) as exc:
            require_api_key(x_api_key=None)
        assert exc.value.status_code == 401
        assert exc.value.detail["code"] == "unauthorized"

    def test_wrong_key_rejected(self, monkeypatch):
        monkeypatch.setenv(API_KEY_ENV, "s3cret")
        with pytest.raises(HTTPException) as exc:
            require_api_key(x_api_key="wrong")
        assert exc.value.status_code == 401

    def test_correct_key_accepted(self, monkeypatch):
        monkeypatch.setenv(API_KEY_ENV, "s3cret")
        assert require_api_key(x_api_key="s3cret") is None


def _client(mock_jm=None):
    """A TestClient over the real router with the job manager/container mocked."""
    from api.deps import get_container, get_job_manager
    from api.routes import router

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_job_manager] = lambda: mock_jm or MagicMock()
    app.dependency_overrides[get_container] = lambda: MagicMock()
    return TestClient(app)


class TestGateWiredToRoutes:
    """The gate must be applied to every mutating endpoint and to none of the
    reads. A 401 fires in the dependency before the body, so no container setup
    is needed."""

    def test_create_thesis_401_without_key(self, monkeypatch):
        monkeypatch.setenv(API_KEY_ENV, "s3cret")
        res = _client().post("/api/theses", json={"query": "digital lending"})
        assert res.status_code == 401
        assert res.json()["detail"]["code"] == "unauthorized"

    def test_refinement_401_without_key(self, monkeypatch):
        monkeypatch.setenv(API_KEY_ENV, "s3cret")
        res = _client().post(
            "/api/theses/x/refinements", json={"feedback": ["Too broad"]})
        assert res.status_code == 401

    def test_approval_401_without_key(self, monkeypatch):
        monkeypatch.setenv(API_KEY_ENV, "s3cret")
        res = _client().put("/api/theses/x/approval")
        assert res.status_code == 401

    def test_valid_key_passes_the_gate(self, monkeypatch):
        """A correct key clears the gate (approval reaches the handler -> 404,
        not 401, since the job does not exist)."""
        monkeypatch.setenv(API_KEY_ENV, "s3cret")
        mock_jm = MagicMock()
        mock_jm.get_job.return_value = None
        res = _client(mock_jm).put(
            "/api/theses/x/approval", headers={"X-API-Key": "s3cret"})
        assert res.status_code == 404
        assert res.json()["detail"]["code"] == "job_not_found"

    def test_reads_stay_open_when_key_set(self, monkeypatch):
        """Reads are intentionally ungated (deferred RBAC)."""
        monkeypatch.setenv(API_KEY_ENV, "s3cret")
        mock_jm = MagicMock()
        mock_jm.list_jobs.return_value = []
        client = _client(mock_jm)
        assert client.get("/api/theses").status_code == 200
        assert client.get("/api/feedback-options").status_code == 200


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
