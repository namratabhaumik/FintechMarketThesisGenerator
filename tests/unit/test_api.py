"""Unit tests for the FastAPI layer with Supabase-backed job manager."""

import asyncio

import pytest
from unittest.mock import AsyncMock, MagicMock, Mock

from api.schemas import ThesisRequest, RefinementRequest


class TestSupabaseJobManager:
    """Tests for SupabaseJobManager with a mocked async Supabase client.

    Methods are async now, so the terminal .execute() is an AsyncMock and the
    coroutines are driven with asyncio.run (no pytest-asyncio dependency).
    """

    @pytest.fixture
    def mock_client(self):
        """Create a mock async Supabase client."""
        client = MagicMock()
        # Chain: await client.table(TABLE).insert(row).execute()
        client.table.return_value.insert.return_value.execute = AsyncMock(
            return_value=Mock(data=None)
        )
        return client

    @pytest.fixture
    def jm(self, mock_client):
        """Create a SupabaseJobManager with the mocked async client injected."""
        from api.supabase_job_manager import SupabaseJobManager
        return SupabaseJobManager(mock_client)

    def test_create_job(self, jm, mock_client):
        """Test creating a new job inserts a row and returns a proxy."""
        job = asyncio.run(jm.create_job("digital lending"))
        assert job.id is not None
        assert len(job.id) == 12
        assert job.query == "digital lending"
        mock_client.table.return_value.insert.assert_called_once()

    def test_create_job_with_fields_is_single_insert(self, jm, mock_client):
        """Extra fields are serialised into the SAME insert (atomic write) -
        no follow-up update that could strand a half-written row."""
        from core.models.thesis import StructuredThesis

        thesis = StructuredThesis(key_themes=["Fintech"], opportunity_score=3.5)
        asyncio.run(jm.create_job("digital lending", thesis=thesis))

        row = mock_client.table.return_value.insert.call_args.args[0]
        assert row["thesis"]["key_themes"] == ["Fintech"]  # serialised, not dataclass
        mock_client.table.return_value.update.assert_not_called()

    def test_get_job_found(self, jm, mock_client):
        """Test retrieving an existing job."""
        mock_client.table.return_value.select.return_value.eq.return_value.maybe_single.return_value.execute = AsyncMock(
            return_value=Mock(
                data={"id": "abc123", "query": "neobanking", "thesis": None}
            )
        )
        job = asyncio.run(jm.get_job("abc123"))
        assert job is not None
        assert job.id == "abc123"

    def test_get_job_not_found(self, jm, mock_client):
        """Test that missing job returns None."""
        mock_client.table.return_value.select.return_value.eq.return_value.maybe_single.return_value.execute = AsyncMock(
            return_value=Mock(data=None)
        )
        assert asyncio.run(jm.get_job("nonexistent")) is None

    def test_update_job_serialises_thesis(self, jm, mock_client):
        """Test that update_job serialises StructuredThesis to dict."""
        from core.models.thesis import StructuredThesis
        mock_client.table.return_value.update.return_value.eq.return_value.execute = AsyncMock(
            return_value=Mock(data=None)
        )
        thesis = StructuredThesis(key_themes=["AI"], risks=["Regulatory"])
        asyncio.run(jm.update_job("abc123", thesis=thesis))
        call_args = mock_client.table.return_value.update.call_args[0][0]
        assert call_args["thesis"]["key_themes"] == ["AI"]
        assert call_args["thesis"]["risks"] == ["Regulatory"]

    def test_update_job_guarded_true_when_row_matched(self, jm, mock_client):
        """Guards become WHERE filters (eq for values, is null for None) and a
        matched row returns True."""
        chain = mock_client.table.return_value.update.return_value
        chain.eq.return_value.eq.return_value.is_.return_value.execute = AsyncMock(
            return_value=Mock(data=[{"id": "abc123"}], count=1)
        )
        ok = asyncio.run(jm.update_job_guarded(
            "abc123", {"refinement_count": 2, "approved_at": None},
            refinement_count=3))
        assert ok is True
        # id filter first, then the value guard, then the IS NULL guard.
        chain.eq.assert_called_once_with("id", "abc123")
        chain.eq.return_value.eq.assert_called_once_with("refinement_count", 2)
        chain.eq.return_value.eq.return_value.is_.assert_called_once_with(
            "approved_at", "null")

    def test_update_job_guarded_false_when_race_lost(self, jm, mock_client):
        """Zero matched rows (another writer invalidated a guard) returns False."""
        chain = mock_client.table.return_value.update.return_value
        chain.eq.return_value.eq.return_value.is_.return_value.execute = AsyncMock(
            return_value=Mock(data=[], count=0)
        )
        ok = asyncio.run(jm.update_job_guarded(
            "abc123", {"refinement_count": 2, "approved_at": None},
            refinement_count=3))
        assert ok is False

    def test_list_jobs(self, jm, mock_client):
        """Test listing jobs returns proxies ordered by created_at desc."""
        mock_client.table.return_value.select.return_value.order.return_value.execute = AsyncMock(
            return_value=Mock(
                data=[
                    {"id": "job2", "query": "second", "status": "pending"},
                    {"id": "job1", "query": "first", "status": "completed"},
                ]
            )
        )
        jobs = asyncio.run(jm.list_jobs())
        assert len(jobs) == 2
        assert jobs[0].id == "job2"
        assert jobs[1].id == "job1"


class TestRowProxy:
    """Tests for _RowProxy rehydration of stored data."""

    def test_rehydrates_thesis(self):
        """Test that a stored thesis dict becomes a StructuredThesis."""
        from api.supabase_job_manager import _RowProxy
        row = {
            "id": "test1", "query": "q", "status": "completed",
            "thesis": {"key_themes": ["Fintech"], "risks": [], "investment_signals": [],
                       "sources": [], "raw_output": "text", "opportunity_score": 3.5,
                       "confidence_level": 0.8, "recommendation": "Pursue", "key_risk_factors": []},
        }
        proxy = _RowProxy(row)
        assert proxy.thesis is not None
        assert proxy.thesis.key_themes == ["Fintech"]
        assert proxy.thesis.opportunity_score == 3.5

    def test_null_thesis_stays_none(self):
        """Test that null thesis in DB stays None."""
        from api.supabase_job_manager import _RowProxy
        row = {"id": "test3", "query": "q", "status": "pending", "thesis": None}
        proxy = _RowProxy(row)
        assert proxy.thesis is None

    def test_thesis_with_unknown_stored_key_still_rehydrates(self):
        """A stored thesis carrying a key the dataclass no longer has (field
        renamed/removed in a later schema) degrades gracefully instead of
        raising TypeError on every read of the old row."""
        from api.supabase_job_manager import _RowProxy
        row = {
            "id": "test4", "query": "q", "status": "completed",
            "thesis": {"key_themes": ["Fintech"], "some_renamed_field": "old value"},
        }
        proxy = _RowProxy(row)
        assert proxy.thesis is not None
        assert proxy.thesis.key_themes == ["Fintech"]
        assert not hasattr(proxy.thesis, "some_renamed_field")


class TestSchemas:
    """Tests for Pydantic request/response schemas."""

    def test_thesis_request_valid(self):
        """Test valid thesis request."""
        req = ThesisRequest(query="digital lending in Asia")
        assert req.query == "digital lending in Asia"

    def test_thesis_request_empty_query_rejected(self):
        """Test that empty query is rejected."""
        with pytest.raises(Exception):
            ThesisRequest(query="")

    def test_refinement_request_valid(self):
        """Test valid refinement request."""
        req = RefinementRequest(feedback=["Too broad", "Missing trends"])
        assert len(req.feedback) == 2

    def test_refinement_status_coerces_and_rejects(self):
        """refinement_status is a str-enum: valid strings from storage coerce to
        the enum and serialise back to the wire string; unknown values reject."""
        from api.schemas import JobResponse, RefinementStatus
        job = JobResponse(job_id="j", query="q", refinement_status="refining")
        assert job.refinement_status == RefinementStatus.REFINING
        assert job.model_dump(mode="json")["refinement_status"] == "refining"
        with pytest.raises(Exception):
            JobResponse(job_id="j", query="q", refinement_status="bogus")


class TestSerializers:
    """Tests for serialise_job_fields enum unwrapping."""

    def test_refinement_status_enum_unwrapped_for_storage(self):
        """A RefinementStatus enum is stored as its string value."""
        from api.schemas import RefinementStatus
        from api.serializers import serialise_job_fields
        payload = serialise_job_fields(refinement_status=RefinementStatus.REFINED)
        assert payload["refinement_status"] == "refined"


def _row(job_id="test123", query="digital lending", **overrides):
    """A completed job row dict in the stored (Supabase) shape."""
    row = {
        "id": job_id, "query": query,
        "thesis": {"key_themes": ["Fintech"], "risks": ["Regulatory"],
                   "investment_signals": [], "sources": [], "raw_output": "text",
                   "opportunity_score": 3.5, "confidence_level": 0.8,
                   "recommendation": "Pursue", "key_risk_factors": []},
        "refinement_count": 0,
        "refinement_status": "N/A", "feedback_history": [], "execution_log": [],
        "retrieved_docs": [
            {"page_content": "chunk", "metadata": {
                "title": "Article A", "url": "https://a.example",
                "published_at": "2026-06-01T00:00:00+00:00",
                "similarity": 0.8123}},
        ],
        "created_at": "2026-07-01T00:00:00+00:00", "approved_at": None,
        "query_embedding": None, "thesis_history": [],
    }
    row.update(overrides)
    return row


class TestAPIEndpoints:
    """Tests for FastAPI route handlers with mocked Supabase."""

    @pytest.fixture
    def client(self):
        """Create a test client by overriding FastAPI dependencies."""
        from fastapi.testclient import TestClient
        from api.routes import router
        from api.auth import AuthUser, get_current_user, get_user_job_manager
        from api.deps import get_container
        from fastapi import FastAPI

        # Job manager methods are async now -> AsyncMock so `await jm.x()` works.
        mock_jm = MagicMock()
        mock_jm.get_job = AsyncMock(return_value=None)
        mock_jm.list_jobs = AsyncMock(return_value=[])
        mock_jm.create_job = AsyncMock()
        mock_jm.update_job = AsyncMock()
        mock_jm.update_job_guarded = AsyncMock(return_value=True)
        mock_jm.match_jobs = AsyncMock(return_value=[])

        mock_container = MagicMock()

        test_app = FastAPI()
        test_app.include_router(router)
        # Override the per-request user-scoped manager (bypasses JWT auth in tests).
        test_app.dependency_overrides[get_user_job_manager] = lambda: mock_jm
        test_app.dependency_overrides[get_container] = lambda: mock_container
        # Some routes also depend on get_current_user directly (e.g. list scoping);
        # stand in a fixed non-admin caller so those handlers run without a JWT.
        test_app.dependency_overrides[get_current_user] = lambda: AuthUser(
            id="test-user", token="t", role="user"
        )

        self._mock_jm = mock_jm
        self._mock_container = mock_container
        return TestClient(test_app)

    def test_health_check(self, client):
        """Test health endpoint returns ok."""
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_get_nonexistent_job_returns_404(self, client):
        """Test that missing job returns 404 with a machine-readable code."""
        response = client.get("/api/theses/nonexistent")
        assert response.status_code == 404
        assert response.json()["detail"]["code"] == "job_not_found"

    def test_create_thesis_sync_returns_201(self, client):
        """Test that thesis creation runs synchronously and returns 201."""
        from api.supabase_job_manager import _RowProxy
        from core.models.thesis import StructuredThesis
        from langchain_core.documents import Document

        self._mock_container.get_embedding_model.return_value \
            .get_embeddings.return_value.embed_query.return_value = [0.1, 0.2]
        # All three Silver tag dimensions present, so the taxonomy-gap guard
        # lets the request through to generation.
        self._mock_container.get_retrieval_service.return_value \
            .retrieve.return_value = [Document(page_content="chunk", metadata={
                "themes": ["Payments"], "risks": ["Regulatory"], "signals": ["Growth"]})]
        self._mock_container.get_thesis_service.return_value \
            .generate_thesis = AsyncMock(return_value=StructuredThesis(
                key_themes=["Fintech"], recommendation="Pursue"))

        self._mock_jm.create_job.return_value = _RowProxy(_row())
        self._mock_jm.get_job.return_value = _RowProxy(_row())

        response = client.post("/api/theses", json={"query": "digital lending"})
        assert response.status_code == 201
        assert response.headers["location"] == "/api/theses/test123"
        data = response.json()
        assert data["job_id"] == "test123"
        assert data["thesis"]["key_themes"] == ["Fintech"]
        # ONE atomic write carries the full completed state and the embedding;
        # there is no separate update that could strand a half-written row.
        create_kwargs = self._mock_jm.create_job.call_args.kwargs
        assert create_kwargs["query_embedding"] == [0.1, 0.2]
        self._mock_jm.update_job.assert_not_called()

    def test_create_thesis_no_docs_returns_422(self, client):
        """Test that an empty retrieval creates no job and returns 422."""
        self._mock_container.get_retrieval_service.return_value \
            .retrieve.return_value = []
        response = client.post("/api/theses", json={"query": "digital lending"})
        assert response.status_code == 422
        assert response.json()["detail"]["code"] == "no_relevant_documents"
        self._mock_jm.create_job.assert_not_called()

    def test_get_thesis_full_representation(self, client):
        """Test that GET returns the full state including sources."""
        from api.supabase_job_manager import _RowProxy
        self._mock_jm.get_job.return_value = _RowProxy(_row())
        response = client.get("/api/theses/test123")
        assert response.status_code == 200
        data = response.json()
        assert data["sources"] == [{
            "title": "Article A", "url": "https://a.example",
            "published_at": "2026-06-01T00:00:00+00:00",
            "similarity": 0.8123}]
        assert data["thesis_history"] == []
        assert data["approved_at"] is None

    def test_list_theses(self, client):
        """Test listing returns slim summaries."""
        from api.supabase_job_manager import _RowProxy
        self._mock_jm.list_jobs.return_value = [_RowProxy(_row())]
        response = client.get("/api/theses")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["job_id"] == "test123"
        assert data[0]["opportunity_score"] == 3.5
        assert "thesis" not in data[0]

    def test_refine_escalated_returns_409(self, client):
        """Test that refining an escalated job is rejected as a conflict."""
        from api.supabase_job_manager import _RowProxy
        self._mock_jm.get_job.return_value = _RowProxy(
            _row(refinement_status="escalated", refinement_count=3))
        response = client.post(
            "/api/theses/test123/refinements", json={"feedback": ["Too broad"]})
        assert response.status_code == 409
        assert response.json()["detail"]["code"] == "max_refinements_reached"

    def test_refine_approved_returns_409(self, client):
        """Test that refining an approved thesis is rejected."""
        from api.supabase_job_manager import _RowProxy
        self._mock_jm.get_job.return_value = _RowProxy(
            _row(approved_at="2026-07-01T12:00:00+00:00"))
        response = client.post(
            "/api/theses/test123/refinements", json={"feedback": ["Too broad"]})
        assert response.status_code == 409
        assert response.json()["detail"]["code"] == "already_approved"

    def _refine_with_graph(self, client, result_status="refining", change_thesis=True):
        """POST a refinement with the graph mocked to one round.

        change_thesis=False models an escalate/skip round: the graph returns
        with the thesis untouched."""
        from dataclasses import replace

        from api.supabase_job_manager import _RowProxy

        self._mock_jm.get_job.return_value = _RowProxy(_row())

        async def fake_ainvoke(state, config=None):
            thesis = state["current_thesis"]
            if change_thesis:
                thesis = replace(thesis, raw_output="rewritten narrative")
            return {
                "current_thesis": thesis,
                "refinement_count": state["refinement_count"] + 1,
                "status": result_status,
                "feedback_history": state["feedback_history"],
                "execution_log": [],
                "messages": [],
            }

        graph = MagicMock()
        graph.ainvoke = AsyncMock(side_effect=fake_ainvoke)
        self._mock_container.get_refinement_graph.return_value = (graph, None)
        return client.post(
            "/api/theses/test123/refinements", json={"feedback": ["Too broad"]})

    def test_refine_persists_guarded(self, client):
        """A refinement persists via the guarded update, keyed on the count it
        read and on the job being unapproved; the changed round snapshots the
        prior thesis into history."""
        response = self._refine_with_graph(client)
        assert response.status_code == 200
        args, kwargs = self._mock_jm.update_job_guarded.call_args
        assert args[0] == "test123"
        assert args[1] == {"refinement_count": 0, "approved_at": None}
        assert kwargs["refinement_count"] == 1
        assert kwargs["thesis"].raw_output == "rewritten narrative"
        # Previous Versions gains exactly the pre-refinement thesis.
        assert [t.raw_output for t in kwargs["thesis_history"]] == ["text"]
        self._mock_jm.update_job.assert_not_called()

    def test_refine_unchanged_round_adds_no_history_entry(self, client):
        """A round that leaves the thesis untouched (escalate, or a planner
        skip) must NOT snapshot a duplicate 'previous version'."""
        response = self._refine_with_graph(client, change_thesis=False)
        assert response.status_code == 200
        kwargs = self._mock_jm.update_job_guarded.call_args.kwargs
        assert kwargs["thesis_history"] == []

    def test_refine_lost_race_returns_409_conflict(self, client):
        """If an approval (or a second tab's refinement) lands while the agent
        runs, the guarded update matches no row and the round is discarded
        with a 409 instead of silently overwriting the newer state."""
        self._mock_jm.update_job_guarded = AsyncMock(return_value=False)
        response = self._refine_with_graph(client)
        assert response.status_code == 409
        assert response.json()["detail"]["code"] == "conflict"

    def test_approve_stamps_and_is_idempotent(self, client):
        """Test approval persists a timestamp once and re-approval is a no-op.

        refinement_status is decoupled from approval: a never-refined job (N/A)
        keeps its status untouched (only approved_at is written)."""
        from api.supabase_job_manager import _RowProxy
        self._mock_jm.get_job.return_value = _RowProxy(_row())
        response = client.put("/api/theses/test123/approval")
        assert response.status_code == 200
        update_kwargs = self._mock_jm.update_job.call_args.kwargs
        assert "refinement_status" not in update_kwargs
        assert update_kwargs["approved_at"]

        # Already approved: no further write, same state returned.
        self._mock_jm.update_job.reset_mock()
        self._mock_jm.get_job.return_value = _RowProxy(
            _row(approved_at="2026-07-01T12:00:00+00:00"))
        response = client.put("/api/theses/test123/approval")
        assert response.status_code == 200
        assert response.json()["approved_at"] == "2026-07-01T12:00:00+00:00"
        self._mock_jm.update_job.assert_not_called()

    def test_approve_mid_refinement_finalizes_status(self, client):
        """Approving a job that is mid-refinement finalizes refinement_status
        to "refined" alongside the approval timestamp."""
        from api.supabase_job_manager import _RowProxy
        self._mock_jm.get_job.return_value = _RowProxy(
            _row(refinement_status="refining", refinement_count=1))
        response = client.put("/api/theses/test123/approval")
        assert response.status_code == 200
        update_kwargs = self._mock_jm.update_job.call_args.kwargs
        assert update_kwargs["refinement_status"] == "refined"
        assert update_kwargs["approved_at"]

    def test_feedback_options(self, client):
        """Test the feedback options endpoint serves the configured list."""
        from config.settings import FEEDBACK_OPTIONS
        response = client.get("/api/feedback-options")
        assert response.status_code == 200
        assert response.json() == FEEDBACK_OPTIONS

    def test_list_theses_status_filter_forwarded(self, client):
        """A valid status filter is forwarded to the job manager query."""
        self._mock_jm.list_jobs.return_value = []
        response = client.get("/api/theses?status=refining")
        assert response.status_code == 200
        assert self._mock_jm.list_jobs.call_args.kwargs["status"] == "refining"

    def test_list_theses_scopes_to_caller_by_default(self, client):
        """Without ?all, the list is scoped to the caller's own user_id."""
        self._mock_jm.list_jobs.return_value = []
        response = client.get("/api/theses")
        assert response.status_code == 200
        assert self._mock_jm.list_jobs.call_args.kwargs["user_id"] == "test-user"

    def test_list_theses_all_forbidden_for_non_admin(self, client):
        """A non-admin asking for all=true is refused, not silently narrowed."""
        response = client.get("/api/theses?all=true")
        assert response.status_code == 403
        assert response.json()["detail"]["code"] == "forbidden"

    def test_list_theses_all_excludes_admins_own_for_admin(self):
        """An admin with ?all=true lists every OTHER user's theses: no owner
        filter, but the admin's own rows are excluded so the cross-user view
        doesn't duplicate their personal library."""
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        from api.routes import router
        from api.auth import AuthUser, get_current_user, get_user_job_manager
        from api.deps import get_container

        mock_jm = MagicMock()
        mock_jm.list_jobs = AsyncMock(return_value=[])
        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_user_job_manager] = lambda: mock_jm
        app.dependency_overrides[get_container] = lambda: MagicMock()
        app.dependency_overrides[get_current_user] = lambda: AuthUser(
            id="admin-user", token="t", role="admin"
        )
        response = TestClient(app).get("/api/theses?all=true")
        assert response.status_code == 200
        kwargs = mock_jm.list_jobs.call_args.kwargs
        assert kwargs["user_id"] is None
        assert kwargs["exclude_user_id"] == "admin-user"

    def test_list_theses_rejects_unknown_status(self, client):
        """An unknown status value is rejected by enum validation (422)."""
        response = client.get("/api/theses?status=bogus")
        assert response.status_code == 422
