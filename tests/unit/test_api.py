"""Unit tests for the FastAPI layer with Supabase-backed job manager."""

import asyncio

import pytest
from unittest.mock import AsyncMock, MagicMock, Mock

from api.schemas import JobStatus, ThesisRequest, RefinementRequest


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
        assert job.status == JobStatus.PENDING
        mock_client.table.return_value.insert.assert_called_once()

    def test_get_job_found(self, jm, mock_client):
        """Test retrieving an existing job."""
        mock_client.table.return_value.select.return_value.eq.return_value.maybe_single.return_value.execute = AsyncMock(
            return_value=Mock(
                data={"id": "abc123", "query": "neobanking", "status": "completed", "progress": "Done", "thesis": None, "articles": [], "error": None}
            )
        )
        job = asyncio.run(jm.get_job("abc123"))
        assert job is not None
        assert job.id == "abc123"
        assert job.status == JobStatus.COMPLETED

    def test_get_job_not_found(self, jm, mock_client):
        """Test that missing job returns None."""
        mock_client.table.return_value.select.return_value.eq.return_value.maybe_single.return_value.execute = AsyncMock(
            return_value=Mock(data=None)
        )
        assert asyncio.run(jm.get_job("nonexistent")) is None

    def test_update_status(self, jm, mock_client):
        """Test updating job status calls update on the table."""
        mock_client.table.return_value.update.return_value.eq.return_value.execute = AsyncMock(
            return_value=Mock(data=None)
        )
        asyncio.run(jm.update_status("abc123", JobStatus.GENERATING, "Generating thesis..."))
        mock_client.table.return_value.update.assert_called_once_with(
            {"status": "generating", "progress": "Generating thesis..."}
        )

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

    def test_list_jobs(self, jm, mock_client):
        """Test listing jobs returns proxies ordered by created_at desc."""
        mock_client.table.return_value.select.return_value.order.return_value.execute = AsyncMock(
            return_value=Mock(
                data=[
                    {"id": "job2", "query": "second", "status": "pending", "articles": []},
                    {"id": "job1", "query": "first", "status": "completed", "articles": []},
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
            "articles": [],
        }
        proxy = _RowProxy(row)
        assert proxy.thesis is not None
        assert proxy.thesis.key_themes == ["Fintech"]
        assert proxy.thesis.opportunity_score == 3.5

    def test_rehydrates_articles(self):
        """Test that stored article dicts become Article objects."""
        from api.supabase_job_manager import _RowProxy
        row = {
            "id": "test2", "query": "q", "status": "pending",
            "thesis": None,
            "articles": [
                {"title": "Test Article", "text": "Content here", "source": "techcrunch.com", "url": "https://example.com", "published_at": "2026-01-01T00:00:00+00:00"},
            ],
        }
        proxy = _RowProxy(row)
        assert len(proxy.articles) == 1
        assert proxy.articles[0].title == "Test Article"
        assert proxy.articles[0].published_at.year == 2026

    def test_null_thesis_stays_none(self):
        """Test that null thesis in DB stays None."""
        from api.supabase_job_manager import _RowProxy
        row = {"id": "test3", "query": "q", "status": "pending", "thesis": None, "articles": []}
        proxy = _RowProxy(row)
        assert proxy.thesis is None


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

    def test_job_status_enum_values(self):
        """Test that all expected job statuses exist."""
        assert JobStatus.PENDING == "pending"
        assert JobStatus.COMPLETED == "completed"
        assert JobStatus.FAILED == "failed"
        assert JobStatus.FETCHING_ARTICLES == "fetching_articles"
        assert JobStatus.GENERATING == "generating"
        assert JobStatus.REFINING == "refining"

    def test_refinement_status_coerces_and_rejects(self):
        """refinement_status is a str-enum: valid strings from storage coerce to
        the enum and serialise back to the wire string; unknown values reject."""
        from api.schemas import JobResponse, RefinementStatus
        job = JobResponse(job_id="j", query="q", status=JobStatus.COMPLETED,
                          refinement_status="refining")
        assert job.refinement_status == RefinementStatus.REFINING
        assert job.model_dump(mode="json")["refinement_status"] == "refining"
        with pytest.raises(Exception):
            JobResponse(job_id="j", query="q", status=JobStatus.COMPLETED,
                        refinement_status="bogus")


class TestSerializers:
    """Tests for serialise_job_fields enum unwrapping."""

    def test_refinement_status_enum_unwrapped_for_storage(self):
        """A RefinementStatus enum is stored as its string value, like status."""
        from api.schemas import RefinementStatus
        from api.serializers import serialise_job_fields
        payload = serialise_job_fields(refinement_status=RefinementStatus.REFINED)
        assert payload["refinement_status"] == "refined"


def _row(job_id="test123", query="digital lending", **overrides):
    """A completed job row dict in the stored (Supabase) shape."""
    row = {
        "id": job_id, "query": query, "status": "completed",
        "thesis": {"key_themes": ["Fintech"], "risks": ["Regulatory"],
                   "investment_signals": [], "sources": [], "raw_output": "text",
                   "opportunity_score": 3.5, "confidence_level": 0.8,
                   "recommendation": "Pursue", "key_risk_factors": []},
        "articles": [], "error": None, "refinement_count": 0,
        "refinement_status": "N/A", "feedback_history": [], "execution_log": [],
        "retrieved_docs": [
            {"page_content": "chunk", "metadata": {
                "title": "Article A", "url": "https://a.example",
                "published_at": "2026-06-01T00:00:00+00:00"}},
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
        from api.auth import get_user_job_manager
        from api.deps import get_container
        from fastapi import FastAPI

        # Job manager methods are async now -> AsyncMock so `await jm.x()` works.
        mock_jm = MagicMock()
        mock_jm.get_job = AsyncMock(return_value=None)
        mock_jm.list_jobs = AsyncMock(return_value=[])
        mock_jm.create_job = AsyncMock()
        mock_jm.update_job = AsyncMock()
        mock_jm.match_jobs = AsyncMock(return_value=[])

        mock_container = MagicMock()

        test_app = FastAPI()
        test_app.include_router(router)
        # Override the per-request user-scoped manager (bypasses JWT auth in tests).
        test_app.dependency_overrides[get_user_job_manager] = lambda: mock_jm
        test_app.dependency_overrides[get_container] = lambda: mock_container

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
        self._mock_container.get_retrieval_service.return_value \
            .retrieve.return_value = [Document(page_content="chunk", metadata={})]
        self._mock_container.get_thesis_service.return_value \
            .generate_thesis = AsyncMock(return_value=StructuredThesis(
                key_themes=["Fintech"], recommendation="Pursue"))

        self._mock_jm.create_job.return_value = _RowProxy(
            _row(thesis=None, retrieved_docs=[], status="pending"))
        self._mock_jm.get_job.return_value = _RowProxy(_row())

        response = client.post("/api/theses", json={"query": "digital lending"})
        assert response.status_code == 201
        assert response.headers["location"] == "/api/theses/test123"
        data = response.json()
        assert data["job_id"] == "test123"
        assert data["status"] == "completed"
        assert data["thesis"]["key_themes"] == ["Fintech"]
        # The persisted write carries the completed state and the embedding.
        update_kwargs = self._mock_jm.update_job.call_args.kwargs
        assert update_kwargs["status"] == JobStatus.COMPLETED
        assert update_kwargs["query_embedding"] == [0.1, 0.2]

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
            "published_at": "2026-06-01T00:00:00+00:00"}]
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

    def test_list_theses_rejects_unknown_status(self, client):
        """An unknown status value is rejected by enum validation (422)."""
        response = client.get("/api/theses?status=bogus")
        assert response.status_code == 422
