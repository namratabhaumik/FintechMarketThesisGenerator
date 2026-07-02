"""Unit tests for the FastAPI layer with Supabase-backed job manager."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from api.schemas import JobStatus, ThesisRequest, RefinementRequest


class TestSupabaseJobManager:
    """Tests for SupabaseJobManager with mocked Supabase client."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock Supabase client."""
        client = MagicMock()
        # Chain: client.table(TABLE).insert(row).execute()
        client.table.return_value.insert.return_value.execute.return_value = Mock(data=None)
        return client

    @pytest.fixture
    def jm(self, mock_client):
        """Create a SupabaseJobManager with mocked client."""
        with patch("api.supabase_job_manager.create_client", return_value=mock_client):
            from api.supabase_job_manager import SupabaseJobManager
            return SupabaseJobManager(url="https://fake.supabase.co", service_role_key="fake-key")

    def test_create_job(self, jm, mock_client):
        """Test creating a new job inserts a row and returns a proxy."""
        job = jm.create_job("digital lending")
        assert job.id is not None
        assert len(job.id) == 12
        assert job.query == "digital lending"
        assert job.status == JobStatus.PENDING
        mock_client.table.return_value.insert.assert_called_once()

    def test_get_job_found(self, jm, mock_client):
        """Test retrieving an existing job."""
        mock_client.table.return_value.select.return_value.eq.return_value.maybe_single.return_value.execute.return_value = Mock(
            data={"id": "abc123", "query": "neobanking", "status": "completed", "progress": "Done", "thesis": None, "articles": [], "error": None}
        )
        job = jm.get_job("abc123")
        assert job is not None
        assert job.id == "abc123"
        assert job.status == JobStatus.COMPLETED

    def test_get_job_not_found(self, jm, mock_client):
        """Test that missing job returns None."""
        mock_client.table.return_value.select.return_value.eq.return_value.maybe_single.return_value.execute.return_value = Mock(data=None)
        assert jm.get_job("nonexistent") is None

    def test_update_status(self, jm, mock_client):
        """Test updating job status calls update on the table."""
        jm.update_status("abc123", JobStatus.GENERATING, "Generating thesis...")
        mock_client.table.return_value.update.assert_called_once_with(
            {"status": "generating", "progress": "Generating thesis..."}
        )

    def test_update_job_serialises_thesis(self, jm, mock_client):
        """Test that update_job serialises StructuredThesis to dict."""
        from core.models.thesis import StructuredThesis
        thesis = StructuredThesis(key_themes=["AI"], risks=["Regulatory"])
        jm.update_job("abc123", thesis=thesis)
        call_args = mock_client.table.return_value.update.call_args[0][0]
        assert call_args["thesis"]["key_themes"] == ["AI"]
        assert call_args["thesis"]["risks"] == ["Regulatory"]

    def test_list_jobs(self, jm, mock_client):
        """Test listing jobs returns proxies ordered by created_at desc."""
        mock_client.table.return_value.select.return_value.order.return_value.execute.return_value = Mock(
            data=[
                {"id": "job2", "query": "second", "status": "pending", "articles": []},
                {"id": "job1", "query": "first", "status": "completed", "articles": []},
            ]
        )
        jobs = jm.list_jobs()
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
        from api.deps import get_container, get_job_manager
        from fastapi import FastAPI

        mock_jm = MagicMock()
        mock_jm.get_job.return_value = None
        mock_jm.list_jobs.return_value = []

        mock_container = MagicMock()

        test_app = FastAPI()
        test_app.include_router(router)
        test_app.dependency_overrides[get_job_manager] = lambda: mock_jm
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
            .generate_thesis.return_value = StructuredThesis(
                key_themes=["Fintech"], recommendation="Pursue")

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
        """Test approval persists a timestamp once and re-approval is a no-op."""
        from api.supabase_job_manager import _RowProxy
        self._mock_jm.get_job.return_value = _RowProxy(_row())
        response = client.put("/api/theses/test123/approval")
        assert response.status_code == 200
        update_kwargs = self._mock_jm.update_job.call_args.kwargs
        assert update_kwargs["refinement_status"] == "refined"
        assert update_kwargs["approved_at"]

        # Already approved: no further write, same state returned.
        self._mock_jm.update_job.reset_mock()
        self._mock_jm.get_job.return_value = _RowProxy(
            _row(approved_at="2026-07-01T12:00:00+00:00"))
        response = client.put("/api/theses/test123/approval")
        assert response.status_code == 200
        assert response.json()["approved_at"] == "2026-07-01T12:00:00+00:00"
        self._mock_jm.update_job.assert_not_called()

    def test_feedback_options(self, client):
        """Test the feedback options endpoint serves the configured list."""
        from config.settings import FEEDBACK_OPTIONS
        response = client.get("/api/feedback-options")
        assert response.status_code == 200
        assert response.json() == FEEDBACK_OPTIONS
