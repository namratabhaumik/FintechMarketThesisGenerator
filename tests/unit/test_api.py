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
            return SupabaseJobManager(url="https://fake.supabase.co", anon_key="fake-key")

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
                {"title": "Test Article", "text": "Content here", "source": "techcrunch.com", "url": "https://example.com"},
            ],
        }
        proxy = _RowProxy(row)
        assert len(proxy.articles) == 1
        assert proxy.articles[0].title == "Test Article"

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
        return TestClient(test_app)

    def test_health_check(self, client):
        """Test health endpoint returns ok."""
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_get_nonexistent_job_returns_404(self, client):
        """Test that missing job returns 404."""
        response = client.get("/api/thesis/nonexistent")
        assert response.status_code == 404

    def test_create_thesis_returns_202(self, client):
        """Test that thesis creation returns 202 with job_id."""
        from api.supabase_job_manager import _RowProxy
        self._mock_jm.create_job.return_value = _RowProxy({
            "id": "test123", "query": "digital lending", "status": "pending",
            "thesis": None, "articles": [], "progress": None, "error": None,
        })
        response = client.post("/api/thesis", json={"query": "digital lending"})
        assert response.status_code == 202
        data = response.json()
        assert data["job_id"] == "test123"
        assert data["status"] == "pending"

    def test_list_jobs(self, client):
        """Test listing jobs returns empty list."""
        response = client.get("/api/jobs")
        assert response.status_code == 200
        assert response.json() == []
