"""Unit tests for the FastAPI layer."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from api.job_manager import JobManager
from api.schemas import JobStatus, ThesisRequest, RefinementRequest


class TestJobManager:
    """Tests for the in-memory job manager."""

    def test_create_job(self):
        """Test creating a new job."""
        jm = JobManager()
        job = jm.create_job("digital lending")
        assert job.id is not None
        assert len(job.id) == 12
        assert job.query == "digital lending"
        assert job.status == JobStatus.PENDING

    def test_get_job(self):
        """Test retrieving a job by ID."""
        jm = JobManager()
        job = jm.create_job("test query")
        retrieved = jm.get_job(job.id)
        assert retrieved is not None
        assert retrieved.id == job.id

    def test_get_nonexistent_job(self):
        """Test that missing job returns None."""
        jm = JobManager()
        assert jm.get_job("nonexistent") is None

    def test_update_status(self):
        """Test updating job status."""
        jm = JobManager()
        job = jm.create_job("test")
        jm.update_status(job.id, JobStatus.FETCHING_ARTICLES, "Fetching...")
        updated = jm.get_job(job.id)
        assert updated.status == JobStatus.FETCHING_ARTICLES
        assert updated.progress == "Fetching..."

    def test_list_jobs_ordered_recent_first(self):
        """Test that jobs are listed most recent first."""
        jm = JobManager()
        job1 = jm.create_job("first")
        job2 = jm.create_job("second")
        jobs = jm.list_jobs()
        assert len(jobs) == 2
        assert jobs[0].id == job2.id
        assert jobs[1].id == job1.id


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
    """Tests for FastAPI route handlers using TestClient."""

    @pytest.fixture
    def client(self):
        """Create a test client with mocked dependencies."""
        from fastapi.testclient import TestClient
        from api.main import app
        return TestClient(app)

    def test_health_check(self, client):
        """Test health endpoint returns ok."""
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_create_thesis_returns_202(self, client):
        """Test that thesis creation returns 202 with job_id."""
        response = client.post("/api/thesis", json={"query": "digital lending"})
        assert response.status_code == 202
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "pending"
        assert data["query"] == "digital lending"

    def test_get_nonexistent_job_returns_404(self, client):
        """Test that missing job returns 404."""
        response = client.get("/api/thesis/nonexistent")
        assert response.status_code == 404

    def test_list_jobs(self, client):
        """Test listing jobs."""
        # Create a job first
        client.post("/api/thesis", json={"query": "test"})
        response = client.get("/api/jobs")
        assert response.status_code == 200
        jobs = response.json()
        assert len(jobs) >= 1

    def test_refine_nonexistent_job_returns_404(self, client):
        """Test that refining a nonexistent job returns 404."""
        response = client.post(
            "/api/thesis/nonexistent/refine",
            json={"feedback": ["Too broad"]},
        )
        assert response.status_code == 404
